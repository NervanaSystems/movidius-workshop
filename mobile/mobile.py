# Copyright(c) 2018 Intel Corporation.

from flask import Flask, jsonify, request
import numpy, cv2, os, sys, uuid
from mvnc import mvncapi as mvnc2
from typing import List

app = Flask(__name__)

IMAGE_DIMENSIONS_FOR_MNIST = (28,28)
IMAGE_DIMENSIONS_FOR_INCEPTION = (224,224)
PATH_TO_CATEGORIES = 'inception_v1/categories.txt'

GRAPH_PATH_FOR_MNIST = 'mnist/mnist.graph'
GRAPH_PATH_FOR_INCEPTION = 'inception_v1/inception_v1.graph'


@app.route('/recognize_digit', methods=['GET', 'POST'])
def recognize_digit():
    if request.method == 'POST':
        f = request.files['the_file']
        image_name = str(uuid.uuid4())
        f.save(image_name)

        try:
            # Initialize the neural compute device via the NCAPI
            device, graph, input_fifo, output_fifo = do_initialize(GRAPH_PATH_FOR_MNIST)

            # Loop through all the input images and run inferences and show results
            infer_labels, infer_probabilities = do_inference_for_mnist(graph, input_fifo, output_fifo, image_name)

            # Clean up the NCAPI devices
            do_cleanup(device, graph, input_fifo, output_fifo)
        except Exception as ex:
            print(ex.message)
            return jsonify({'status': False, 'message': ex.message})
        finally:
            os.remove(image_name)

        return jsonify({'status': True, 'message': "", 'labels': infer_labels, 'prob': infer_probabilities})

    return jsonify({'status': False, 'message': "Request method unsupported"})

@app.route('/classify_picture', methods=['GET', 'POST'])
def classify_picture():
    if request.method == 'POST':
        f = request.files['the_file']
        image_name = str(uuid.uuid4())
        f.save(image_name)

        try:
            # Initialize the neural compute device via the NCAPI
            device, graph, input_fifo, output_fifo = do_initialize(GRAPH_PATH_FOR_INCEPTION)

            # Loop through all the input images and run inferences and show results
            top_inds, categories, probabilities = do_inference_for_inception_v1(graph, input_fifo, output_fifo, image_name)

            # Clean up the NCAPI devices
            do_cleanup(device, graph, input_fifo, output_fifo)
        except Exception as ex:
            print(ex.message)
            return jsonify({'status': False, 'message': ex.message})
        finally:
            os.remove(image_name)

        return jsonify({'status': True, 'message': "", 'top_inds': top_inds, 'categories': categories, 'prob': probabilities})

    return jsonify({'status': False, 'message': "Request method unsupported"})

def do_initialize(graph_path) -> (mvnc2.Device, mvnc2.Graph, mvnc2.Fifo, mvnc2.Fifo):
    """Creates and opens the Neural Compute device and creates a graph that can execute inferences on it. """

    # Get a list of ALL the sticks that are plugged in
    devices = mvnc2.enumerate_devices()
    if len(devices) == 0:
        ex = Exception()
        ex.message = 'No devices found'
        raise ex

    # Pick the first stick to run the network
    device = mvnc2.Device(devices[0])

    # Open the NCS
    try:
        device.open()
    except Exception as ex:
        ex.message = 'Error opening device'
        raise

    filefolder = os.path.dirname(os.path.realpath(__file__))
    graph_filename = os.path.join(filefolder, graph_path)

    # Load graph file
    try:
        with open(graph_filename, mode='rb') as f:
            in_memory_graph = f.read()
    except:
        device.close()
        device.destroy()
        ex.message = 'Error reading graph file: ' + graph_filename
        raise

    graph = mvnc2.Graph("mvnc2 graph")
    input_fifo, output_fifo = graph.allocate_with_fifos(device, in_memory_graph,
                                                        input_fifo_data_type=mvnc2.FifoDataType.FP16,
                                                        output_fifo_data_type=mvnc2.FifoDataType.FP16)

    if device == None or graph == None or input_fifo == None or output_fifo == None:
        ex = Exception()
        ex.message = 'Could not initialize device'
        raise ex

    return device, graph, input_fifo, output_fifo


def do_inference_for_mnist(graph: mvnc2.Graph, input_fifo: mvnc2.Fifo, output_fifo: mvnc2.Fifo, 
                 image_filename: str, number_results: int = 5) -> (List[str], List[numpy.float16]):
    """ Executes one inference which will determine the top classifications for an image file. """

    # Text labels for each of the possible classfications
    labels=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Load image from disk and preprocess it to prepare it for the network assuming we are reading a .jpg or .png
    image_for_inference = cv2.imread(image_filename)
    image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)
    image_for_inference=cv2.resize(image_for_inference, IMAGE_DIMENSIONS_FOR_MNIST)
    image_for_inference = image_for_inference.astype(numpy.float32)
    image_for_inference[:] = ((image_for_inference[:] ) * (1.0/255.0))

    # Start the inference by sending to the device/graph
    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, image_for_inference.astype(numpy.float16), None)

    # Get the result from the device/graph.
    output, userobj = output_fifo.read_elem()

    # Sort indices in order of highest probabilities
    top_inds = (-output).argsort()[:number_results]

    # Get the labels and probabilities for the top results from the inference
    inference_labels = []
    inference_probabilities = []

    for index in range(0, number_results):
        inference_labels.append(labels[top_inds[index]])
        inference_probabilities.append(str(output[top_inds[index]]))

    return inference_labels, inference_probabilities


def do_inference_for_inception_v1(graph: mvnc2.Graph, input_fifo: mvnc2.Fifo, output_fifo: mvnc2.Fifo, 
                 image_filename: str, number_results : int = 5) -> (List[numpy.int], List[str], List[numpy.float16]):
    """ Executes one inference which will determine the top classifications for an image file. """

    #Load preprocessing data
    mean = 128 
    std = 1.0/128.0 

    #Load categories
    categories = []
    with open(PATH_TO_CATEGORIES, 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
    
    img = cv2.imread(image_filename).astype(numpy.float32)

    dx,dy,dz = img.shape
    delta=float(abs(dy-dx))
    if dx > dy: #crop the x dimension
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
    else:
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
        
    img = cv2.resize(img, IMAGE_DIMENSIONS_FOR_INCEPTION)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean) * std

    # Start the inference by sending to the device/graph
    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img.astype(numpy.float16), None)

    # Get the result from the device/graph.
    probabilities, userobj = output_fifo.read_elem()

    # Sort indices in order of highest probabilities
    top_inds = (-probabilities).argsort()[:number_results]

    # Get the indices, categories and probabilities for the top results from the inference
    inference_top_inds = []
    inference_categories = []
    inference_probabilities = []

    for index in range(0, number_results):
        inference_top_inds.append(str(top_inds[index]))
        inference_categories.append(categories[top_inds[index]])
        inference_probabilities.append(str(probabilities[top_inds[index]]))

    return inference_top_inds, inference_categories, inference_probabilities


def do_cleanup(device: mvnc2.Device, graph: mvnc2.Graph, input_fifo: mvnc2.Fifo, output_fifo: mvnc2.Fifo) -> None:
    """ Cleans up the NCAPI resources. """

    input_fifo.destroy()
    output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()
