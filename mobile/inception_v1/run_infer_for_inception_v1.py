#! /usr/bin/env python3

# Copyright(c) 2018 Intel Corporation.

import numpy
import cv2
import os
import sys

from mvnc import mvncapi as mvnc2

from typing import List

PATH_TO_NETWORKS = './'
PATH_TO_CATEGORIES = 'categories.txt'
GRAPH_FILENAME = 'inception_v1.graph'
IMAGE_DIMENSIONS = (224, 224)

# Example: ./run_infer_for_inception_v1.py images/nps_electric_guitar.png

def do_initialize() -> (mvnc2.Device, mvnc2.Graph, mvnc2.Fifo, mvnc2.Fifo):
    """Creates and opens the Neural Compute device and creates a graph that 
       can execute inferences on it.

    Returns
    -------
    device : mvnc2.Device
        The opened device. Will be None if couldn't open Device.
    graph : mvnc2.Graph
        The allocated graph to use for inferences. Will be None if couldn't allocate graph.
    input_fifo : mvnc2.Fifo
        The input FIFO queue for network.
    output_fifo : mvnc2.Fifo
        The output FIFO queue for network.
    """
    # Get a list of ALL the sticks that are plugged in
    devices = mvnc2.enumerate_devices()
    if len(devices) == 0:
        print('Error - No devices found')
        return (None, None, None, None)

    # Pick the first stick to run the network
    device = mvnc2.Device(devices[0])

    # Open the NCS
    device.open()

    filefolder = os.path.dirname(os.path.realpath(__file__))
    graph_filename = os.path.join(filefolder, GRAPH_FILENAME)

    # Load graph file
    try :
        with open(graph_filename, mode='rb') as f:
            in_memory_graph = f.read()
    except :
        print ("Error reading graph file: " + graph_filename)

    graph = mvnc2.Graph("mvnc2 graph")
    input_fifo, output_fifo = graph.allocate_with_fifos(device, in_memory_graph,
                                                        input_fifo_data_type=mvnc2.FifoDataType.FP16,
                                                        output_fifo_data_type=mvnc2.FifoDataType.FP16)

    return device, graph, input_fifo, output_fifo


def do_inference(graph: mvnc2.Graph, input_fifo: mvnc2.Fifo, output_fifo: mvnc2.Fifo, 
                 image_filename: str, number_results : int = 5) -> (List[numpy.int], List[str], List[numpy.float16]):
    """ executes one inference which will determine the top classifications for an image file.

    Parameters
    ----------
    graph : Graph
        The graph to use for the inference.  This should be initialize prior to calling.
    input_fifo : mvnc2.Fifo
        The input FIFO queue for network.
    output_fifo : mvnc2.Fifo
        The output FIFO queue for network.
    image_filename : string
        The filename (full or relative path) to use as the input for the inference.
    number_results : int
        The number of results to return, defaults to 5

    Returns
    -------
    top_inds: List[numpy.int]
        The top category numbers for the inference.
    categories: List[str]
        The top categories for the inference. categories[i] corresponds to top_inds[i]
    probabilities: List[numpy.float16]
        The top probabilities for the inference. probabilities[i] corresponds to categories[i] 
    """

    #Load preprocessing data
    mean = 128 
    std = 1.0/128.0 

    #Load categories
    categories = []
    with open(PATH_TO_NETWORKS + PATH_TO_CATEGORIES, 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
        print('Number of categories:', len(categories))
    
    img = cv2.imread(image_filename).astype(numpy.float32)

    dx,dy,dz = img.shape
    delta=float(abs(dy-dx))
    if dx > dy: #crop the x dimension
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
    else:
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
        
    img = cv2.resize(img, IMAGE_DIMENSIONS)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean) * std

    print('Start download to NCS...')

    # Start the inference by sending to the device/graph
    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img.astype(numpy.float16), None)

    # Get the result from the device/graph.
    probabilities, userobj = output_fifo.read_elem()

    # Sort indices in order of highest probabilities
    top_inds = (-probabilities).argsort()[:number_results]

    return top_inds, categories, probabilities


def do_cleanup(device: mvnc2.Device, graph: mvnc2.Graph, input_fifo: mvnc2.Fifo, output_fifo: mvnc2.Fifo) -> None:
    """Cleans up the NCAPI resources.

    Parameters
    ----------
    device : mvncapi.Device
            Device instance that was initialized in the do_initialize method.
    graph : mvncapi.Graph
            Graph instance that was initialized in the do_initialize method.
    input_fifo : mvnc2.Fifo
            The input FIFO queue for network.
    output_fifo : mvnc2.Fifo
            The output FIFO queue for network.

    Returns
    -------
    None

    """
    input_fifo.destroy()
    output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()


def main():
    """ Main function, return an int for program return value

    Opens device, reads graph file, runs inferences on files in images
    subdirectory, prints results, closes device
    """

    if len(sys.argv) <= 1:
        print('Missing input image file')
        return 1

    input_path = sys.argv[1]

    if os.path.exists(input_path) == False:
        print('File not found')
        return 1

    # Initialize the neural compute device via the NCAPI
    device, graph, input_fifo, output_fifo = do_initialize()

    if (device == None or graph == None or input_fifo == None or output_fifo == None):
        print ("Could not initialize device.")
        quit(1)

    top_inds, categories, probabilities = do_inference(graph, input_fifo, output_fifo, input_path, 5)

    # Show inference results
    print(''.join(['-' for i in range(79)]))

    for i in range(5):
        print(top_inds[i], categories[top_inds[i]], probabilities[top_inds[i]])

    print(''.join(['-' for i in range(79)]))

    # Clean up the NCAPI devices
    do_cleanup(device, graph, input_fifo, output_fifo)


if __name__ == "__main__":
    sys.exit(main())
