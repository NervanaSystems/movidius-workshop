#!/usr/bin/env python3
# Copyright(c) 2018 Intel Corporation.

import logging
import os
import queue
import time
import uuid

import cv2
import numpy
from flask import Flask, jsonify, render_template, request
from mvnc import mvncapi as mvnc2

FORMAT = '[%(asctime)-11s %(name)s] %(message)s'
logging.basicConfig(filename='mvnc2.log', level=logging.INFO, format=FORMAT)

class WebApp():
    def __init__(self, ):
        logging.info("Initializing web api...")
        self.image_storage_dir = "request_images"
        if not os.path.exists(self.image_storage_dir):
            os.mkdir(self.image_storage_dir)

        self.requests_number = 0
        self.app = Flask(__name__)
        self.setup_flask()

        self.devices_queue = queue.Queue()
        self.setup_device_queue()
        logging.info("Web api initialization done.")

    def setup_flask(self):
        start_time = time.time()
        self.app.add_url_rule('/', 'index', self.start_page, methods=['GET'])
        self.app.add_url_rule('/recognize_digit', 'recognize_digit', self.recognize_digit, methods=['GET', 'POST'])
        self.app.add_url_rule('/classify_picture', 'classify_picture', self.classify_picture, methods=['GET', 'POST'])
        setup_time = (time.time() - start_time) * 1000
        logging.info("Flask setup done in %f ms.", setup_time)

    def setup_device_queue(self):
        start_time = time.time()
        mvc2_devices = mvnc2.enumerate_devices()
        logging.info("Found %d devices.", len(mvc2_devices))

        device_id = 0
        for mvc2_device in mvc2_devices:
            dev = mvnc2.Device(mvc2_device)
            dev.open()
            self.devices_queue.put({"device": dev,
                               "device_id": device_id})
            device_id += 1

        setup_time = (time.time() - start_time) * 1000
        logging.info("Movidius devices setup done in %f ms.", setup_time)

    def run(self, host="0.0.0.0", port=5000, threaded=False, debug=False):
        self.app.run(host=host, port=port, threaded=threaded, debug=debug)

    def response(self, app_request, model):
        if app_request.method == 'POST':
            response_start_time = time.time()
            self.requests_number += 1
            request_ip = app_request.remote_addr
            logger = logging.getLogger("{} {}".format(self.requests_number, request_ip))
            request_file = app_request.files['the_file']
            image_path = os.path.join(self.image_storage_dir, str(uuid.uuid4()))
            request_file.save(image_path)

            in_queue_start_time = time.time()
            device_instance = self.devices_queue.get()
            in_queue_time = (time.time() - in_queue_start_time) * 1000
            logger.info("Spent %f ms in queue.", in_queue_time)

            device = device_instance.get("device")
            device_id = device_instance.get("device_id")
            logger.info("Running on device: %d", device_id)
            inference = Inference(device, model, logger)

            try:
                inference.run(image_path)
                probabilities = inference.get_propabilities()
            except Exception as ex:
                logger.error(str(ex))
                return jsonify({'status': False, 'message': str(ex)})
            finally:
                os.remove(image_path)
                inference.cleanup()
                self.devices_queue.put(device_instance)
                logger.info("Device %d returned to queue.", device_id)

            response_time = (time.time() - response_start_time) * 1000
            logger.info("Request response time: %f ms.", response_time)
            return jsonify({"status": True, "message": "", "probabilities": probabilities})

        return jsonify({'status': False, 'message': "Request method unsupported"})

    def start_page(self):
        return render_template("index.html")

    def recognize_digit(self):
        return self.response(request, "mnist")

    def classify_picture(self):
        return self.response(request, "inception_v1")


class Model():
    def load_categories(self):
        raise NotImplementedError("Not implemented.")

    def load_data(self, image_path: str):
        raise NotImplementedError("Not implemented.")


class InceptionV1(Model):
    def __init__(self):
        self.image_dimensions = (224, 224)
        self.graph_path = "inception_v1/inception_v1.graph"
        self.categories_path = "inception_v1/categories.txt"

    def load_categories(self):
        #Load categories
        categories = []
        with open(self.categories_path, 'r') as categories_file:
            for line in categories_file:
                cat = line.split('\n')[0]
                if cat != 'classes':
                    categories.append(cat)

        return categories

    def load_data(self, image_path: str):
        #Load preprocessing data
        mean = 128
        std = 1.0/128.0
        img = cv2.imread(image_path).astype(numpy.float32)

        dx, dy, _ = img.shape
        delta = float(abs(dy-dx))
        if dx > dy: #crop the x dimension
            img = img[int(0.5*delta):dx-int(0.5*delta), 0:dy]
        else:
            img = img[0:dx, int(0.5*delta):dy-int(0.5*delta)]

        img = cv2.resize(img, self.image_dimensions)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(3):
            img[:, :, i] = (img[:, :, i] - mean) * std

        return img


class Mnist(Model):
    def __init__(self):
        self.image_dimensions = (28, 28)
        self.graph_path = "mnist/mnist.graph"

    def load_categories(self):
        #Load categories
        categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        return categories

    def load_data(self, image_path: str):
        # Load image from disk and preprocess it to prepare it for the network assuming we are reading a .jpg or .png
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.image_dimensions)
        img = img.astype(numpy.float32)
        img[:] = ((img[:]) * (1.0/255.0))
        return img


class Inference():
    """
    Inference class.
    """
    initialization_queue = queue.Queue(maxsize=1)

    def __init__(self, device, model_name, logger):
        """
        Creates and opens the Neural Compute device and creates a graph that can execute inferences on it.
        """
        if device is None:
            raise Exception("No devices found.")
        else:
            self.device = device

        self.logger = logger

        # Init model
        self.logger.info("Initializing %s model.", model_name)
        if model_name.lower() == "mnist":
            self.model = Mnist()
        elif model_name.lower() == "inception_v1":
            self.model = InceptionV1()
        else:
            raise Exception("Model not recognized. Please use 'mnist' or 'incetpion_v1'.")

        graph_file_path = self.model.graph_path
        # Load graph file
        start_time = time.time()
        try:
            with open(graph_file_path, mode="rb") as graph_file:
                in_memory_graph = graph_file.read()
        except Exception:
            self.logger.error("Error reading graph file: %s.", graph_file_path)
            raise

        self.graph = None
        self.input_fifo = None
        self.output_fifo = None

        self.initialization_queue.put(0)
        self.graph = mvnc2.Graph("mvnc2 graph")
        self.input_fifo, self.output_fifo = self.graph.allocate_with_fifos(self.device, in_memory_graph,
                                                                           input_fifo_data_type=mvnc2.FifoDataType.FP16,
                                                                           output_fifo_data_type=mvnc2.FifoDataType.FP16)
        _ = self.initialization_queue.get()
        graph_alloc_time = (time.time() - start_time) * 1000
        self.logger.info("Graph allocated in %f ms.", graph_alloc_time)

        if self.graph is None or self.input_fifo is None or self.output_fifo is None:
            raise Exception("Could not initialize device.")

        self.inference_results = 0

    def run(self, image_path: str):
        data_start = time.time()
        data = self.model.load_data(image_path)
        data_load_time = (time.time() - data_start) * 1000
        self.logger.info("Data loaded in %f.", data_load_time)

        # Start the inference by sending to the device/graph
        start_time = time.time()
        self.graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, data.astype(numpy.float16), None)

        # Get the result from the device/graph.
        self.inference_results, _ = self.output_fifo.read_elem()
        execution_time = (time.time() - start_time) * 1000
        self.logger.info("Evaluation done in %f ms.", execution_time)

    def get_propabilities(self, number_results: int = 5):
        # Get dataset categories
        categories = self.model.load_categories()

        # Sort indices in order of highest probabilities
        top_inds = (-self.inference_results).argsort()[:number_results]

        # Get the labels and probabilities for the top results from the inference
        inference_top_inds = []
        inference_categories = []
        inference_probabilities = []

        for index in range(0, number_results):
            inference_top_inds.append(str(top_inds[index]))
            inference_categories.append(categories[top_inds[index]])
            inference_probabilities.append(str(self.inference_results[top_inds[index]]))

        results = {
            "inference_top_inds": inference_top_inds,
            "inference_categories": inference_categories,
            "inference_probabilities": inference_probabilities
        }
        return results
        #return inference_top_inds, inference_categories, inference_probabilities

    def cleanup(self):
        """ Cleans up the NCAPI resources. """
        if self.input_fifo:
            self.input_fifo.destroy()

        if self.output_fifo:
            self.output_fifo.destroy()

        if self.graph:
            self.graph.destroy()


if __name__ == "__main__":
    web_api = WebApp()
    web_api.run(threaded=True)
