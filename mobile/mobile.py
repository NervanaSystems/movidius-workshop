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

logging.basicConfig(filename='mvnc2.log', level=logging.INFO)
DEVICES = queue.Queue()
app = Flask(__name__)

@app.route('/', methods=['GET'])
def start_page():
    return render_template("index.html")

@app.route('/recognize_digit', methods=['GET', 'POST'])
def recognize_digit():
    if request.method == 'POST':
        request_file = request.files['the_file']
        image_name = str(uuid.uuid4())
        request_file.save(image_name)
        device = None
        while not device:
            time.sleep(1)
            device = DEVICES.get()

        inference = Inference(device, "mnist")

        try:
            inference.run(image_name)
            probabilities = inference.get_propabilities()
        except Exception as ex:
            logging.error(str(ex))
            return jsonify({'status': False, 'message': str(ex)})
        finally:
            os.remove(image_name)
            DEVICES.put(device)

        return jsonify({'status': True, 'message': ""} + probabilities)

    return jsonify({'status': False, 'message': "Request method unsupported"})

@app.route('/classify_picture', methods=['GET', 'POST'])
def classify_picture():
    if request.method == 'POST':
        request_file = request.files['the_file']
        image_name = str(uuid.uuid4())
        request_file.save(image_name)
        device = None
        while not device:
            time.sleep(1)
            device = DEVICES.get()

        inference = Inference(device, "inception_v1")

        try:
            inference.run(image_name)
            probabilities = inference.get_propabilities()
        except Exception as ex:
            logging.error(str(ex))
            return jsonify({'status': False, 'message': str(ex)})
        finally:
            os.remove(image_name)
            DEVICES.put(device)

        return jsonify({'status': True, 'message': ""} + probabilities)

    return jsonify({'status': False, 'message': "Request method unsupported"})

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

        logging.info('Number of categories: %d', len(categories))
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
        logging.info('Number of categories: %d', len(categories))
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

    def __init__(self, device, model_name):
        """
        Creates and opens the Neural Compute device and creates a graph that can execute inferences on it.
        """
        if device is None:
            raise Exception("No devices found.")
        else:
            self.device = device

        # Open the NCS
        try:
            self.device.open()
        except Exception():
            logging.error("Error opening device.")
            raise

        # Init model
        if model_name.lower() == "mnist":
            self.model = Mnist()
        elif model_name.lower() == "inception_v1":
            self.model = InceptionV1()
        else:
            raise Exception("Model not recognized. Please use 'mnist' or 'incetpion_v1'.")

        graph_file_path = self.model.graph_path
        # Load graph file
        try:
            with open(graph_file_path, mode="rb") as graph_file:
                in_memory_graph = graph_file.read()
        except Exception():
            logging.error("Error reading graph file: %s.", graph_file_path)
            device.close()
            device.destroy()
            raise

        self.graph = mvnc2.Graph("mvnc2 graph")
        self.input_fifo, self.output_fifo = self.graph.allocate_with_fifos(self.device, in_memory_graph,
                                                                           input_fifo_data_type=mvnc2.FifoDataType.FP16,
                                                                           output_fifo_data_type=mvnc2.FifoDataType.FP16)

        if self.graph is None or self.input_fifo is None or self.output_fifo is None:
            raise Exception("Could not initialize device.")

    def run(self, image_path: str):
        data = self.model.load_data(image_path)

        # Start the inference by sending to the device/graph
        self.graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, data.astype(numpy.float16), None)

    def get_propabilities(self, number_results: int = 5):
        categories = self.model.load_categories()

        # Get the result from the device/graph.
        output, _ = self.output_fifo.read_elem()

        # Sort indices in order of highest probabilities
        top_inds = (-output).argsort()[:number_results]

        # Get the labels and probabilities for the top results from the inference
        inference_top_inds = []
        inference_categories = []
        inference_probabilities = []

        for index in range(0, number_results):
            inference_top_inds.append(str(top_inds[index]))
            inference_categories.append(categories[top_inds[index]])
            inference_probabilities.append(str(output[top_inds[index]]))

        results = {
            "inference_top_inds": inference_top_inds,
            "inference_categories": inference_categories,
            "inference_probabilities": inference_probabilities
        }
        return results
        #return inference_top_inds, inference_categories, inference_probabilities

    def cleanup(self):
        """ Cleans up the NCAPI resources. """
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        self.graph.destroy()
        self.device.close()
        self.device.destroy()


if __name__ == "__main__":
    # Get a list of ALL the sticks that are plugged in
    for mvc2_device in mvnc2.enumerate_devices():
        DEVICES.put(mvnc2.Device(mvc2_device))
