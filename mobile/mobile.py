#!/usr/bin/env python3
# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import logging
import os
import queue
import time
import uuid

import cv2
import numpy
from flask import Flask, jsonify, render_template, request
from mvnc import mvncapi as mvnc2

from models import model_factory

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

    def response(self, app_request, model=None):
        if app_request.method == "POST":
            response_start_time = time.time()
            self.requests_number += 1
            request_ip = app_request.remote_addr
            logger = logging.getLogger("{} {}".format(self.requests_number, request_ip))
            request_file = app_request.files["the_file"]
            if model is None:
                model = app_request.form["model"]

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
            return render_template("result.html", model=model, probabilities=probabilities)

        return jsonify({'status': False, 'message': "Request method unsupported"})

    def start_page(self):
        return render_template("index.html")

    def recognize_digit(self):
        return self.response(request, "mnist")

    def classify_picture(self):
        return self.response(request)


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
        self.model = model_factory.get_model(model_name.lower())

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
        if number_results >= len(categories):
            number_results = len(categories) - 1

        # Sort indices in order of highest probabilities
        top_inds = (-self.inference_results).argsort()[:number_results]

        # Get the labels and probabilities for the top results from the inference
        # inference_top_inds = []
        # inference_categories = []
        # inference_probabilities = []

        results = []
        # Get the labels and probabilities for the top results from the inference
        for index in range(0, number_results):
            results.append({"category_id": str(top_inds[index]),
                            "category_name": categories[top_inds[index]],
                            "probability": str(self.inference_results[top_inds[index]])})

            #inference_top_inds.append(str(top_inds[index]))
            #inference_categories.append(categories[top_inds[index]])
            #inference_probabilities.append(str(self.inference_results[top_inds[index]]))

        # results = {
        #     "inference_top_inds": inference_top_inds,
        #     "inference_categories": inference_categories,
        #     "inference_probabilities": inference_probabilities
        # }
        return results

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
