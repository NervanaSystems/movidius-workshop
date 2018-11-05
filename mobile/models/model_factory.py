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
from datasets.datasets import Dataset, ImagenetData, MnistData, AgeData, GenderData


def get_model(model_name):
    models = {
        "mnist": Mnist(),
        "inception_v1": InceptionV1(),
        "inception_v4": InceptionV4(),
        "agenet": AgeNet(),
        "gendernet": GenderNet()
    }
    return models.get(model_name)


class Model():
    """
    Generic model class.
    """
    def __init__(self):
        self.dataset = Dataset()

    def load_categories(self):
        return self.dataset.categories

    def load_data(self, image_path: str):
        return self.dataset.load_data(image_path)


class Mnist(Model):
    def __init__(self):
        self.image_dimensions = (28, 28)
        self.graph_path = "models/mnist/mnist.graph"
        self.dataset = MnistData(self.image_dimensions)


class AgeNet(Model):
    def __init__(self):
        self.image_dimensions = (227, 227)
        self.graph_path = "models/age_gender/agenet.graph"
        self.dataset = AgeData(self.image_dimensions)


class GenderNet(Model):
    def __init__(self):
        self.image_dimensions = (227, 227)
        self.graph_path = "models/age_gender/gendernet.graph"
        self.dataset = GenderData(self.image_dimensions)


class InceptionV1(Model):
    def __init__(self):
        self.image_dimensions = (224, 224)
        self.graph_path = "models/inception_v1/inception_v1.graph"
        self.dataset = ImagenetData(self.image_dimensions)


class InceptionV4(Model):
    def __init__(self):
        self.image_dimensions = (299, 299)
        self.graph_path = "models/inception_v4/inception_v4.graph"
        self.dataset = ImagenetData(self.image_dimensions)
