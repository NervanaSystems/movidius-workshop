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
import cv2
import numpy


class Dataset():
    @property
    def categories(self):
        raise NotImplementedError()

    def load_data(self, image_path: str):
        raise NotImplementedError()


class ImagenetData(Dataset):
    def __init__(self, image_dimensions):
        self.image_dimensions = image_dimensions
        self.categories_path = "datasets/imagenet/categories.txt"
        self.channels_num = 3

    @property
    def categories(self):
        """
        Return dataset categories.

        :return: dataset categories
        """
        categories = []
        with open(self.categories_path, 'r') as categories_file:
            for line in categories_file:
                cat = line.split('\n')[0]
                if cat != 'classes':
                    categories.append(cat)

        return categories

    def load_data(self, image_path: str):
        """
        Preprocessing image data

        :return: preprocessed image
        """
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

        for i in range(self.channels_num):
            img[:, :, i] = (img[:, :, i] - mean) * std

        return img


class AgeData(Dataset):
    def __init__(self, image_dimensions):
        self.image_dimensions = image_dimensions

    @property
    def categories(self):
        """
        Return dataset categories.

        :return: dataset categories
        """
        categories = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
        return categories

    def load_data(self, image_path: str):
        """
        Preprocessing image data

        :return: preprocessed image
        """
        ilsvrc_mean = numpy.load("models/age_gender/age_gender_mean.npy").mean(1).mean(1)
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_dimensions)
        img = img.astype(numpy.float32)
        img[:, :, 0] = (img[:, :, 0] - ilsvrc_mean[0])
        img[:, :, 1] = (img[:, :, 1] - ilsvrc_mean[1])
        img[:, :, 2] = (img[:, :, 2] - ilsvrc_mean[2])
        return img


class GenderData(Dataset):
    def __init__(self, image_dimensions):
        self.image_dimensions = image_dimensions

    @property
    def categories(self):
        """
        Return dataset categories.

        :return: dataset categories
        """
        categories = ['Male', 'Female']
        return categories

    def load_data(self, image_path: str):
        """
        Preprocessing image data

        :return: preprocessed image
        """
        ilsvrc_mean = numpy.load("models/age_gender/age_gender_mean.npy").mean(1).mean(1)
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_dimensions)
        img = img.astype(numpy.float32)
        img[:, :, 0] = (img[:, :, 0] - ilsvrc_mean[0])
        img[:, :, 1] = (img[:, :, 1] - ilsvrc_mean[1])
        img[:, :, 2] = (img[:, :, 2] - ilsvrc_mean[2])
        return img


class MnistData(Dataset):
    def __init__(self, image_dimensions):
        self.image_dimensions = image_dimensions

    @property
    def categories(self):
        """
        Return dataset categories.

        :return: dataset categories
        """
        categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return categories

    def load_data(self, image_path: str):
        """
        Preprocessing image data

        :return: preprocessed image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.image_dimensions)
        img = img.astype(numpy.float32)
        img[:] = ((img[:]) * (1.0/255.0))
        return img
