import numpy as np
import gzip
import os
import platform
import pickle
from torchvision import transforms
import torchvision.datasets
from torch.utils.data import DataLoader
import torch
import cv2
import pandas as pd
from PIL import Image

class GetDataSet(object):
    def __init__(self, dataSetName, isIID, datadim):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID, datadim)
        elif self.name == 'cifar10':
            self.CIFAR10DataSetConstruct(isIID,datadim)
        elif self.name == 'GTSRB':
            self.GTSRBDataSetConstruct(isIID,datadim)
        else:
            pass



    def GTSRBDataSetConstruct(self, isIID, datadim):
        Train_data_dir = r'./GTSRB/Training_Image_Transfer/GTSRB/TRAIN/csv_train_data/train_data.txt'
        Test_data_dir = r'./GTSRB/Test_Image_Transfer/GTSRB/TEST/csv_test_data/test_data.csv'
        Train_csv = pd.read_csv(Train_data_dir,header = None, names=['file_name', 'label'])
        Test_csv = pd.read_csv(Test_data_dir,header = None, names=['file_name', 'label'])

        Train_data = []
        Test_data = []
        Train_label = []
        Test_label = []

        for i in Train_csv.index:
            file_name = list(Train_csv['file_name'][i])
            file_name[0] = 'GTSRB'
            file_name = ''.join(file_name)
            file_name = file_name.replace('\\','/')
            image = cv2.imread(file_name)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((32, 32))
            Train_data.append(np.array(resize_image))
            Train_label.append(Train_csv['label'][i])
        for i in Test_csv.index:
            file_name = list(Test_csv['file_name'][i])
            file_name[0] = 'GTSRB'
            file_name = ''.join(file_name)
            file_name = file_name.replace('\\', '/')
            image = cv2.imread(file_name)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((32, 32))
            Test_data.append(np.array(resize_image))
            Test_label.append(Test_csv['label'][i])

        Train_data = np.array(Train_data)/ 255.
        Test_data = np.array(Test_data)/ 255.

        x_train = Train_data.astype(float)
        y_train = np.array(Train_label)
        x_test = Test_data.astype(float)
        y_test = np.array(Test_label)

        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        self.train_data_size = x_train.shape[0]
        self.test_data_size = x_test.shape[0]

        assert x_train.shape[3] == 3
        assert x_test.shape[3] == 3



        if datadim == 3:
            train_images = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], x_train.shape[3])
            test_images = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], x_test.shape[3])
        else:
            train_images = x_train.transpose(0, 3, 1, 2)
            test_images = x_test.transpose(0, 3, 1, 2)

        # train_images = train_images.astype(np.float32)
        # train_images = np.multiply(train_images, 1.0 / 255.0)
        # test_images = test_images.astype(np.float32)
        # test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = y_train[order]
        else:
            # labels = np.argmax(train_labels, axis=1)
            order = np.argsort(y_train)
            self.train_data = train_images[order]
            self.train_label = y_train[order]

        self.test_data = test_images
        self.test_label = y_test


    def CIFAR10DataSetConstruct(self, isIID, datadim):
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # cifar_train = torchvision.datasets.CIFAR10(
        #     root='../data/cifar-10-python/',
        #     train=True,
        #     transform=transforms.Compose([
        #         transforms.Resize((32, 32)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        #     ]),
        #     download=False
        # )

        cifar_train = torchvision.datasets.CIFAR10(
            root='../data/cifar-10-python/',
            train=True,
            transform=None,
            download=False
        )
        train_images = cifar_train.data
        train_labels = np.array(cifar_train.targets)

        cifar_test = torchvision.datasets.CIFAR10(
            root='../data/cifar-10-python/',
            train=False,
            transform=None,
            download=False
        )
        test_images = cifar_test.data
        test_labels = np.array(cifar_test.targets)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 3
        assert test_images.shape[3] == 3

        if datadim == 3:
            train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2], train_images.shape[3])
            test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2], test_images.shape[3])
        else:
            train_images = train_images.transpose(0, 3, 1, 2)
            test_images = test_images.transpose(0, 3, 1, 2)

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # labels = np.argmax(train_labels, axis=1)
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

    def mnistDataSetConstruct(self, isIID, datadim):
        # data_dir = r'.\data\MNIST'
        # data_dir = r'./data/MNIST'
        data_dir = '../data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        if datadim == 2:
            train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
            test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        else:
            train_images = train_images.transpose(0, 3, 1, 2)
            test_images = test_images.transpose(0, 3, 1, 2)

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]



        self.test_data = test_images
        self.test_label = test_labels
        self.test_label = np.argmax(test_labels, axis=1)

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


# if __name__=="__main__":
#     'test data set'
#     mnistDataSet = GetDataSet('mnist', True) # test NON-IID
#     if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
#             type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
#         print('the type of data is numpy ndarray')
#     else:
#         print('the type of data is not numpy ndarray')
#     print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
#     print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
#     print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

