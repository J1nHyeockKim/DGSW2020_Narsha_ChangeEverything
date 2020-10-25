import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16, VGG16
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

import sys

from PIL import Image
from tensorflow_addons.layers import InstanceNormalization

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0], ], "REFLECT", )


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.conv2d = layers.Conv2D(channels, kernel_size, strides=strides)

    def call(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class UpsampleConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.conv2d = layers.Conv2D(channels, kernel_size, strides=strides)
        self.up2d = layers.UpSampling2D(size=upsample)

    def call(self, x):
        x = self.up2d(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in1 = InstanceNormalization()
        self.conv2 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in2 = InstanceNormalization()

    def call(self, inputs):
        residual = inputs
        x = self.in1(self.conv1(inputs))
        x = tf.nn.relu(x)
        x = self.in2(self.conv2(x))
        x = x + residual
        return x


class TransformerNet(tf.keras.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()

        self.conv1 = ConvLayer(16, kernel_size=9, strides=1)
        self.in1 = InstanceNormalization()

        self.conv2 = ConvLayer(32, kernel_size=3, strides=2)
        self.in2 = InstanceNormalization()

        self.conv3 = ConvLayer(64, kernel_size=3, strides=2)
        self.in3 = InstanceNormalization()

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)

        self.deconv1 = UpsampleConvLayer(32, kernel_size=3, strides=1, upsample=2)
        self.in4 = InstanceNormalization()
        self.deconv2 = UpsampleConvLayer(16, kernel_size=3, strides=1, upsample=2)
        self.in5 = InstanceNormalization()
        self.deconv3 = ConvLayer(3, kernel_size=9, strides=1)

        self.relu = layers.ReLU()

    def call(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
        x = layers.Activation('tanh')(x)
        x = (x + 1) * 127.5
        return x


transformer = TransformerNet()

transformer.build((None, None, None, 3))

basepath = 'filter/'
# 전처리 끝


# 이게 필터 불러오는 코드
transformer.load_weights(basepath + 'transformer_1.h5')

import sys
import os
import threading
import cv2
from PyQt5 import uic, QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *

form_class = uic.loadUiType("mainwindow.ui")[0]
filterDialog = uic.loadUiType("filterdialog.ui")[0]
running = True
count = 1


class MyWindow(QMainWindow, form_class):
    def __init__(self):
        global count
        super().__init__()
        self.setupUi(self)
        self.start()

        self.dialog = DialogWindow()
        self.saveButton.clicked.connect(self.save_changed)
        self.changeButton.clicked.connect(self.dialog_open)

        save_path = "save/"
        file_list = os.listdir(save_path)

        if not file_list:
            count = 1
        else:
            fileString = os.path.splitext(file_list[-1])[0]
            count = int(fileString[-1]) + 1

    def dialog_open(self):
        self.dialog.show()

    def save_changed(self, arg1):
        global count
        # print(arg1)
        img = cv2.cvtColor(self.img_result, cv2.COLOR_BGR2RGB)
        filename = 'save/changed{}.jpg'.format(count)
        print(filename)
        cv2.imwrite(filename, img)
        count += 1

    def run(self):
        global running

        cap = cv2.VideoCapture(0)
        cap.set(3, 960)
        cap.set(4, 540)

        while running:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape

                np_image = np.array(img).copy().astype('float32') / 127.5 - 1

                self.img_result = transformer.predict(np_image[tf.newaxis, :, :, :])[0].astype('uint8')

                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                qImg2 = QtGui.QImage(self.img_result.data, w, h, w * c, QtGui.QImage.Format_RGB888)

                pixmap = QtGui.QPixmap.fromImage(qImg)
                pixmap2 = QtGui.QPixmap.fromImage(qImg2)

                self.originalVideo.setPixmap(pixmap)
                self.changedVideo.setPixmap(pixmap2)

            else:
                QtWidgets.QMessageBox.about(self.window(), "Error", "Cannot read frame.")
                print("cannot read frame.")
                break

        cap.release()
        print("Thread end")

    def start(self):
        global running
        running = True
        th = threading.Thread(target=self.run)
        th.daemon = True
        th.start()
        print("started")

    def stop(self):
        global running
        running = False
        print("stop")


class DialogWindow(QDialog, filterDialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()

    sys.exit(app.exec_())
