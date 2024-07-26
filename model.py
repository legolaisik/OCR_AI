import os
import math
import string
import zipfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps
from split_image import split_image

import tensorflow as tf
import keras.backend as K
from keras.backend import get_value, ctc_decode
from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import *


def get_text(img_list):
    # Функция для применения всех функций для предобработки
    def preprocess(img):
        for func in [resize_n_rotate, add_adaptiveThreshold]:
            img = func(img)
        return img


    # Функция для изменения размера изображения и поворота
    def resize_n_rotate(img, shape_to=(128, 512)):
        if img.shape[0] > shape_to[0] or img.shape[1] > shape_to[1]:
            shrink_multiplayer = min(math.floor(shape_to[0] / img.shape[0] * 100) / 100,
                                     math.floor(shape_to[1] / img.shape[1] * 100) / 100)
            img = cv2.resize(img, None,
                             fx=shrink_multiplayer,
                             fy=shrink_multiplayer,
                             interpolation=cv2.INTER_AREA)

        img = cv2.copyMakeBorder(img, math.ceil(shape_to[0] / 2) - math.ceil(img.shape[0] / 2),
                                 math.floor(shape_to[0] / 2) - math.floor(img.shape[0] / 2),
                                 math.ceil(shape_to[1] / 2) - math.ceil(img.shape[1] / 2),
                                 math.floor(shape_to[1] / 2) - math.floor(img.shape[1] / 2),
                                 cv2.BORDER_CONSTANT, value=255)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


    # Функция для применения всех функций для предобработки изображения с рукописным текстом
    def preprocess_l1(img):
        for func in [resize_n_rotate_l1, add_adaptiveThreshold]:
            img = func(img)
        return img


    # Функция для изменения размера изображения с рукописным текстом и поворота
    def resize_n_rotate_l1(img, shape_to=(128, 512)):
        shrink_multiplayer = min(math.floor(shape_to[0] / img.shape[0] * 100) / 100,
                                 math.floor(shape_to[1] / img.shape[1] * 100) / 100)
        img = cv2.resize(img, None,
                         fx=shrink_multiplayer,
                         fy=shrink_multiplayer,
                         interpolation=cv2.INTER_AREA)

        img = cv2.copyMakeBorder(img, math.ceil(shape_to[0] / 2) - math.ceil(img.shape[0] / 2),
                                 math.floor(shape_to[0] / 2) - math.floor(img.shape[0] / 2),
                                 math.ceil(shape_to[1] / 2) - math.ceil(img.shape[1] / 2),
                                 math.floor(shape_to[1] / 2) - math.floor(img.shape[1] / 2),
                                 cv2.BORDER_CONSTANT, value=255)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


    # Функция для изменения цвета изображения на чернобелый и уменьшения шумов
    def add_adaptiveThreshold(img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10).astype('bool')


    alphabet = ' !%%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoprstuvwxyz«\xad°»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё–—…€№'


    # Функция получения символа по индексу в алфавите
    def num_to_label(num, alphabet):
        text = ""
        for ch in num:
            if ch == len(alphabet):
                break
            else:
                text += alphabet[ch]
        return text


    # Функция для раскодирования пердсказания
    def decode_text(nums):
        values = get_value(
            ctc_decode(nums, input_length=np.ones(nums.shape[0]) * nums.shape[1],
                       greedy=True)[0][0])

        texts = []
        for i in range(nums.shape[0]):
            value = values[i]
            texts.append(num_to_label(value[value >= 0], alphabet))
        return texts


    alphabet_l1 = ' !"%\'(),-./0123456789:;?R[]bcehioprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё'


    # Функция для раскодирования пердсказания рукописного текст
    def decode_text_l1(nums):
        values = get_value(
            ctc_decode(nums, input_length=np.ones(nums.shape[0]) * nums.shape[1],
                       greedy=True)[0][0])

        texts = []
        for i in range(nums.shape[0]):
            value = values[i]
            texts.append(num_to_label(value[value >= 0], alphabet_l1))
        return texts


    class CERMetric(tf.keras.metrics.Metric):
        def __init__(self, name='CER_metric', **kwargs):
            super(CERMetric, self).__init__(name=name, **kwargs)
            self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
            self.counter = self.add_weight(name="cer_count", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            input_shape = K.shape(y_pred)
            input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

            decode, log = K.ctc_decode(y_pred, input_length, greedy=True)

            decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
            y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
            y_true_sparse = tf.sparse.retain(y_true_sparse,
                                             tf.not_equal(y_true_sparse.values, tf.math.reduce_max(y_true_sparse.values)))

            decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
            distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

            self.cer_accumulator.assign_add(tf.reduce_sum(distance))
            self.counter.assign_add(K.cast(len(y_true), 'float32'))

        def result(self):
            return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

        def reset_state(self):
            self.cer_accumulator.assign(0.0)
            self.counter.assign(0.0)


    def CTCLoss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss


    model = load_model("./models/Class1.h5")

    labels = []
    imgs = []
    k = 5
    temp_img_list = []
    for i, n in enumerate(img_list):
        split_image(f"./{n}", k, 1, False, False, output_dir="./uploads/")
        file_name, file_format = n.split(".")[0], n.split(".")[1]
        for i in range(k):
            temp_img_list.append(f"{file_name}_{i}.{file_format}")

    img_list = temp_img_list
    for i, n in enumerate(img_list):
        img = cv2.imread(f"./{n}", 1)
        img1 = cv2.resize(img, (256, 64))
        img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img2 = cv2.resize(img2, (256, 64))
        img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img3 = cv2.resize(img3, (256, 64))
        img4 = cv2.rotate(img, cv2.ROTATE_180)
        img4 = cv2.resize(img4, (256, 64))
        imgs.append(img1)
        imgs.append(img2)
        imgs.append(img3)
        imgs.append(img4)

    predict_dict = {}
    class_pred = model.predict(np.array(imgs))
    for i, img in enumerate(imgs):
        if i % 4 == 3:
            p = [class_pred[i - 3], class_pred[i - 2],
                 class_pred[i - 1], class_pred[i]]
            if sum(p) > 2:
                predict_dict.update({img_list[int(i / 4)]: 1})
                labels.append(1)
            else:
                predict_dict.update({img_list[int(i / 4)]: 0})
                labels.append(0)

    printed_X_test = []

    handwritten_X_test = []

    for n in tqdm(img_list):
        img = cv2.imread(f"./{n}", 0)
        if predict_dict[n] == 0:
            printed_X_test.append(preprocess(img))
        else:
            handwritten_X_test.append(preprocess_l1(img))

    result = ""

    if len(printed_X_test) != 0:
        model2 = load_model('./models/Model_L0.h5', custom_objects={'CTCLoss': CTCLoss, 'CERMetric': CERMetric})

        predicts_printed = model2.predict(np.array(printed_X_test))
        predicts_printed = decode_text(predicts_printed)
        result += "\n".join(predicts_printed)

    if len(handwritten_X_test) != 0:
        model = load_model('./models/Model_L1.h5', custom_objects={'CTCLoss': CTCLoss, 'CERMetric': CERMetric})

        predicts_handwritten = model.predict(np.array(handwritten_X_test))
        predicts_handwritten = decode_text_l1(predicts_handwritten)
        result += "\n" + "\n".join(predicts_handwritten)

    return result