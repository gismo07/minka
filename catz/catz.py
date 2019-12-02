import cv2
from tqdm import tqdm
from random import shuffle
import subprocess
from pathlib import Path


import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os


# DL stuff
import h5py
import tensorflow as tf
from tensorflow import keras as tfKeras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras import layers as tfL
import numpy as np
import os

# from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import time
import datetime
import json
from sklearn.metrics import mean_squared_error


class CATZ:
    def __init__(self):
        pass

    class _ImageCallback(Callback):
        def __init__(self, valData):
            self.valData = valData

        def on_epoch_end(self, epoch, logs):
            validation_X = self.valData[:10]
            output = self.model.predict(validation_X)

            validation_X = np.multiply(validation_X, 255.0)
            output = np.multiply(output, 255.0)

            validation_X = validation_X.astype(np.uint8)
            output = output.astype(np.uint8)

            wandb.log(
                {
                    "input": [
                        wandb.Image(np.concatenate([validation_X[i], o], axis=1))
                        for i, o in enumerate(validation_X)
                    ],
                    "output": [
                        wandb.Image(np.concatenate([output[i], o], axis=1))
                        for i, o in enumerate(output)
                    ],
                    "comparison":[
                        wandb.Image(np.concatenate([validation_X[i], output[i]], axis=1))
                        for i, o in enumerate(validation_X)
                    ]
                },
                commit=False,
            )

    def _perceptual_distance(self, y_true, y_pred):
        y_true = 255.0 * y_true
        y_pred = 255.0 * y_pred
        rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
        r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
        g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
        b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

        return K.mean(
            K.sqrt(
                (((512 + rmean) * r * r) / 256)
                + 4 * g * g
                + (((767 - rmean) * b * b) / 256)
            )
        )

    def _isTPUAvailable(self):
        tpuAvailable = False
        TPU_ADDRESS = ""
        try:
            device_name = os.environ["COLAB_TPU_ADDR"]
            TPU_ADDRESS = "grpc://" + device_name
            print("Found TPU at: {}".format(TPU_ADDRESS))
            tpuAvailable = True
        except KeyError:
            print("TPU not found")
        return tpuAvailable, TPU_ADDRESS

    def prepare(self):
        self.testImages = []
        testImagePaths = []
        self.trainImages = []
        trainImagePaths = []

        if not os.path.exists("catz"):
            subprocess.check_output(
                "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz",
                shell=True,
            )

        for filename in Path("./catz/test/").glob("**/*.jpg"):
            testImagePaths.append(filename)
        for filename in Path("./catz/train/").glob("**/*.jpg"):
            trainImagePaths.append(filename)

        shuffle(testImagePaths)
        shuffle(trainImagePaths)

        for p in tqdm(trainImagePaths):
            self.trainImages.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))

        for p in tqdm(testImagePaths):
            self.testImages.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))

        print("\nNum of lodaded train images: %d" % len(trainImagePaths))
        print("Num of loaded test images: %d" % len(testImagePaths))

        self.trainImages = np.divide(np.asarray(self.trainImages), 255.0, dtype=np.float32)
        self.testImages = np.divide(np.asarray(self.testImages), 255.0, dtype=np.float32)

    def run(self, config, customCallbacks):

        learningRage = config["learningRage"]
        batchSize = config["batchSize"]
        epochs = config["epochs"]

        tf.reset_default_graph()
        tfKeras.backend.clear_session()

        model = tfKeras.Sequential()
        model.add(
            tfL.Conv2D(
                16,
                (3, 3),
                strides=1,
                activation="relu",
                padding="same",
                input_shape=self.testImages[0].shape,
            )
        )
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(
            tfL.SeparableConv2D(
                32, (2, 2), strides=2, activation="relu", padding="valid"
            )
        )
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(
            tfL.SeparableConv2D(
                32, (2, 2), strides=2, activation="relu", padding="valid"
            )
        )
        model.add(tfL.BatchNormalization())
        model.add(
            tfL.SeparableConv2D(
                8, (2, 2), strides=2, activation="relu", padding="valid"
            )
        )
        model.add(tfL.BatchNormalization())
        model.add(tfL.Flatten())
        model.add(tfL.Reshape((12, 12, 8)))
        model.add(tfL.UpSampling2D(2))
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(tfL.UpSampling2D(2))
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(tfL.UpSampling2D(2))
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same"))
        model.add(tfL.BatchNormalization())
        model.add(tfL.Conv2D(3, (3, 3), strides=1, activation="hard_sigmoid", padding="same"))

        model.compile(
            optimizer=tfKeras.optimizers.Adam(lr=learningRage),
            loss="mse",
            metrics=[self._perceptual_distance],
        )

        # _useTPU, _tpuAdress = self._isTPUAvailable()
        # model = (
        #     tf.contrib.tpu.keras_to_tpu_model(
        #         model,
        #         strategy=tf.contrib.tpu.TPUDistributionStrategy(
        #             tf.contrib.cluster_resolver.TPUClusterResolver(_tpuAdress)
        #         ),
        #     )
        #     if _useTPU
        #     else model
        # )

        reduce_lr = tfKeras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.75,
            patience=5,
            min_lr=0.0000001,
            mode="min",
            verbose=1,
        )

        imageCallback = self._ImageCallback(self.testImages)

        callbacks = [imageCallback, reduce_lr, *customCallbacks]

        X = self.trainImages
        Y = self.trainImages

        print('SHAPE: '+ str(X.shape))

        model.fit(
            x=X,
            y=Y,
            batch_size=128,
            epochs=100,
            verbose=0,
            validation_data=(self.testImages, self.testImages),
            shuffle=True,
            callbacks=callbacks,
        )

        hist = model.history.history
        min_val_loss = np.min(hist["val_loss"])

        # define the optimization result
        result = min_val_loss

        evalMetrics = {"mse": min_val_loss}

        logArrays = {
            "val_loss": np.asarray(hist["val_loss"], np.float32),
            "loss": np.asarray(hist["loss"], np.float32),
        }

        return result, evalMetrics, logArrays

    def cleanup(self):
        self.testImages = None
        self.trainImages = None
