# -*- coding: utf-8 -*-
"""MinkaNew.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j_jbQir3fM8E1OrkOPe2Va3-X-TdXCji
"""

# with this command, we can get our current ip (from the vm provided by google)
!curl ipecho.net/plain

# first we have to install or update a bunch of packages
!pip install --upgrade minka-johann-haselberger

# If we want to use wandb, we must create a project and replace <YOUR-WANDB-PROJECT-ID> with the 
# corresponding project id, if no wandb service should be used, just comment the following line out
!wandb login <YOUR-WANDB-PROJECT-ID>

# as we want to load some data from our google drive, we first have to virtually mount our storage
from google.colab import drive
drive.mount('/content/drive')

# import some needed packages
from minka import minka
import numpy as np
from loguru import logger
import os
import time
import datetime
import json

# import the DL stuff
import h5py
import tensorflow as tf
from tensorflow import keras as tfKeras
from tensorflow.keras import layers as tfL

#from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error

# minka makes it easy to handle all the logging and optimization tasks. To do so, we have to define our optimization problem.
# This is bundled into a class with the functions prepare, run, and cleanup.
# The experiment can consist of multiple runs - eg. multiple parameter configurations. 
# However the functions prepare and cleanup are only run once! This avoids long loading times, eg. to load the dataset.
# If the DNN model itself is not changed over the experiments, maybee move the model creation into the prepare function. 
class YOUR_OBJECTIVE:
  def __init__(self):
      pass
    
  def prepare(self):
    # as this is 'only' the prepare function, we dont have access to the config (defined parameters) here
    logger.info("Preparing data")
    
    #TODO: load your data
    # for example self.X = ...

    # this is also the place for rough data preparation (normalization, train & test split)

    logger.info('Data prepared successfully')

  def run(self, config, customCallbacks):
    # the main run function
    # this function is triggerd per run with different parameter configuration bundled into the config object.
    # to access a needed parameter simply get it by its name from the config object.
    # CAUTION: only parameters defined within the configuration.json are accessable on runtime.  
    logger.info("Starting the test run")
    
    # get the parameters
    someParameter = config['someParameter']
    learningRate = config['learningRate']
    
    # if we create the model per run, it's good practise to reset the graphs!    
    tf.reset_default_graph()
    tfKeras.backend.clear_session()

    # now we can define our model
    model = tfKeras.Sequential()
    # TODO: add some layers
    model.add(tfL.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=tfKeras.optimizers.Adam(lr=learningRate))

    # define some callbacks for keras
    # in this case we only define the learning rate scheduler to reduce the lr if we come to a plateo
    # TODO: maybee adjust the parameters, but they work pretty well 
    reduce_lr = tfKeras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, min_lr=0.0000001, mode='min', verbose=1)
    
    # also include some minka intern callbacks (dont worry about that yet)
    callbacks = [reduce_lr,*customCallbacks]
    
    # lets start the training
    model.fit(
        x=X,
        y=Y,
        batch_size=batchSize,
        epochs=epochs,
        verbose=0,
        validation_data=(X_val, Y_val),
        shuffle=True,
        callbacks = callbacks
    )
    
    # get the history (keras training logs)
    hist = model.history.history
    
    logger.info("Calculating the evaluation metrics")

    # just an example if we want to calc. some performance metrics AFTER training:
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    mse = mean_squared_error(Y_val, val_predictions)
    mse_test = mean_squared_error(Y_test, test_predictions)
    min_val_loss = np.min(hist['val_loss'])

    # define the optimization result
    # this is important: as we want to automatically adjust the hyperparameters (the ones definded in the config.json) we have
    # to return a SINGLE float as performance indicator. Minka searches for the LOWEST one.
    result = min_val_loss

    # besides the result, we can log any value into our database
    # just fill in the evalMetrics
    evalMetrics = {
        'mse': mse,
        'mse_test': mse_test,
        'min_val_loss': min_val_loss
    }

    # besides scalar metrics, we can also log any value or even arrays into our database
    # just ad some key and the values to the logArrays object.
    # right now, only floats and ints are supported ...
    logArrays = {
        'val_loss': np.asarray(hist['val_loss'], np.float32),
        'loss': np.asarray(hist['loss'], np.float32),
        'labels': np.squeeze(Y_test),
        'predictions': np.squeeze(test_predictions)
    }

    return result, evalMetrics, logArrays
    
  def cleanup(self):
    logger.info("Doing the cleanup")
    pass



# this is our main context
# super easy: create a object of your optimization task, pass it to minka and start the optimization with a single line!
# Nice to know: fill in the expName like you named your wandb project - the number within the opt call defines the number of 
# optimization iterations (how many times the parameters within the config.json are changed)
te = YOUR_OBJECTIVE()
minka(te,'./config.json',attachMongoDB=True,attachTelegram=False,attachWandB=True,expName='someCoolName').opt(30)

# ===> As MINKA is still work in process:
# - make sure to always look for updates
# - the chance to encounter some bugs is quite high, however Minka was tested in some szenarions
# - if you find some issues or need any help, please create an issue on the project git site!
# - feel free to start a pull request if you created any improvements