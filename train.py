# import
import os
import time
import numpy as np
from SRGAN import SRGAN
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import utils
from tqdm.auto import tqdm, trange
from keras.applications.vgg19 import VGG19, preprocess_input

if __name__ == "__main__":
    
    '''
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print("Tensorflow Version: ", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Load images from directory with normalization
    train_lr, train_hr = utils.image_dataset_loader(
        img_dir_path='data/DIV2K_train_HR/', 
        lr_shape=(64, 64, 3), 
        hr_shape=(256, 256, 3), 
        num_patch=10
    )
    
    # Build SRGAN model
    srgan = SRGAN(
        lr_shape=(64, 64, 3), 
        hr_shape=(256, 256, 3)
    )

    # Train SRGAN model
    srgan.train_gan(
        train_lr, train_hr, epochs=300, batch_size=1, steps_per_epoch=100
    )

    # # Plot the final results
    utils.plot_predict(train_lr, train_hr, srgan)