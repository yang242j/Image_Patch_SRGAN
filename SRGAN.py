# import
import os
import time
import tensorflow as tf
from keras import Model
from keras.models import load_model
from keras.layers import Input, Conv2D, BatchNormalization, PReLU, UpSampling2D, Add
from keras.layers import Flatten, LeakyReLU, Dense, Lambda, Dropout, GlobalAvgPool2D
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19, preprocess_input
from tqdm.auto import tqdm, trange
import utils  # utils.py
import numpy as np

class SRGAN:
    
    def __init__(self, lr_shape, hr_shape, show_summary=False):
        
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        
        # Build feature_extractor using vgg19
        self.feature_extractor = self.build_feature_extractor(num_vgg_layers=9)
        if show_summary: self.feature_extractor.summary()

        # Build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer=Adam(learning_rate=2e-6, beta_1=0.5),
            loss='mse',
            metrics=['accuracy']
        )
        if show_summary: self.discriminator.summary()

        # Build generator
        self.generator = self.build_generator(num_res_blocks=16)
        if show_summary: self.generator.summary()
        
        # Build GAN model
        self.build_gan()
        self.gan_model.compile(
            optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1]
        )
        if show_summary: self.gan_model.summary()
        for layer in self.gan_model.layers:
            print(f'{layer.name} trainable: {layer.trainable}')

    def build_generator(self, num_res_blocks):
                
        def res_block(x):
            inputs = x
            
            x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = PReLU(shared_axes=[1, 2])(x)
            
            x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            
            return Add()([inputs, x])
        
        def upscale_block(x):
            
            def pixel_shuffle(scale):
                return Lambda(lambda x: tf.nn.depth_to_space(x, scale))
            
            x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
            # x = pixel_shuffle(scale=2)(x)
            x = UpSampling2D(2)(x)
            x = PReLU(shared_axes=[1, 2])(x)
            
            return x
        
        # Input Layer
        lr_img = Input(shape=self.lr_shape)

        # First convolution block without batch normalization.
        x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(lr_img)
        x = PReLU(shared_axes=[1, 2])(x)
        gen_temp = x
        
        # Residual blocks
        for _ in range(num_res_blocks):
            x = res_block(x)
            
        # Second convolution block with batch normalization
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, gen_temp])
        
        # Upscale blocks
        x = upscale_block(x)
        x = upscale_block(x)
        
        # Output layer
        sr_img = Conv2D(filters=3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)
        
        # Define generator model
        return Model(inputs=lr_img, outputs=sr_img, name='Generator')
    
    def build_discriminator(self):
        
        def d_block(x, filters, strides, batch_norm=False):
            
            # Convolution block: Conv2D -> BN -> LeakyReLU
            x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x) if batch_norm else x
            x = LeakyReLU(alpha=0.2)(x)
            
            return x
        
        # Input layer
        hr_img = Input(shape=self.hr_shape)
        
        # Convolution blocks: filters = 64
        x = d_block(hr_img, filters=64, strides=1)
        x = d_block(x, filters=64, strides=2, batch_norm=True)
        # x = Dropout(0.2)(x)
        
        # Convolution blocks: filters = 128
        x = d_block(x, filters=128, strides=1, batch_norm=True)
        x = d_block(x, filters=128, strides=2, batch_norm=True)
        # x = Dropout(0.2)(x)
        
        # Convolution blocks: filters = 256
        x = d_block(x, filters=256, strides=1, batch_norm=True)
        x = d_block(x, filters=256, strides=2, batch_norm=True)
        # x = Dropout(0.2)(x)
        
        # Convolution blocks: filters = 512
        x = d_block(x, filters=512, strides=1, batch_norm=True)
        x = d_block(x, filters=512, strides=2, batch_norm=True)
        # x = Dropout(0.2)(x)
        
        # Fully connected layers
        # x = Flatten()(x)
        # x = GlobalAvgPool2D()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        # x = Dropout(0.5)(x)
        
        # Output layer
        confidence = Dense(1, activation='sigmoid')(x)
        
        # Define Discriminator model
        return Model(inputs=hr_img, outputs=confidence, name='Discriminator')
        
    def build_feature_extractor(self, num_vgg_layers):
        
        # Original VGG19 Model
        vgg_model = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=self.hr_shape
        )

        # Pruned VGG19 Model
        return Model(
            inputs=vgg_model.inputs,
            outputs=vgg_model.layers[num_vgg_layers].output,
            name='VGG19_' + str(num_vgg_layers)
        )
    
    # def preprocess_vgg_input(self, data):
        
    #     if isinstance(data, np.ndarray):
    #         return preprocess_input(utils.img_un_norm(data, from_norm_code=1))
    #     else:            
    #         return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(data)    
    
    def build_gan(self):
        
        # Freeze non-trainable models
        self.discriminator.trainable = False
        self.feature_extractor.trainable = False
        
        # GAN inputs
        lr_img = Input(shape=self.lr_shape, name='LR_INPUT')
        
        # GAN middle layer
        sr_img = self.generator(lr_img)
        
        # GAN outputs
        d_output = self.discriminator(sr_img)
        fe_output = self.feature_extractor(sr_img)
        
        self.gan_model = Model(
            inputs=[lr_img],
            outputs=[d_output, fe_output],
            name='SRGAN'
        )
    
    def load_gan_weights(self, filepath):
        self.gan_model.load_weights(filepath, by_name=True)
    
    def save_gan_weights(self, epoch=0):
        self.gan_model.save_weights(
            filepath=f'model/gan_weights/e_{epoch}.h5' if epoch != 0 else 'model/gan_weights.h5'
        )
        print(f'[INFO] GAN model weights epoch_{epoch} saved.')

    def train_gan(self, train_lr, train_hr, epochs, batch_size=1, steps_per_epoch=100, save_epoch_weights=True, save_final_weights=True):
        
        def batch_data_gen():
            while True:
                idx = np.random.randint(0, len(train_lr), batch_size)
                lr_imgs = utils.img_norm(train_lr[idx], norm_code=1)
                hr_imgs = utils.img_norm(train_hr[idx], norm_code=1)
                yield(lr_imgs, hr_imgs)
        
        data_gen = batch_data_gen()

        # init fitting history lists
        g_losses, d_losses, d_real_losses, d_fake_losses = [], [], [], []
        d_real_accuracies, d_fake_accuracies = [], []
        
        # init label shape
        label_shape = list(self.discriminator.output_shape)  # (None, 16, 16, 1)
        label_shape[0] = batch_size  # (batch_size, 16, 16, 1)
        label_shape = tuple(label_shape)

        # for each epoch
        for epoch in range(1, epochs+1):
            
            # start of epoch
            print(f'\nEpoch {epoch}/{epochs}')
            start_time = time.time()
            
            gan_bce_losses, gan_mse_losses = [], []
        
            # for each batch_step
            for _ in trange(steps_per_epoch):
                
                # ----------------------------
                # Train Discriminator
                # ----------------------------
                
                # Fetch D training data
                lr_imgs, hr_imgs = next(data_gen)
                
                # Generate SR images
                sr_imgs = self.generator.predict_on_batch(lr_imgs)

                # define real_label & fake_label
                # real_label = np.ones(label_shape) - np.random.random_sample(label_shape)*0.1
                # fake_label = np.random.random_sample(label_shape)*0.1
                real_label = np.ones(label_shape)
                fake_label = np.zeros(label_shape)
                                
                # Train the Discriminator
                self.discriminator.trainable = True
                d_loss_real, d_acc_real = self.discriminator.train_on_batch(hr_imgs, real_label)
                d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(sr_imgs, fake_label)
                self.discriminator.trainable = False

                # Average D loss
                d_loss_avg = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ----------------------------
                # Train Generator through GAN
                # ----------------------------
                
                # Fetch G training data
                lr_imgs, hr_imgs = next(data_gen)
                
                # Collect ground truth HR features
                hr_features = self.feature_extractor.predict_on_batch(hr_imgs)
                
                # Train the GAN Generator
                gan_g_loss, gan_bce_loss, gan_mse_loss = self.gan_model.train_on_batch(
                    x=[lr_imgs], 
                    y=[np.ones(label_shape), hr_features]
                )
                
                # store fitting step losses
                g_losses.append(gan_g_loss)
                gan_bce_losses.append(gan_bce_loss)
                gan_mse_losses.append(gan_mse_loss)
                d_losses.append(d_loss_avg)
                d_real_losses.append(d_loss_real)
                d_fake_losses.append(d_loss_fake)
                d_real_accuracies.append(d_acc_real)
                d_fake_accuracies.append(d_acc_fake)
              
            print(f'Time Elapsed:{(time.time() - start_time):.0f}s')
            print(f'g_loss: {np.average(g_losses[-100:]):.3f}; g_bce_loss: {np.average(gan_bce_losses):.3f}; g_mse_loss: {np.average(gan_mse_losses):.3f}')
            print(f'd_loss: {np.average(d_losses[-100:]):.3f}; real_dis_acc: {np.average(d_real_accuracies[-100:]):.3f}; fake_dis_acc: {np.average(d_fake_accuracies[-100:]):.3f}')
            
            # end of epoch
            test_lr_img = utils.img_norm(train_lr[0], norm_code=1)
            test_hr_img = utils.img_norm(train_hr[0], norm_code=1)
            test_sr_img = self.generator.predict_on_batch(np.expand_dims(test_lr_img, axis=0))[0]
            utils.end_of_epoch(
                epoch, test_lr_img, test_sr_img, test_hr_img,
                losses=[g_losses, d_losses, d_real_losses, d_fake_losses],
                accuracies=[d_real_accuracies, d_fake_accuracies]
            )
            
            # Save model every 20 epoch
            if epoch % 20 == 0 and save_epoch_weights:
                self.save_gan_weights(epoch)
                
            # Save model of last epoch
            if epoch == epochs and save_final_weights:
                self.save_gan_weights(epoch=0)
