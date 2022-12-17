# import
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from tqdm.auto import tqdm, trange

##################  Load Dataset

def image_dataset_loader(img_dir_path, lr_shape, hr_shape, num_patch=10):
    """
    Image Dataset Loader:
    - Load original HR images from path
    - For each original image, Crop number of random image patches with defined HR training shape
    - For each HR image patch, resize to LR training shape
    - Normalize both HR and LR images with given norm_code
    - Append HR and LR as two lists, to numpy array
    - Return np.array(), np.array()

    Args:
        - img_dir_path, path to load the original images, "data/DIV2K_train_HR"
        - lr_shape, the shape of returned low-resolution image patches, "(64, 64, 3)"
        - hr_shape, the shape of returned high-resolution image patches, "(256, 256, 3)"
        - num_patch, how many image patches to collect out of original images, default = 10
    """
    
    hr_images, lr_images = [], []
    
    for img_name in tqdm(os.listdir(path=img_dir_path)):
        
        # collect original images
        ori_img = cv2.imread(f'{img_dir_path}/{img_name}')
        
        # convert color BGR -> RGB
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        
        # random collect image patch with hr_shape
        for _ in range(num_patch):
            
            # get random patch position
            rand_y = np.random.randint(0, ori_img.shape[0]-hr_shape[0])
            rand_x = np.random.randint(0, ori_img.shape[1]-hr_shape[1])
            
            # collect hr image patch
            hr_img = ori_img[
                rand_y : rand_y + hr_shape[0],
                rand_x : rand_x + hr_shape[1],
                :
            ]
            hr_images.append(hr_img)
                       
            # resize to lr shape
            lr_img = cv2.resize(hr_img, dsize=lr_shape[:2])
            lr_images.append(lr_img)
            
    lr_images = np.array(lr_images)
    hr_images = np.array(hr_images)
    
    assert lr_images.shape[1:] == lr_shape, 'LR images shape incorrect'
    assert hr_images.shape[1:] == hr_shape, 'HR images shape incorrect'
    print(f'[INFO] The Dataset now has {len(lr_images)} LR-HR image pairs.')

    return lr_images, hr_images

def img_norm(images, norm_code):

    if norm_code == 0:  # (0, 255) -> (0, 1)
        return images.astype(np.float32) / 255.0
    elif norm_code == 1:  # (0, 255) -> (-1, 1)
        return (images.astype(np.float32) - 127.5) / 127.5
    else:
        raise ValueError('norm_code can only be 0 or 1')
        
def img_un_norm(images, from_norm_code):
    
    if from_norm_code == 0:  # (0, 1) -> (0, 255)
        return np.clip(images*255, 0, 255).astype(np.uint8)
    elif from_norm_code == 1:  # (-1, 1) -> (0, 255)
        return np.clip(((images + 1) * 127.5), 0, 255).astype(np.uint8)
    else:
        raise ValueError('from_norm_code can only be 0 or 1')

##################  End Of Epoch

def plot_g_loss(g_loss):
    
    plt.plot(g_loss, label='g_loss')
    plt.title('Loss plot of Generator Fitting')
    plt.xlabel('step')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(fname='images/gan_g_loss_plot.png')
    
    # Clear the figure and axes
    plt.clf()
    plt.cla()
    plt.close()

def plot_d_loss(d_loss):
    
    plt.plot(d_loss, label='d_loss')
    plt.title('Loss plot of Discriminator')
    plt.xlabel('step')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig(fname='images/gan_d_loss_plot.png')
    
    # Clear the figure and axes
    plt.clf()
    plt.cla()
    plt.close()

def plot_fitting(d_loss_real, d_loss_fake, d_acc_real, d_acc_fake):
    
    # losses plot
    plt.subplot(2, 1, 1)
    plt.plot(d_loss_real, label='d_loss_real')
    plt.plot(d_loss_fake, label='d_loss_fake')
    plt.ylabel('cost')
    plt.legend()
    
    # accuracy plot
    plt.subplot(2, 1, 2)
    plt.plot(d_acc_real, label='d_acc_real')
    plt.plot(d_acc_fake, label='d_acc_fake')
    plt.ylabel('accuracy')
    plt.legend()
    
    # save plot 
    plt.savefig(fname='images/gan_loss_acc_plot.png')
    
    # Clear the figure and axes
    plt.clf()
    plt.cla()
    plt.close()

def plot_images(lr_img, sr_img, hr_img, save=False, save_name='images/results/gan_0.png'):
    
    # print(sr_img.shape, sr_img.dtype, np.min(sr_img), np.max(sr_img))
    
    # De-normalization
    lr_img = img_un_norm(lr_img, from_norm_code=1)
    sr_img = img_un_norm(sr_img, from_norm_code=1)
    hr_img = img_un_norm(hr_img, from_norm_code=1)
    
    # print(sr_img.shape, sr_img.dtype, np.min(sr_img), np.max(sr_img))
    
    # Compute PSNR and SSIM
    psnr = tf.image.psnr(hr_img, sr_img, max_val=255.0).numpy()
    ssim = tf.image.ssim(hr_img, sr_img, max_val=255.0).numpy()
    
    # print(f'PSNR: {psnr:.3f}dB, SSIM: {ssim:.3f}')

    # Plot images
    plt.figure(figsize=(12, 4))
    plt.tight_layout()
    
    plt.subplot(1, 3, 1)
    plt.imshow(lr_img)
    plt.grid('off')
    plt.axis('off')
    plt.title('LR (64, 64, 3)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sr_img)
    plt.grid('off')
    plt.axis('off')
    plt.title(f'[PSNR {psnr:.2f}db / SSIM {ssim:.2f}]')
    
    plt.subplot(1, 3, 3)
    plt.imshow(hr_img)
    plt.grid('off')
    plt.axis('off')
    plt.title(f'HR (256, 256, 3)')
    
    if save:
        plt.savefig(fname=save_name, bbox_inches='tight')
        print("[INFO] LR-SR-HR Image saved.")
        
    plt.show()
    
    # Clear the figure and axes
    plt.clf()
    plt.cla()
    plt.close()

def end_of_epoch(epoch_num, lr_img, sr_img, hr_img, losses, accuracies):
    
    plot_images(lr_img, sr_img, hr_img, save=True, save_name=f'images/results/gan_{epoch_num}.png')
    
    g_loss, d_loss, d_loss_real, d_loss_fake = losses
    d_acc_real, d_acc_fake = accuracies

    plot_g_loss(g_loss)
    plot_d_loss(d_loss)
    plot_fitting(d_loss_real, d_loss_fake, d_acc_real, d_acc_fake)

####################  Model Testing

def plot_predict(lr_images, hr_images, srgan_model, n_imgs=4):

    idx = np.random.randint(0, len(lr_images) - 1, n_imgs)
    
    lr_img = img_norm(lr_images[idx], norm_code=1)
    hr_img = img_norm(hr_images[idx], norm_code=1)
    sr_img = srgan_model.generator.predict_on_batch(lr_img)
    
    # De-normalization
    lr_img = img_un_norm(lr_img, from_norm_code=1)
    sr_img = img_un_norm(sr_img, from_norm_code=1)
    hr_img = img_un_norm(hr_img, from_norm_code=1)
    
    # Bi-cubic interpolation up-sampled image
    bi_img = np.array([
        cv2.resize(src=img, dsize=hr_img.shape[1:3], interpolation=cv2.INTER_CUBIC)
        for img in lr_img
    ])
    
    # Compute PSNR and SSIM
    psnr_sr = tf.image.psnr(hr_img, sr_img, max_val=255.0).numpy()
    ssim_sr = tf.image.ssim(hr_img, sr_img, max_val=255.0).numpy()
    psnr_bi = tf.image.psnr(hr_img, bi_img, max_val=255.0).numpy()
    ssim_bi = tf.image.ssim(hr_img, bi_img, max_val=255.0).numpy()
    
    # Plot images
    plt.figure(figsize=(n_imgs*4, n_imgs*4))
    # plt.tight_layout()
    
    plt.subplot(n_imgs, 4, 1)
    plt.imshow(lr_img[0])
    plt.grid('off')
    plt.axis('off')
    plt.title('LR (64, 64, 3)')
    
    plt.subplot(n_imgs, 4, 2)
    plt.imshow(bi_img[0])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'Bi-cubic [{psnr_bi[0]:.2f}db / {ssim_bi[0]:.2f}]')
    
    plt.subplot(n_imgs, 4, 3)
    plt.imshow(sr_img[0])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'SR [{psnr_sr[0]:.2f}db / {ssim_sr[0]:.2f}]')
    
    plt.subplot(n_imgs, 4, 4)
    plt.imshow(hr_img[0])
    plt.grid('off')
    plt.axis('off')
    plt.title('HR (256, 256, 3)')
    
    plt.subplot(n_imgs, 4, 5)
    plt.imshow(lr_img[1])
    plt.grid('off')
    plt.axis('off')
    plt.title('LR (64, 64, 3)')
    
    plt.subplot(n_imgs, 4, 6)
    plt.imshow(bi_img[1])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'Bi-cubic [{psnr_bi[1]:.2f}db / {ssim_bi[1]:.2f}]')
    
    plt.subplot(n_imgs, 4, 7)
    plt.imshow(sr_img[1])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'SR [{psnr_sr[1]:.2f}db / {ssim_sr[1]:.2f}]')
    
    plt.subplot(n_imgs, 4, 8)
    plt.imshow(hr_img[1])
    plt.grid('off')
    plt.axis('off')
    plt.title('HR (256, 256, 3)')
    
    plt.subplot(n_imgs, 4, 9)
    plt.imshow(lr_img[2])
    plt.grid('off')
    plt.axis('off')
    plt.title('LR (64, 64, 3)')
    
    plt.subplot(n_imgs, 4, 10)
    plt.imshow(bi_img[2])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'Bi-cubic [{psnr_bi[2]:.2f}db / {ssim_bi[2]:.2f}]')
    
    plt.subplot(n_imgs, 4, 11)
    plt.imshow(sr_img[2])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'SR [{psnr_sr[2]:.2f}db / {ssim_sr[2]:.2f}]')
    
    plt.subplot(n_imgs, 4, 12)
    plt.imshow(hr_img[2])
    plt.grid('off')
    plt.axis('off')
    plt.title('HR (256, 256, 3)')
    
    plt.subplot(n_imgs, 4, 13)
    plt.imshow(lr_img[3])
    plt.grid('off')
    plt.axis('off')
    plt.title('LR (64, 64, 3)')
    
    plt.subplot(n_imgs, 4, 14)
    plt.imshow(bi_img[3])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'Bi-cubic [{psnr_bi[3]:.2f}db / {ssim_bi[3]:.2f}]')
    
    plt.subplot(n_imgs, 4, 15)
    plt.imshow(sr_img[3])
    plt.grid('off')
    plt.axis('off')
    plt.title(f'SR [{psnr_sr[3]:.2f}db / {ssim_sr[3]:.2f}]')
    
    plt.subplot(n_imgs, 4, 16)
    plt.imshow(hr_img[3])
    plt.grid('off')
    plt.axis('off')
    plt.title('HR (256, 256, 3)')

    plt.savefig('lr_bicubic_sr_hr.png')
    
