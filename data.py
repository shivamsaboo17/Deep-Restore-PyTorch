import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from sys import platform
from random import choice
from string import ascii_letters
import os
import scipy
import random
import cv2

class NoisyDataset(Dataset):
    
    def __init__(self, root_dir, crop_size=128, train_noise_model=('gaussian', 50), clean_targ=False):
        """
        root_dir: Path of image directory
        crop_size: Crop image to given size
        clean_targ: Use clean targets for training
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targ = clean_targ
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.imgs = os.listdir(root_dir)

    
    def _random_crop_to_size(self, imgs):
        
        w, h = imgs[0].size
        assert w >= self.crop_size and h >= self.crop_size, 'Cannot be croppped. Invalid size'
        

        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 2)
        j = np.random.randint(0, w - self.crop_size + 2)

        for img in imgs:
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))
        
        #cropped_imgs = cv2.resize(np.array(imgs[0]), (self.crop_size, self.crop_size))
        return cropped_imgs
    
    def _add_gaussian_noise(self, image):
        """
        Added only gaussian noise
        """
        w, h = image.size
        c = len(image.getbands())
        
        std = np.random.uniform(0, self.noise_param)
        _n = np.random.normal(0, std, (h, w, c))
        noisy_image = np.array(image) + _n
        
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image':Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_poisson_noise(self, image):
        """
        Added poisson Noise
        """
        noise_mask = np.random.poisson(np.array(image))
        #print(noise_mask.dtype)
        #print(noise_mask)
        return {'image':noise_mask.astype(np.uint8), 'mask': None, 'use_mask': False}

    def _add_m_bernoulli_noise(self, image):
        """
        Multiplicative bernoulli
        """
        sz = np.array(image).shape[0]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=(sz, sz), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return {'image':np.multiply(image, mask).astype(np.uint8), 'mask':mask.astype(np.uint8), 'use_mask': True}


    def _add_text_overlay(self, image):
        """
        Add text overlay to image
        """
        assert self.noise_param < 1, 'Text parameter should be probability of occupancy'

        w, h = image.size
        c = len(image.getbands())

        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'

        text_img = image.copy()
        text_draw = ImageDraw.Draw(text_img)
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        max_occupancy = np.random.uniform(0, self.noise_param)

        def get_occupancy(x):
            y = np.array(x, np.uint8)
            return np.sum(y) / y.size

        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break
        
        return {'image':text_img, 'mask':None, 'use_mask': False}

    def corrupt_image(self, image):
        
        if self.noise == 'gaussian':
            return self._add_gaussian_noise(image)
        elif self.noise == 'poisson':
            return self._add_poisson_noise(image)
        elif self.noise == 'multiplicative_bernoulli':
            return self._add_m_bernoulli_noise(image)
        elif self.noise == 'text':
            return self._add_text_overlay(image)
        else:
            raise ValueError('No such image corruption supported')

    def __getitem__(self, index):
        """
        Read a image, corrupt it and return it
        """
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')

        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]

        source_img_dict = self.corrupt_image(image)
        source_img_dict['image'] = tvF.to_tensor(source_img_dict['image'])

        if source_img_dict['use_mask']:
            source_img_dict['mask'] = tvF.to_tensor(source_img_dict['mask'])

        if self.clean_targ:
            #print('clean target')
            target = tvF.to_tensor(image)
        else:
            #print('corrupt target')
            _target_dict = self.corrupt_image(image)
            target = tvF.to_tensor(_target_dict['image'])

        if source_img_dict['use_mask']:
            return [source_img_dict['image'], source_img_dict['mask'], target]
        else:
            return [source_img_dict['image'], target]

    def __len__(self):
        return len(self.imgs)

