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

        return cropped_imgs
    
    def _add_noise(self, image):
        """
        Added only gaussian noise
        """
        w, h = image.size
        c = len(image.getbands())

        if self.noise == 'gaussian':
            std = np.random.uniform(0, self.noise_param)
            _n = np.random.normal(0, std, (h, w, c))
            noisy_image = np.array(image) + _n
        
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

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

        return text_img

    def corrupt_image(self, image):
        
        if self.noise == 'gaussian':
            return self._add_noise(image)
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

        source_img = tvF.to_tensor(self.corrupt_image(image))

        if self.clean_targ:
            target = tvF.to_tensor(image)
        else:
            target = tvF.to_tensor(self.corrupt_image(image))

        return source_img, target

    def __len__(self):
        return len(self.imgs)

