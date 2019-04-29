from chainercv import transforms
import PIL
import numpy as np


class Transform(object):

    def __init__(
            self, mean,
            crop_size, scale_range=[0.5, 2.0]):
        self.mean = mean
        self.scale_range = scale_range
        self.crop_size = crop_size

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale
        scaled_H = int(scale * H)
        scaled_W = int(scale * W)
        img = transforms.resize(img, (scaled_H, scaled_W), PIL.Image.BICUBIC)
        label = transforms.resize(
            label[None], (scaled_H, scaled_W), PIL.Image.NEAREST)[0]

        # Crop
        if (scaled_H < self.crop_size[0]) or (scaled_W < self.crop_size[1]):
            shorter_side = min(img.shape[1:])
            img, param = transforms.random_crop(
                img, (shorter_side, shorter_side), True)
        else:
            img, param = transforms.random_crop(img, self.crop_size, True)
        label = label[param['y_slice'], param['x_slice']]

        # Rotate
        angle = np.random.uniform(-10, 10)
        img = transforms.rotate(img, angle, expand=False)
        label = transforms.rotate(
            label[None], angle, expand=False,
            interpolation=PIL.Image.NEAREST,
            fill=-1)[0]

        # Resize
        if ((img.shape[1] < self.crop_size[0])
                or (img.shape[2] < self.crop_size[1])):
            img = transforms.resize(img, self.crop_size, PIL.Image.BICUBIC)
        if ((label.shape[0] < self.crop_size[0])
                or (label.shape[1] < self.crop_size[1])):
            label = transforms.resize(
                label[None].astype(np.float32),
                self.crop_size, PIL.Image.NEAREST)
            label = label.astype(np.int32)[0]
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = transforms.flip(img, x_flip=True)
            label = transforms.flip(label[None], x_flip=True)[0]

        # Mean subtraction
        img = img - self.mean
        return img, label
