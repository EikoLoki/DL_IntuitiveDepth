# Author: Keshuai Xu, Gopika
# Implement some data augmentation methods here

import albumentations as a
import torchvision.transforms.functional as tf


def augment_scared(strength=0.5):
    """ Generate augmentations for left->right image pairs
    Usage:
        transformed = transformer(left=left_images, right=right_images)
        transformed_left = transformed['left']
        transformed_right = transformed['right']

    :param strength: a number between 0 and 1. scales down the probability of transformation applied.
    :return: transformer
    """
    targets = {'left': 'image', 'right': 'image'}
    transformations = [a.MotionBlur(p=0.5 * strength),
                       a.RandomBrightnessContrast(p=1 * strength),
                       a.RandomGamma(p=0.5 * strength),
                       a.VerticalFlip(p=0.5 * strength)]
    return a.Compose(transformations, additional_targets=targets)

def augment_color(strength=0.5):
    """ Generate color augmentation for left->right image pairs
    Usage:
        transformed = transformer(left=left_images, right=right_images)
        transformed_left = transformed['left']
        transformed_right = transformed['right']

    :param strength: a number between 0 and 1. scales down the probability of transformation applied.
    :return: transformer
    """
    targets = {'left': 'image', 'right': 'image'}
    transformation = a.OneOf([a.Compose([a.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5 * strength),
                                         a.RandomGamma(gamma_limit=(80, 120), p=0.5 * strength),
                                         a.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=0, val_shift_limit=0, p=0.5 * strength)])])
    return a.Compose(transformation, additional_targets=targets)

def augment_quality(strength=0.5):
    """ Generate quality augmentation for left->right image pairs
    Usage:
        transformed = transformer(left=left_images, right=right_images)
        transformed_left = transformed['left']
        transformed_right = transformed['right']

    :param strength: a number between 0 and 1. scales down the probability of transformation applied.
    :return: transformer
    """
    targets = {'left': 'image', 'right': 'image'}
    transformation = a.OneOf([a.Compose([a.Blur(p=0.5 * strength),
                                         a.MedianBlur(p=0.5 * strength),
                                         a.JpegCompression(quality_lower=20, quality_upper=100, p=0.5 * strength)])])
    return a.Compose(transformation, additional_targets=targets)

def augment_noise(strength=0.5):
    """ Generate noise augmentation for left->right image pairs
    Usage:
        transformed = transformer(left=left_images, right=right_images)
        transformed_left = transformed['left']
        transformed_right = transformed['right']

    :param strength: a number between 0 and 1. scales down the probability of transformation applied.
    :return: transformer
    """
    targets = {'left': 'image', 'right': 'image'}
    transformation = a.OneOf([a.Compose([a.GaussNoise(var_limit=(10, 30), p=0.5 * strength),
                                         a.IAAAdditiveGaussianNoise(loc=0, scale=(0.005 * 255, 0.02 * 255), p=0.5 * strength)])])
    return a.Compose(transformation, additional_targets=targets)

def swap_lr(left, right):
    """ Rotate the endoscope 180 degrees. Left becomes right and right becomes left

    :return: new left and right image sets.
    """
    new_right = tf.rotate(left, 180)
    new_left = tf.rotate(right, 180)
    return new_left, new_right
