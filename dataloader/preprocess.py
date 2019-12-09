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
    targets = {'right': 'image'}
    transformations = [a.MotionBlur(p=0.5 * strength),
                       a.RandomBrightnessContrast(p=1 * strength),
                       a.RandomGamma(p=0.5 * strength),
                       a.VerticalFlip(p=0.5 * strength)]
    return a.Compose(transformations, additional_targets=targets)


def swap_lr(left, right):
    """ Rotate the endoscope 180 degrees. Left becomes right and right becomes left

    :return: new left and right image sets.
    """
    new_right = tf.rotate(left, 180)
    new_left = tf.rotate(right, 180)
    return new_left, new_right
