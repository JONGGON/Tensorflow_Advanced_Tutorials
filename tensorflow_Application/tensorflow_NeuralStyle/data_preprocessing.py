import cv2
import numpy as np


def data_preprocessing(content_image=None, style_image=None, image_size=()):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    # 1. content_image

    ci = cv2.imread(content_image, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(ci)
    ci = cv2.merge([r, g, b])
    ci = cv2.resize(ci, dsize=(image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    ci = np.divide(ci, 255)
    ci = (ci - mean) / std
    ci = ci.reshape((-1,) + ci.shape)

    # 2. style_image
    si = cv2.imread(style_image, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(si)
    si = cv2.merge([r, g, b])
    si = cv2.resize(si, dsize=(image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    si = np.divide(si, 255)
    si = (si - mean) / std
    si = si.reshape((-1,) + si.shape)

    # 3.noise image
    noise = np.random.uniform(low=0, high=1, size=image_size + (3,))
    noise = noise.reshape((-1,) + noise.shape)

    return ci, si, noise
