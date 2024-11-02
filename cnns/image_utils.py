import skimage.transform

from .custom_types import NPImages


def resize_images(images: NPImages, size: tuple[int, int]) -> NPImages:
    n_images = images.shape[0]
    return skimage.transform.resize(images, (n_images, size[0], size[1], 3), order=1)
