from PIL.Image import RASTERIZE
import numpy as np
from numpy.lib.function_base import interp


def my_imfilter(image, filter, convmode='asstride'):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
     with matrices is fine and encouraged. Using opencv or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take an absurdly long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    # padding
    # p = (f-1)/2 keep the same resolution after convlution
    img_v, img_h = image.shape[:2]
    krnl_v, krnl_h = filter.shape[:2]
    pad_v = (krnl_v-1)//2
    pad_h = (krnl_h-1)//2
    filtered_image = np.zeros(image.shape, dtype='float32')
    channels = image.shape[2]
    _filter = filter.astype('float32')
    img_pad = np.pad(image, [(pad_v, pad_v), (pad_h, pad_h), (0, 0)], 'constant')
    if convmode == 'multiply':
        # method1: multiply() and sum()
        for k in range(channels):
            for i in range(img_v):
                for j in range(img_h):
                    filtered_image[i, j, k] = np.sum(
                        np.multiply(img_pad[i:i+2*pad_v+1, j:j+2*pad_h+1, k], _filter))

    elif convmode == 'asstride':
        # method2: as_strided() and tensordot()
        img_pad_v, img_pad_h = img_pad.shape[:2]
        shape = np.array([img_pad_v - krnl_v + 1, img_pad_h - krnl_h + 1, krnl_v, krnl_h])
        strides = np.array([img_pad_h, 1, img_pad_h, 1]) * img_pad.itemsize * channels
        for i in range(channels):
            img_stride = np.lib.stride_tricks.as_strided(img_pad[:, :, i], shape=shape, strides=strides)
            filtered_image[:, :, i] = np.tensordot(img_stride, _filter, axes=[(2, 3), (0, 1)])
    ### END OF STUDENT CODE ####
    ############################

    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
      as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###
    image1_low = my_imfilter(image1, filter)
    image2_low = my_imfilter(image2, filter)
    image2_high = image2 - image2_low

    low_frequencies = np.clip(image1_low, 0, 1)
    high_frequencies = np.clip(image2_high+0.5, 0, 1)
    hybrid_image = np.clip(image1_low + image2_high, 0, 1)

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
