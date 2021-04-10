import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image, im_range
from student_code import my_imfilter, create_hybrid_image

image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')

cutoff_frequency = 7
filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1, sigma=cutoff_frequency)
filter = np.dot(filter, filter.T)

low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1, image2, filter)
vis = vis_hybrid_image(hybrid_image)

save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)