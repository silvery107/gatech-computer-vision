from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FIG_H = 15
FIG_W = 18
DTYPE = 'float32'


np.asarray()

img = cv2.imread('CityView.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img)
plt.show()
cv2.imshow("hh",img)
cv2.waitKey(0)
print(np.shape(img))
np.max(img)-np.min(img)

