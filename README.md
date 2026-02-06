# Week-01-Assignment---Sunnat-Po-latov---220063
!pip install pillow opencv-python
from PIL import Image
import matplotlib.pyplot as plt

# Open image using PIL
img_pil = Image.open('/Assignment/sample_image/image/images.jpeg')

# Print image information
print("Type:", type(img_pil))
print("Size:", img_pil.size)
print("Mode:", img_pil.mode)

# Display image
plt.imshow(img_pil)
plt.axis("off")
plt.show()

import numpy as np

img_np = np.array(img_pil)

print("Shape:", img_np.shape)
print("Data type:", img_np.dtype)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img_mpl = mpimg.imread('/Assignment/sample_image/image/images.jpeg')

print("Shape:", img_mpl.shape)

plt.imshow(img_mpl)
plt.axis("off")
plt.show()
from skimage import io, img_as_float
import matplotlib.pyplot as plt

img_sk = io.imread('/Assignment/sample_image/image/images.jpeg')
img_float = img_as_float(img_sk)

print("Min pixel value:", img_float.min())
print("Max pixel value:", img_float.max())

plt.imshow(img_float)
plt.axis("off")
plt.show()
import cv2
import matplotlib.pyplot as plt

# Read grayscale and color images
img_gray = cv2.imread('/Assignment/sample_image/image/images.jpeg', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('/Assignment/sample_image/image/images.jpeg', cv2.IMREAD_COLOR)

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
edges = cv2.Canny(img_gray, 100, 200)

plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()
import glob
import cv2
import matplotlib.pyplot as plt

path = '/Assignment/sample_image/image/images.jpeg'

for file in glob.glob(path):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(file)
    plt.axis("off")
    plt.show()

    Q: Why does Matplotlib treat images as NumPy arrays?

Matplotlib treats images as NumPy arrays because images are numerical data. Representing images as arrays allows efficient mathematical operations and easy integration with scientific computing libraries.

Q: Why is using astype(float) discouraged compared to img_as_float?

Using astype(float) only changes the data type without normalizing pixel values. img_as_float correctly rescales values to the range [0, 1], which is required for consistent image processing.
