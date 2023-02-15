# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
original_img = cv2.imread("whale.jpg")

# Convert the input image into grayscale
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

imgShape = gray_img.shape
height = imgShape[0]
width = imgShape[1]

# Display the original histogram
store = np.zeros((256,), dtype=np.int32)
saved = np.zeros((256,), dtype=np.int32)
for i in range(height):
    for j in range(width):
        k = gray_img[i, j]
        store[k] += 1
print('Store Value')
print(store)
x = np.arange(0, 256)
plt.bar(x, store, color="r", align="center")
plt.title('Org hist')
plt.show()

# Perform the cumulative distribution function
sum_hist = np.cumsum(store)
print('Sum Histogram')
print(sum_hist)
print('Sum Histogram')
print(sum_hist[255])
# Show the plot of cumulative distribution
x = np.arange(0, 256)
plt.bar(x, sum_hist, color="r", align="center")
plt.title('cdf hist')
plt.show()

# Get the new pixel value from the cumulative distribution
for x in range(0, 256):
    saved[x] = sum_hist[x] * 255 / sum_hist[255]
# Show the histogram of new pixel value
x = np.arange(0, 256)
plt.bar(x, saved, color="r", align="center")
plt.title('new pixel hist')
plt.show()

# Write new pixel value into the old image
for i in range(height):
    for j in range(width):
        k = gray_img[i, j]
        gray_img[i, j] = saved[k]
# Display the new image histogram
store = np.zeros((256,), dtype=np.int32)
saved = np.zeros((256,), dtype=np.int32)
for i in range(height):
    for j in range(width):
        k = gray_img[i, j]
        store[k] += 1
print('Stored Value')
print(store)
x = np.arange(0, 256)
plt.bar(x, store, color="r", align="center")
plt.title('Output hist')
plt.show()

# Show and save the result
cv2.imshow('Display the result', gray_img)
cv2.imwrite("result.jpg", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
