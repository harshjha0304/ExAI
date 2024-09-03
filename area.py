import cv2
import numpy as np

# Load the image
image_path = './data/wound_dataset/test/predictions/2019-12-19 01%3A53%3A15.480800/ba4755da013537410deae3f8386d08ba_0.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert the image to binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the area of the white portion
white_area_pixels = 0
for contour in contours:
    white_area_pixels += cv2.contourArea(contour)

# pixel to centimeter ratio
pixel_to_cm_ratio = 0.026458333333719  # cm per pixel
white_area_cm2 = white_area_pixels * (pixel_to_cm_ratio ** 2)

print(f'The area of the white portion is: {white_area_cm2:.2f} cm²')

# If you want to visualize the contours (optional)
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
