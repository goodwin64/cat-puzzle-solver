import cv2
import numpy as np
from find_contours import find_contours, show_contours
from shape_matcher import match_shapes, visualize_match

# Read the image
image_with_shapes = cv2.imread('/Users/maxdonchenko/cat-puzzle-solver/IMG_20241006_174328.jpg')
image_with_frame = cv2.imread('/Users/maxdonchenko/cat-puzzle-solver/IMG_20241006_174350.jpg')

resized_image_with_shapes = cv2.resize(image_with_shapes, (0, 0), fx=0.5, fy=0.5)
resized_image_with_frame = cv2.resize(image_with_frame, (0, 0), fx=0.5, fy=0.5)

contours_with_shapes = find_contours(resized_image_with_shapes)
contours_with_frame = find_contours(resized_image_with_frame, cv2.RETR_LIST)

frame_contour = min(contours_with_frame, key=cv2.contourArea)
shape_contours = contours_with_shapes

# Try to match shapes to the frame
matches = match_shapes(frame_contour, shape_contours)
print("matches:", matches)

# Visualize matches (for debugging)
for match in matches:
    shape_index, angle, scale = match
    matched_image = visualize_match(resized_image_with_frame, frame_contour, shape_contours[shape_index], match)
    print("matched_image:", matched_image)
    cv2.imshow(f"Matched Shape {shape_index}", matched_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
