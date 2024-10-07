import cv2
import numpy as np

def find_contours(resized_image, mode=cv2.RETR_EXTERNAL):
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 3)

    edged_detected_image = cv2.Canny(blurred_image, 80, 200)

    # dilation 
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(edged_detected_image, kernel, iterations=1)

    # Apply threshold to create a binary image
    _, thresh = cv2.threshold(dilated_image, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)

    # filter out small contours
    contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

    return contours

def show_contours(image, contours):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show only contours without original image
    contours_only = np.zeros_like(gray)

    cv2.drawContours(contours_only, contours, -1, (255), 2)

    # let cv2 imshow show a single frame with 3 images
    cv2.imshow('Contours Only', cv2.hconcat([gray, contours_only]))

    cv2.waitKey(0)
