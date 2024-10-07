import cv2
import numpy as np

def rotate_contour(contour, angle):
    """Rotate a contour by a given angle."""
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.transform(contour.reshape(1, -1, 2), M)
    return rotated.reshape(-1, 1, 2)

def scale_contour(contour, scale):
    """Scale a contour by a given factor."""
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    scaled = contour.astype(np.float32) - [cx, cy]
    scaled = scaled * scale + [cx, cy]
    return scaled.astype(np.int32)

def match_shapes(frame_contour, shape_contours, threshold=0.01):
    """
    Try to fit shape contours into the frame contour.
    Returns a list of (shape_index, angle, scale) tuples for matched shapes.
    """
    matches = []
    frame_area = cv2.contourArea(frame_contour)
    
    for i, shape in enumerate(shape_contours):
        shape_area = cv2.contourArea(shape)
        scale_factor = np.sqrt(frame_area / shape_area)
        
        for angle in range(0, 360, 5):  # Check every 5 degrees
            rotated_shape = rotate_contour(shape, angle)
            scaled_shape = scale_contour(rotated_shape, scale_factor)
            
            match = cv2.matchShapes(frame_contour, scaled_shape, cv2.CONTOURS_MATCH_I2, 0)
            if match < threshold:
                matches.append((i, angle, scale_factor))
                break  # Move to the next shape once a match is found
    
    return matches

def visualize_match(frame, frame_contour, shape, match):
    """Visualize a matched shape within the frame."""
    shape_index, angle, scale = match
    rotated_shape = rotate_contour(shape, angle)
    scaled_shape = scale_contour(rotated_shape, scale)
    
    result = frame.copy()
    cv2.drawContours(result, [frame_contour], 0, (0, 255, 0), 2)
    cv2.drawContours(result, [scaled_shape], 0, (0, 0, 255), 2)
    
    return result