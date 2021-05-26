import cv2
import numpy as np

def combine_binarized_thresholded_img(img, sobelx_thresh, sobely_thresh, magthresh, sthresh, vthresh, sobel_kernel=3):
    # Calculate the x and y gradients using Sobel
    sobelx = abs_sobel_thresh(img, orient="x", sobel_kernel = sobel_kernel, thresh=sobelx_thresh)
    sobely = abs_sobel_thresh(img, orient="y", sobel_kernel = sobel_kernel, thresh=sobely_thresh)

    # Calculate the magnitude gradients using Sobel
    mag = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=magthresh)

    # Calculate color gradients
    color = color_thresh(img, sthresh=sthresh, vthresh=vthresh)

    # Combine and output
    output = np.zeros_like(color)
    output[( (sobelx == 1) & (sobely == 1) & (mag == 1))  | (color == 1)] = 1
    return output

def color_thresh(img, sthresh=(0, 255), vthresh=(0, 255)):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Apply a threshold to the S channel
    s_channel = hsv[:,:,1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    # Apply a threshold to the V channel
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    # Return a binary image of threshold result
    output = np.zeros_like(s_channel)
    output[( s_binary == 1) & (v_binary == 1)] = 1

    # Return binary output image
    return output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)

    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    # Return binary output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the magnitude of the gradient
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Scale to 8-bit (0-255) then convert to type = np.uint8
    scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))

    # Create a mask of 1's where the scaled gradient magnitude is within the given thresholds
    mag_binary = np.zeros_like(scaled_gradmag)
    mag_binary[(scaled_gradmag >= thresh[0]) & (scaled_gradmag <= thresh[1])] = 1

    # Return binary output image
    return mag_binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate direction of gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Create a mask of 1's where the gradient direction is within the given thresholds
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return binary output image
    return dir_binary