

**Advanced Lane Finding Project**

The general steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners_found.png "Found Corners"
[image2]: ./output_images/undistorted.png "Undistorted"
[image3]: ./output_images/undistorted2.png "Undistorted Image"
[image4]: ./output_images/undistorted_binary.png "Undistorted Binary Image"
[image5]: ./output_images/top_down_view0.png "Top Down View"
[image6]: ./output_images/top_down_view.png "Top Down View"
[image7]: ./output_images/top_down_view2.png "Top Down View"
[image8]: ./output_images/histogram.png "Histogram"
[image9]: ./output_images/lane_detected_image.png "Final"
[image10]: ./output_images/final.gif "Final"

## Detailed Project Explanation

![alt text][image10]
---

### Step 1: Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients

The code for this step is contained in the file called `calibration.py`(calibration.py).

Note: I saved the camera calibration results as a pickle file instead of calculating them everytime.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image which you can find an example below.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

![alt text][image1]

### 2. Undistort images
We can apply distortion correction to images using the `cv2.undistort()` function. Here is an example result:

![alt text][image2]

---

### Step 2: Pipeline (single images)

#### 1. Undistort Camera Images

Using camera distortion coefficients calculated from last step, we will undistort all input image stream from the camera before we apply any image processing techniques.

Here is a comparision of original distorted image and undistorted image. As you can see, the effect is pretty small for the human eye:

![alt text][image3]

#### 2. Identify lane lines through edge detection

I implemented this step in the file called `utils.py`(utils.py)). Here's an example of my output for this step.

![alt text][image4]

The binary image above uses different techniques, which we previously saw in the course, such as sobel filter, gradient and alternative color spaces.

#### 3. Perspective Transform

The code for my perspective transform is in lines 51 through 67 in the file `lane.py` (lane.py). I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 4. Identify Lane Line Pixels

The code for this step can be found in `lane.py`(lane.py) (lines 82-151).

**Sliding Window Search**

After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image  where the lane lines stand out clearly.

To identify the lane lines, I used the sliding-window technique to identify lane pixels in the frames. I first take a histogram along all the columns in the image, adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines.

![alt text][image8]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

I implemented this step in lines 188 through 202 in my code in `lane.py`(lane.py).

**Radius of Curvature**
Computing curvature radius for both left and right lines defined by 2nd degree polynomial parameters (left_fit and right_fit, using formulas provided in the course).

After finding the curvature of each line, we simply get the average of the two:
```python
 left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
 right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
 self.radius_of_curvature = (left_curverad+right_curverad)/2
```

**Vehicle Position**
A negative distance from the center of the lane indicates that the vehicle is left of center, while a positive distance indicates that the vehicle is right of center.

To calculate the offset, we calculate the x position of each lane line within the image and subtract the the center of the lane (which we assume is the location of the camera), after converting pixels to meters.
```python
 camera_center = (left_fitx[-1] + right_fitx[-1])/2
 self.line_base_pos = (camera_center-unwarped.shape[1]/2)*xm_per_pix
```

#### 6. Plot lane line detection back on to the image

I implemented this step in lines 168 through 174 in my code in `lane.py`(lane.py). We can use the reverse perspective transform to plot the lane back to the original image. Here is an example:

![alt text][image9]

---

### Pipeline (video)


Here's a [link to my video result](./out_project_video.mp4)

---

### Discussion

This pipeline by no means is perfect. It is still very vulnerable to many abnormalities in real life, such as shaddows, lightings changes, window reflections, faded lane lines, etc. The polynomial fit will also fail especially when lines edges detected are not enough, so lines sometimes warbble or jump around.

If I were to improve this pipeline, I will consider more techniques in smoothing to average the polynomials out. I will try to skip a fit if the lane-fit significantly differs from previous timestamps' fits, until a threshold is reached so I will recalculate the fit. I will also consider more robust techiniques for lane detection, than simple color thresholding and edge detection. For example, a Convolutional Neural Network. The goal will be try to identify lane lines under different lighting conditions.
