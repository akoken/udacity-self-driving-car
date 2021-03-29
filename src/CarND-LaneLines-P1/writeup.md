**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./writeup_output/grayscale.jpg "Grayscale"
[image2]: ./writeup_output/blur.jpg "Blur"
[image3]: ./writeup_output/canny.jpg "Canny"
[image4]: ./writeup_output/roi.jpg "ROI"
[image5]: ./writeup_output/hough.jpg "Hough Transform"
[image6]: ./writeup_output/final.jpg "Final"

---

### Reflection

### 1. Pipeline Description

My pipeline consisted of 6 steps.

#### 1. Convert original colored image to grayscale using the `grayscale()` helper function.

![alt text][image1]

#### 2. Apply Gaussian smoothing with kernel size of 9 by using the `gaussian_blur()` helper function.

![alt text][image2]

#### 3. Detect edges with Canny edge detection by using the `canny()` helper function.

![alt text][image3]

#### 4. Define vertices and select region of interest by using the `region_of_interest()` helper function. Then apply mask to the image resulting from the previous step.

![alt text][image4]

#### 5. Find lines with Hough Transform by using the `hough_lines()` helper function.

![alt text][image5]

#### 6. Determine the left and right lane and overlay on original image by using `draw_lines()` and `weighted_image()` helper functions.

![alt text][image6]

In order to draw a single line on the right and left lanes, I had to modify `draw_lines()` function. Hough Transform detects line segments which are represented by a pair of coordinates. I collected all these coordinates since they belong to the same left or right lane. Then extrapolate a longer lane going from bottom of the image up to a certain vertical level of the image. I seperated left and right line segments by calculating slope of them. Using `np.polyfit()` helper function with these coordinates gave me left and right lane line equations (y=mx + b). Then I found (x, y) coordinates to draw extrapolated left and right lanes.


### 2. Potential shortcomings


- Pipeline expects lanes to be straight. When it comes to curve lanes, it fails.
- Light conditions may prevent pipeline working properly. Edges could not be detected.


### 3. Possible improvements

- Making the pipeline more robust for curved lanes.
- Hyperparameters for canny edge detection and hough transform could be optimized.
- Red color of detected lane line could be darker and solid.
