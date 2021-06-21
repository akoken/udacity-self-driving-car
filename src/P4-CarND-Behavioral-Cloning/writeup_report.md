# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane.jpg "Center Lane"
[image2]: ./examples/clockwise.jpg "Clockwise Image"
[image3]: ./examples/right_to_center.jpg "Recovery To Center"
[image4]: ./examples/flipped_image.png "Flipped Image"
[image5]: ./examples/history.png "Loss"
[image6]: ./examples/preprocessed_image.png "Pre processed Image"
[image7]: ./examples/visualization.png "Visualization of Activations"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network layers. The first 3 convolutional layers have 5x5 filter sizes and the last 2 have 3x3 filter sizes. Each CNN layer has different depth changing between 24 and 64 (model.py lines 72-76)

The model includes RELU layers to introduce nonlinearity (code lines 72-83), and the data is normalized in the model using a Keras lambda layer (code line 71).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 78-82).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 62-64). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I used an Adam(Adaptive Moment Estimation) optimizer with initial learning rate 0.0001 (model.py line 101). To determine overfitting or underfitting, I used the validation set results. Here are my hyperparameters:

```
learning_rate = 0.0001
batch_size = 128
epochs = 10
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because the NVIDIA team already controlled a self driving car in real world.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting:

* Reduced the learning rate.
* Added dropout after fully connected layers.
* Increased the dataset.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as first sharp left turn. To improve the driving behavior in these cases, I captured more data in these spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-84) consisted of a convolution neural network with the following layers and layer sizes:
```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```


#### 3. Creation of the Training Set & Training Process

#### Image Processing

I converted RGB images to YUV color space based on NVIDIA paper and then added 3x3 Gaussian Blur to reduce noise. The output image resolution was 320x160. I also added image preprocessing to drive.py.

![alt text][image6]

Note: Cropping and normalization were done in Lambda layer.

#### Data Collect
To capture good driving behavior:

* 3 laps clockwise center lane driving.
* 2 laps counter clockwise center lane driving.
* 1 lap vehicle recovering from the left and right sides of the road back to center.

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data set, I also flipped images and angles thinking that this would help generalize the model. For example, here is an image that has then been flipped:

![alt text][image4]


After the collection process, I had 7085 number of data points. I increased the number of images by flipping images. I also used generator to generate data for training to prevent memory leak.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10.

![alt text][image5]

#### 4. Visualization Activations
The visualize.py file contains the code visualizing the activations. Here is a visualization of third layer.

![alt text][image1]
![alt text][image7]