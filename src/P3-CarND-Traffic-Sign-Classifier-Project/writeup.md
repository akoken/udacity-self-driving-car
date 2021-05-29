# **Traffic Sign Recognition**

## Writeup



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_vis.png "Training Visualization"
[image2]: ./examples/valid_vis.png "Validation Visualization"
[image3]: ./examples/test_vis.png "Test Visualization"
[image4]: ./examples/rand_data.png "Random Traffic Signs"
[image5]: ./examples/preprocessed.png "Preprocessed Traffic Signs"
[image6]: ./examples/loss.png "Training Loss"
[image7]: ./examples/accuracy.png "Training Accuracy"
[image8]: ./examples/sample_traffic_signs.png "Traffic Signs found on the web"
[image9]: ./examples/top_5_probabilities.png "Top 5 Probabilities"
[image10]: ./examples/sign_vis_original.png "Traffic Sign Original"
[image11]: ./examples/feature_map_visualization.png "Feature Map Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here 5 random traffic signs are shown:
![alt text][image4]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution between classes.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the images to YUV color space and took the Y channel as mentioned in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Next, I applied histogram equalization to standardize lighting in the dataset. Its because some images are brighter and others very dim.

After that I normalized the dataset to achieve zero mean and equal variance. This helps the optimization algorithm converge faster thereby saving computational time.

As you can see from the visualization above, the training set is very unbalanced. The number of images per class varies greatly in each of the datasets. This could lead to the fact, that the network is biased towards those categories containing more samples. I used data augmentation process in order to prevent this.

Following transformations applied:

* rotation_range: rotates the image randomly with maximum rotation angle, 15.
* zoom_range: zooms into the image. A random number is chosen in the range, [1-zoom_range, 1+zoom_range].
* shear range: Shear Intensity (Shear angle in counter-clockwise direction as radians)
* width_shift: shifts the image by the fraction of the total image width, if float provided.
* height_shift: shifts the image by the fraction of the total image height, if float provided.

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 normalized image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x60 	|
| RELU					|		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x60
|RELU
| Max pooling	      	| 2x2 stride,  outputs 12x12x60 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x30 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x30
|RELU
| Max pooling	      	| 2x2 stride,  outputs 4x4x30 				|
| Flatten					|					4x4x30 -> 480							|
| Fully connected    | 480 -> 500 |
| RELU					|												|
| Dropout     | keep_prob = 0.5    |
| Fully connected    | 500 -> 43 |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer to minimize the cross entropy loss. I did not implement learning rate decay since Adam Optimizer already handles [learning rate optimization](https://arxiv.org/pdf/1412.6980v8.pdf). You can find the hyperparameters used to train the model below:

| Hyperparameter      | Value	 |
|:--------------:|:-------:|
| Learning Rate         | 0.0005     |
| Epochs     | 5 	   |
| Batch Size				 | 128		 |
| Steps per epoch	 | 2000  |
| Dropout | 0.5  |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 99.3%
* test set accuracy of 98%

![alt text][image6] ![alt text][image7]

If an iterative approach was chosen:
*  First I used the LeNet architecture as a base model. I chose it because it worked well for number images such as MNIST dataset. I thought that it could be a good start for classifying traffic sign images.
* LeNet model could not give the satisfactory result I expected. Validation accuracy was limited to 90%.
* By tweaking the hyperparameters many times and applying data augmentation, I managed to increase test accuracy to 93% which is enough for project goal. From my Deep Learning ND experience, I know that I need deep architecture to extract more features. Therefore I dropped LeNet and implemented my TrafficSignNet architecture mentioned above. For regularization, I used dropout layer in fully connected layer.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]

The first two images might be difficult to classify because they have green background. The third and last images are warped which causes another challange to the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit (120 km/h)    		| Speed Limit (120 km/h)   									|
| Speed Limit (30 km/h)     			| Speed Limit (30 km/h) 										|
| Right of way at the next intersection					| Right of way at the next intersection											|
| Priority Road	      		| Priority Road					 				|
| Stop			| Stop      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

```
TopKV2(values=array([[  9.99996543e-01,   2.72532202e-06,   6.14765213e-07,
          9.45049479e-08,   1.77915260e-09],
       [  9.99909759e-01,   9.02740139e-05,   3.77437803e-10,
          8.68856515e-11,   2.34490934e-11],
       [  9.99999523e-01,   4.92643551e-07,   1.07224417e-10,
          7.05387207e-11,   9.66537614e-12],
       [  1.00000000e+00,   8.64507660e-15,   1.47395740e-15,
          2.18842807e-16,   2.11591702e-16],
       [  1.00000000e+00,   2.23402328e-08,   2.19024399e-09,
          1.57754942e-09,   8.98906893e-10]], dtype=float32), indices=array([[ 8,  5,  7,  0,  2],
       [ 1,  2,  5,  0,  4],
       [11, 30, 27, 21, 28],
       [12, 38, 40, 14, 17],
       [14, 12,  5,  4,  2]], dtype=int32))

```

For the first image, the model is sure that this is speed limit 120km/h sign (probability of 0.999999). The top five soft max probabilities were

![alt text][image9]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

As we can see from the visualization of second convolutional layer, shape is the key feature.

![alt text][image10]
![alt text][image11]