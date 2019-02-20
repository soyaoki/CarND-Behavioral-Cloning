# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

***  Main *** 
* model.py containing the script to create and train the model featured NVIDIA's model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

*** Sub *** 
* model_tl.py containing the script to create and train the model using transfer learing ( InceptionV3 )
* model_tl.h5 containing a trained convolution neural network

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

And for the code works, you have to install pydot and graphviz.
```sh
pip install pydot graphviz
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 1x1, 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 191-212). 

1x1 convolution layer as first layer (code line 198) and ELU as activation funtion (code line 198 etc.) are added to the model to introduce nonlinearrity.

And the model includes The model contains dropout layers in order to reduce overfitting (code lines 207) and the data is normalized in the model using a Keras lambda layer (code line 194).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 224-225). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 221).

And loss function is MSE. So to derease error between predicted and true steering angle, the optimizer optimize model parameters.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

[Training data recored at track1](https://youtu.be/GsNV_fHJyI0)

[Training data recored at track2](https://youtu.be/bT1zx2h4iok)

[Training data ih here.](https://drive.google.com/drive/folders/16eYroYA0rcpTim1paH6-xv5VcyREoag-?usp=sharing)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

I thought this model might be appropriate because this was developed for autonomous driving.

First of all, 1x1 conv. layer ELU and dropout were added to the model. It could make the model to introduce nonlinearrity and not to overfit .

Next, the model was trained using dataset recorded at track1. Following image shows training result in 5 epoch.

![alt text](./training_results_5epoch.png)

Then, the model was tested two track.

[Tested at track1](https://youtu.be/FCUiHn886-c)

[Tested at track2](https://youtu.be/SQEpGinTu6s)

The model drived well at track1(scene included in training data). But it didn't work at all at track2 (scene not included in training data).

So the model was trained again using dataset recorded at track2. Following image shows training result in 5 epoch.

![alt text](./training_results_track2.png)

Then, the model was tested two track.

[Tested at track1](https://youtu.be/pyhaYlEJpZg)

[Tested at track2](https://youtu.be/Cl2_GEv3AM0)

Finally, the model almost drove autonomously at two track.

#### 2. Final Model Architecture

Here is a visualization of the architecture.

![alt text](./model.png)

#### 3. Creation of the Training Set & Training Process

This figure shows distributions of training and validation data. 

![alt text](./train.png)
![alt text](./validation.png)

The distributions were not flat. So this time, data augmentation was implemented especially to increase data which steering angle value was not around zero (model.py line 35-70).

Specificly, when steering angle was more 0.1 or less -0.1, center, left and right images (and every flipped images) were added to dataset. This increased number of data data which steering angle value was not around zero by six times.

![alt text](./images_aug.png)

And then preprocessing this data by cropping was implemented because sky and car hood seemed not to be useful. The cropped image is shonw bellow.

![alt text](./images.png)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

|             | track1 | track2 |
|:-----------:|:---------:|:----------:|
| All data    | 24317 | 5108 | 
| Train data | 19453 | 4086 |
| Valid data | 4864 | 1022 |

#### 4. Extracted Features

Extracted features at first layer in the model are shown below. It implied that the model got an idea which white lane is useful to predict steering angle.

![alt text](./hidden_layer_output.png)

### Transfer Learninig (Inception V3)

And I implemented transefer learning by Inception V3. Code is "model_tl.py".

In lecture, it was said that features from start of convolution layers will be good for case which training data is different and small. So this time, new network (only fully connected layer) was trained using extracted features from first part of Inception V3.

![alt text](./model_tl.png)

Inception V3 needs input image of 299x299. So resizing image layer was added and the resized image is shown bellow.

![alt text](./images_tl.png)

Movies which the model drove two track is following.

[Tested at track1]()

[Tested at track2]()

Extracted features at first layer in the model are shown below. It seems that the model couldn't extract good features for predicting steering angle....

![alt text](./hidden_layer_output_tl.png)

### Reinforcement learning

I REALLY WANTED TO IMPLEMENT...BUT I DON'T KNOW HOW I IMPLEMENT....