# **Behavioral Cloning** 
---

For compiled models of both tracks, see "Releases" tab
For running see [Instructions](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/README.md)
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/graph.png "Model Visualization"
[image2]: ./writeup_images/center.jpg "Center driving"
[image3]: ./writeup_images/left.jpg "Left"
[image4]: ./writeup_images/right.jpg "Right"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./writeup_images/center_flipped.jpg "Center Flipped"
[image7]: ./writeup_images/leftright.png "LR"
[image8]: ./writeup_images/track2.png "LR"
[image9]: ./writeup_images/tb1.png "TensorBoard 1"
[image10]: ./writeup_images/tb2.png "TensorBoard 2"
[image11]: ./writeup_images/p1.png " "
[image12]: ./writeup_images/p2.png " "
[image13]: ./writeup_images/p3.png " "
[image14]: ./writeup_images/p4.png " "
[image15]: ./writeup_images/p55.png " "
[image16]: ./writeup_images/p6.png " "
[image17]: ./writeup_images/training.png " "
[image18]: ./writeup_images/p1.png " "
[image19]: ./writeup_images/p1.png " "
[image20]: ./writeup_images/p1.png " "

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The network architecture is shown in TensorBoard Figure below. (file model.py, function nvidia_model)
![alt text][image1]

The network consists of a cropping layer, a normalization layer, 5 convolutional layers and 3 fully connected layers. The first two layers of the network performs image cropping and normalization. Both are hard-coded and not adjusted in the learning process. Performing these two steps in the network allows them to be accelerated via GPU processing. Cropping the images removes horizon and car's hood from the bottom. Both of these parts don't convey any useful information and only confuse the Neurual Netowrk.

This architecture borrows from [Nvidia's PilotNet](https://arxiv.org/pdf/1604.07316v1.pdf). This design was created empirically by Nvidia. They use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

The five convolutional layers are followed by three fully connected layers leading to an output control value which is the inverse turning radius.

Broadly speaking, the convolutional layers relates to feature extraction and the fully connected layers function as a controller. However by training an system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.

#### 2. Attempts to reduce overfitting in the model
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Training was stopped early to prevent overfitting. I started with 20 epoch but in later runs a value between 5 and 10 was returning better results on simulator

Training data was augmented by flipping center image and adding Left/Right camera images. (code line 67-85). Data as collected by driving in both directions on track. This helps generalize the model and prevents overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 162-198). 
#### 3. Model parameter tuning

During full training runs, I used Adam optimizer(model.py line 191). Default learning rate of Adam is 0.001  as provided in the original [paper](https://arxiv.org/pdf/1412.6980v8.pdf). I modified it to a lower value of 0.0001 because this architecture is prone to overfitting. This worked very well on Track 1.

On Track 2 some parts, specially upward curves were troublesome. I collected additional data on the problematic parts and used Classical SGD with learning rate of 0.001 with a pre-trained model. SGD gives finer level control over the learning. This rate is deliberately kept small to help generalize the model on new and smaller dataset, without overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also drive on both side of the track to collect additional data 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
##### Track 1
Following class lecture, I started with a single Dense layer network to make sure my pipeline is working. Then I implemented [Nvidia's PilotNet](https://arxiv.org/pdf/1604.07316v1.pdf) in Keras because it is proven to work in this kind of use case.

I started with Udacity's provided dataset. This dataset is for track one and only overs about half of it in clockwise direction. I had trouble keeping the car straight. It would take left turn and end up in water. Then I implemented cropping layer to remove horizon and car's hood that serves no purpose but to confuse the network. Things started to improve. The cars was going straight on the root, but too straight. It was clear that there is more data going straight and model is overfitting to it.

I implemented code to randomly drop 70% of the samples who's absolute steering angle was less than 0.25. (code line 128). It brought improvement and car made all the way to the bridge, where it swayed left and stuck with the ledge.
![alt text][image11]

Then I brought in Left and Right camera images and used them to teach recovery. For example when vehicle would has drifted right, center camera would see something similar to what right camera saw during training. A hardcoded value of 0.35 was used as recovery angle. (code line 67-85). After this car made all the way after the bridge but failed to take left turn and wound up on the dirt road. I trained for 15 epoch.
![alt text][image12]

At that point I disabled the logic to drop 70% of the low steering angle. The logic behind my decision was that, now for every center image of low steering angle, there are two more images contributed from the Left and Right cameras that have a high correction steering angle +-0.35. This should reduce the bias of going straight. It proved to be a good decision, the car moved paste the dirt road but then end up in water on the next right S turn.
![alt text][image13]
At that point I decided to augment the data. I drove in the apposite direction. I tried avoid going straight and recorded recovery maneuvers. I started with pre-trained model and trained for 10 epochs. Next on my list was to use YUV formmat like Nivida did, but At this point Track 1 was conquered. I could drive autonomously around Track 1 continuously.     

##### Track 2
Similar to Track 1 I collected training data for Track 2. It work well except for two parts of the track where car would drift left and get stuck. First problematic part was the second accent. The second part was the last right turn before the lap would end.

![alt text][image15]
![alt text][image16]

I collected additional data on those areas and augmented it to the rest of Track 2 data and trained together. It made situation worse. Then I changed my approach. I used pre-trained model for Track 2. Switched to classical SGD instead of Adm to keep a low consistent learning rate. Then I trained for the problematic area data for 1 epoch. Success!!! Track 2 was working autonomously continuously. 

#### 2. Final Model Architecture

The final model architecture consists of a cropping layer, a normalization layer, 5 convolutional layers and 3 fully connected layers. The first two layers of the network performs image cropping and normalization. This architecture borrows from [Nvidia's PilotNet](https://arxiv.org/pdf/1604.07316v1.pdf). They use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

Final network architecture is shown in Figure below. (file model.py, function nvidia_model)
 
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. One using clockwise and other anti-clockwise. Here is an example image of center lane driving:

![alt text][image2]

In order to teach the network how to recover if it ever gets off-centre, I used Left and Right camera images. These images show what a recovery looks like starting from Left and Right :

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped center camera images and angles thinking that this would help generalize the model better. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image6]

Before training I randomly shuffled the data set and put 20% of the data into a validation set. 
###### Nvidia inspired approach

![alt text][image17]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 on Track 1 and 6 on Track 2. as evidenced by following figure.
![alt text][image7]
![alt text][image8]

#### 4. TensorBoard Visualization
![alt text][image9]
![alt text][image10]
