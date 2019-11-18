# **Traffic Sign Recognition** 

## Objective

### In this project, the primary purpose is to use the knowledge about deep neural networks and convolutional neural networks to classify traffic signs. A Model is trained specifically to classify traffic signs from the German traffic board.

---

#### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUpImages/WriteUp_ImageClassifications.jpg "Training set Classifications"
[image2]: ./WriteUpImages/WriteUp_ImageLabels.jpg "Imported Image Labels"

[image3]: ./WriteUpImages/WriteUp_PostProcessImage.jpg "Post Processed traffic image"

[image4]: ./WriteUpImages/WriteUp_NormalizedImage.jpg "Normalized Traffic Sign Image"

[image5]: ./WriteUpImages/LeNet_Original_Image.jpg "LeNet Architecture Image"

[image6]: ./WriteUpImages/WriteUp_TestImages.png "German Traffic Test Images"

[image7]: ./WriteUpImages/WriteUp_TestResults.jpg "Test Results"



---
### Loading Data for Validation and testing
To begin with, a set of data images are load through a pickle file, provided with the exercise project.

The pickled data is a dictionary with 4 key/value pairs:

* 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
* 'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
* 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES.

![alt text][image2]
**Loaded traffic sign images with their respective labels**

### Training image classifications
The images read through the pickle files are classified over 43 various classes, each with different amount of images.

![alt text][image1]
**Training set data classifications**

### Preprocessing the Dataset

The dataset primarily would be ideal to have equal variance across the range. This is best achieved through a random uniform distribution to generate different parameters for translating and shearing the images. The main aspects included in the transformation of the images are:
* Rotation of the image on a preset angle.
* Affine Transformation from the openCV library to shear / warp the image across two defined points.

Furthermore, further image enhancements such as CLAHE Enhancements, scaling the image and cropping the image to avoid image boundary disturbances were implemented to further improve the training results.

![alt text][image3]
**Post Process transformed traffic sign image**

### Creating more images for training and validation set

Over the extent of 43 classes of images, each image in the class are reshaped and transformed. These images are susequently concatenated to their respective classes, assisting in improving the euality of the variance to obtain a better training result.
These transformations are resource intensive. Hence the GPU was first used at this instance of the project. The created images are then stored in a pickle file named transformed_data.pickle to ensure a quick and easy reading of the created images for the next steps of training the neural network.

* Number of training examples after image augument = 86430
* Number of testing examples after image augument = 12630
* Number of training labels after image augument = 86430
* Number of testing labels after image augument = 12630

The training, validation and test images are converted to gray images and subsequently normalized.

![alt text][image4]
**Normalized Traffic Sign Image**

### Model Architecture
![alt text][image5]
**LeNet CNN Architecture**
The LeNet-5 architecture consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier.

This architecture is thereby implemented directly as instructed. The filters and strides recommended are mentioned on the website:

https://engmrk.com/lenet-5-a-classic-cnn-architecture/

The logits for the softmax function are determined towards evaluating the cross entropy.

The training of the model is conducted over 70 EPOCHS to be certain of a correct fitting. A conservative learning rate of 0.0009 is set. A reference to the lesson of stochastic gradient descent was essential to gather the principles of obtaining the correct momentum. 

Over the tensorFlow session, the following accuracies were measured for the trained model:
* Train Accuracy = 0.999
* Validation Accuracy = 0.929
* Test Accuracy = 0.915

### Test the Model on New Images

![alt text][image6]
**Traffic Sign images for testing**
The model training and validation illustrating a high softmax probability was thereby tested on a set of traffic sign images (in total: 38) which are shown above.

The results of the image recognition and correctly attaining the correct probability of the sign to its appropriate class is show below.

![alt text][image7]
**Traffic Sign images for testing**

The test was checked only against the top 5 classes in its softmax probability.

### Scope of Improvement

Machine learning and CNN is a field which is constantly on the edge of development and advancements. To keep up actively with the various improvements and their subsequent usages to applications stems to the need and exposure one has in the particular field.

Understanding the limitations and limits of the LeNet architecture for the odd "corner case" would be essential before one decides to role out a function based solely on the veracity of the developers.

-----