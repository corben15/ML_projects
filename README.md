# ML_projects
Various machine learning projects for classification of different image datasets.

## perceptron

Dataset: Subset of MNISTdataset (https://en.wikipedia.org/wiki/MNIST_database).

This code implements a simple multiclass logistic regressor for digit image recognition. The classifier takes in a handwritten numerical digit from 1 to 5 and classify it to the respective digit. The weights for the discriminant function were learned through stochastic gradient descent.

## nn_np
Dataset: MNISTdataset (https://en.wikipedia.org/wiki/MNIST_database).

This code implementes a deep neural network (NN) with two hidden layers for image digit classification in the MNIST dataset3(MNIST Database). This dataset contains digits zero through nine. The greyscale images of the digits are 28x28, which could be represented as a 784x1 vector of integers between 0 and 255. These integers were normalized simply by dividing by 255, no other normalization was performed. This vector with shape 784x1 will be the input to the neural network. The data was already split into a training data set containing 50000 images of digits and a test data set containing 10000 images of digits. The corresponding labels for the digit images were stored in train_label.txt and test_label.txt files respectively. This code trains a NN with two hidden layers and 100 nodes in each layer, and one output layer with 10 nodes, which corresponded to the 10 digit classes. No regularization was required to be added to the hidden layers of the neural network or the output layer.

## cnn_tf
Dataset: CIFAR-10 dataset

This dataset contains 50000 training images and 5000 test images consisting of 10 different classes of objects, each set of images was provided with a set of corresponding image labels, they are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.  The labels of the images were digits 0 through 9 which represented the actual classification. Each image is a RGB image with dimensions are 32x32x3. Normalization of the images was performed on a batch bases and was incorporated into the TensorFlow computational graph.

This code to trains a CNN with three convolutional layers, and two pooling layers. From the last convolutional layer we had to add a vectorization layer and one fully connected layer so that the output could consist of 10 nodes to apply the SoftMax function to and determine the predicted classification of the object in the image.
