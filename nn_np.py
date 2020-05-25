'''
Nicholas Corbett
Deep Learning 6850 Programming Assignment 2

'''

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import pickle
import random

'''
Activation function is the Rectified Linear Unit function.
If the value is greater than zero then it stays the same, if the value is
negative it is set to zero.
'''
def activationFunction(x):
    x[x<0]=0
    return x
'''
SOFTMAX
Description: The softmax function is used in the output layer to change the
    values to probabilities by taking the exponents of each output and then
    normalize each number by the sum of those exponents so the entire output
    vector adds up to one.

inputs: Y - np.array's with shape (Nx1)

output: np.array (Nx1)
'''
def softmax(Y):
    return np.exp(Y) / np.sum(np.exp(Y), axis=0)
'''
forwardPropagation
descript: Takes in a data point and computes the v

inputs: X -> np.array 784x1
        theta-> list of current weight matrices and vector biases
output: Y -> np.array 10x1
        H_list[0]-> np.array 100x1 hidden layer 1 values
        H_list[1]-> np.array 100x1 hidden layer 1 values
'''
def forwardPropagation(X, theta):
    # This code is for when there is only 2 hidden layers
    H1 = activationFunction(np.matmul(theta[0].T,X) + theta[1])
    assert( np.isnan(H1).any() == False )
    H2 = activationFunction(np.matmul(theta[2].T,H1)+ theta[3])
    assert( np.isnan(H2).any() == False )
    H_list = [H1,H2]
    Y = np.matmul(theta[4].T,H1)+theta[5]
    assert( np.isnan(Y).any() == False )

    # The following code can be used for an arbitrary amount of hidden layers
    '''
    H_list2=[]
    for i in range(0,len(theta),2):
        if(i==0):
            H = activationFunction(np.matmul(theta[i].T,X) + theta[i+1])
            assert(np.all(H>=0)) # Sanity Check: H should always be all positive
            H_list2.append(H)
        elif(i==len(theta)-2):
            Y = np.matmul(theta[i].T,H)+theta[i+1]
        else:
            H = activationFunction(np.matmul(theta[i].T,H) + theta[i+1])
            assert(np.all(H>=0)) # Sanity Check: H should always be all positive
            H_list2.append(H)

    assert(len(H_list) == len(H_list2))
    '''
    assert( np.isnan(Y).any() == False )
    return H_list[0],H_list[1],Y


def lossFunction(Y,y_hat):
    loss = 0
    assert(len(Y)==len(y_hat))
    for i in range(len(Y)):
        loss += Y[i]*math.log(y_hat[i],math.e)
    return np.negative(loss)

def computeOutputGradient(y,y_hat):
    y_grad = np.divide(y,y_hat)
    y_grad[ np.isnan(y_grad)]=0
    y_grad[ np.isinf(y_grad)]=0
    return np.negative(y_grad)

def computeSoftmaxZDerivativeZ(y_hat):
    softmaz_z_dz = np.zeros((len(y_hat),len(y_hat)))

    for i in range(len(y_hat)):
        for k in range(len(y_hat)):
            if(i==k):
                softmaz_z_dz[i][k] = y_hat[i]*(1-y_hat[i])
            else:
                softmaz_z_dz[i][k] = -1*y_hat[i]*y_hat[k]
    return softmaz_z_dz

def computeReluZDerivativeZ(H):
    relu_z_dz = np.zeros((len(H),len(H)))
    for i in range(len(H)):
        for j in range(len(H)):
            if(i==j):
                relu_z_dz[i][j] = 1
    return relu_z_dz

def computeW3Gradient(W,y_hat_gradient,y_hat, H2):
    softmax_z_dz = computeSoftmaxZDerivativeZ(y_hat)

    W3_gradient = np.zeros((len(W),len(y_hat)))
    for i in range(len(y_hat)):
        d_y_hat_W_i_list = []
        for k in range(len(W[i])):
            d_y_hat_W_i_list.append((softmax_z_dz[i][k]*H2))

        W3_gradient_i = np.concatenate(d_y_hat_W_i_list,axis=1)
        W3_gradient += y_hat_gradient[i]*W3_gradient_i
    return W3_gradient

def computeW30Gradient(W3_0,y_hat_gradient,y_hat, H1):
    softmax_z_dz = computeSoftmaxZDerivativeZ(y_hat)

    W3_0_gradient = np.zeros((len(y_hat),1))
    for i in range(len(y_hat)):
        d_y_hat_W_i_list = []
        for k in range(len(W3_0)):
            d_y_hat_W_i_list.append(np.array([softmax_z_dz[i][k]]))

        W3_0_gradient_i = np.concatenate(d_y_hat_W_i_list,axis=0)
        W3_0_gradient_i = W3_0_gradient_i.reshape((len(y_hat),1))
        W3_0_gradient += y_hat_gradient[i]*W3_0_gradient_i
    return W3_0_gradient

def computeH2Gradient(W3, y_hat_gradient, y_hat):
    softmax_z_dz = computeSoftmaxZDerivativeZ(y_hat)
    H_gradient = np.zeros((len(W3),1))
    H_gradient = np.matmul(np.matmul(W3,softmax_z_dz),y_hat_gradient)
    return H_gradient

def computeHGradient(W, H_plus1_gradient, H_plus1):
    relu_z_dz = computeReluZDerivativeZ(H_plus1)
    H_gradient = np.zeros((len(W),1))
    H_gradient = np.matmul(np.matmul(W,relu_z_dz),H_plus1_gradient)
    return H_gradient

def computeWGradient(W, H_gradient, H, X):
    relu_z_dz = computeReluZDerivativeZ(H)

    W_gradient = np.zeros((len(W),len(H)))
    for i in range(len(H)):
        d_y_hat_W_i_list = []
        for k in range(len(W[i])):
            d_y_hat_W_i_list.append(relu_z_dz[i][k]*X)

        W_gradient_i = np.concatenate(H[i]*(1-H[i])*d_y_hat_W_i_list,axis=1)
        W_gradient += H_gradient[i]*W_gradient_i
    return W_gradient

def computeW0Gradient(W_0, H_gradient, H, X):
    relu_z_dz = computeReluZDerivativeZ(H)

    W0_gradient = np.zeros((len(H),1))
    for i in range(len(H)):
        d_y_hat_W_i_list = []
        for k in range(len(W_0)):
            d_y_hat_W_i_list.append(np.array([relu_z_dz[i][k]]))

        W0_gradient_i = np.concatenate(d_y_hat_W_i_list,axis=0)
        W0_gradient_i = H[i]*(1-H[i])*W0_gradient_i.reshape((len(H),1))
        W0_gradient += H_gradient[i]*W0_gradient_i
    return W0_gradient

'''
backPropagate
inputs:
    - X -> np.array (784x1) input vector that was forward propagated
    - theta -> list of weight matrices and vectors
    - y_hat_gradient -> output gradient np.array (10x1)
    - y_hat -> output from forwardPropagation np.array (10x1)
    - H_list -> list of hidden layer values in forward propagation

output:
    - gradients -> list of weight matrices and vector gradients with respect to
    theta

'''
def backPropagate(X,theta, y_hat_gradient, y_hat, H_list):
    gradients = []

    W3_gradient = computeW3Gradient(theta[4],y_hat_gradient,y_hat, H_list[1])
    W3_0_gradient = computeW30Gradient(theta[5],y_hat_gradient,y_hat, H_list[1])
    #print(np.amax(W3_gradient))
    #print(np.amax(W3_0_gradient))

    H2_gradient = computeH2Gradient(theta[4], y_hat_gradient, y_hat)
    #print(np.amax(H2_gradient))

    W2_gradient = computeWGradient(theta[2],H2_gradient,H_list[1], H_list[0])
    W2_0_gradient = computeW0Gradient(theta[3],H2_gradient,H_list[1], H_list[0])
    #print(np.amax(W2_gradient))
    #print(np.amax(W2_0_gradient))

    H1_gradient = computeHGradient(theta[2], H2_gradient, H_list[1])
    #print(np.amax(H1_gradient))

    W1_gradient = computeWGradient(theta[0],H1_gradient,H_list[0], X)
    W1_0_gradient = computeW0Gradient(theta[1],H1_gradient,H_list[0], X)
    #print(np.amax(W1_gradient))
    #print(np.amax(W1_0_gradient))


    gradients.append(W1_gradient)
    gradients.append(W1_0_gradient)
    gradients.append(W2_gradient)
    gradients.append(W2_0_gradient)
    gradients.append(W3_gradient)
    gradients.append(W3_0_gradient)
    #gradients.append(H2_gradient)
    #gradients.append(H1_gradient)

    return gradients

def updateWeights(theta, gradients, learning_rates = [.1,.1,.1,.1,.1,.1]):
    assert(len(theta) == len(gradients))
    for i in range(len(theta)):
        theta[i] = theta[i] - learning_rates[i]*gradients[i]
    return theta

# Randomization function for picking Training data
def randomIndexListGenerator(trained_index_set, batch_size, random_index_wait_time = .05):
    random_index_list = []
    while(len(random_index_list)<batch_size):
        random_index = random_index = random.randint(1,50000)
        if(random_index not in trained_index_set):
            random_index_list.append(random_index)
            trained_index_set.add(random_index)
    return random_index_list


def main():

    # Code to access files
    dir_path = os.path.dirname(os.path.realpath(__file__))

    dir_path_train_data = dir_path + "/data_prog2Spring18/train_data"
    train_labels_file = open(dir_path + "/data_prog2Spring18/labels/train_label.txt", 'r')
    train_labels = train_labels_file.read()
    train_labels = train_labels.split()
    train_labels = [int(digit) for digit in train_labels]
    print("Len Training Set",len(train_labels))

    dir_path_test_data = dir_path + "/data_prog2Spring18/test_data"
    test_labels_file = open(dir_path + "/data_prog2Spring18/labels/test_label.txt", 'r')
    test_labels = test_labels_file.read()
    test_labels = test_labels.split()
    test_labels = [int(digit) for digit in test_labels]
    print("Len Test Set",len(test_labels))
    print()


    # Initialize Weight Vectors
    np.random.seed(0)
    W1 = np.random.normal(0,.05, (784,100))
    W1_0 = np.zeros((100,1))
    np.random.seed(1)
    W2 = np.random.normal(0,.05,(100,100))
    W2_0 = np.zeros((100,1))
    np.random.seed(2)
    W3 = np.random.normal(0,.05,(100,10))
    W3_0 = np.zeros((10,1))


    theta = [ W1, W1_0, W2, W2_0, W3, W3_0]
    learning_rates = [.05, .05, .05, .05, .05, .05]

    M = 50000              # M is the number of data points in training set
    K = 10                 # K is the number of classes
    batch_size = 50         # Batch Size


    update_count = 0


    trained_index_set = set()   # Set to keep track of digits already trained
    train_index_list = []       # List of random digit indices to train

    loss_list = []
    accuracy_dict = {"train":[], "test":[]}
    digit_accurracies = {   0: [],
                            1: [],
                            2: [],
                            3: [],
                            4: [],
                            5: [],
                            6: [],
                            7: [],
                            8: [],
                            9: [],
    }


    train_iter = 0
    batch_counter = 0

    start_time = time.time()
    while(1):

        train_index_list = randomIndexListGenerator(trained_index_set,batch_size)
        #train_index_list = list(range(batch_counter*batch_size+1,batch_counter*batch_size+batch_size+1))
        #batch_counter += 1

        gradients_sum = []

        # To only train 1000 digits
        if(train_iter==3000): break

        loss_sum = 0
        training_correct_count = 0
        for digit_index in train_index_list:
            time.sleep(.1)
            train_iter += 1
            file_name = dir_path_train_data + "/" + str(digit_index).rjust(5,"0") + ".jpg"
            image = mpimg.imread(file_name)

            '''
            imgplot = plt.imshow(image)
            plt.colorbar()
            plt.show()
            '''


            # Change the image into a 784x1 vector and normalize
            x = image.flatten()
            x = np.divide(x,255)
            x = np.reshape(x, (784,1))

            y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
            y[train_labels[digit_index-1]] = 1



            forward_prop = forwardPropagation(x,theta)
            H_list = forward_prop[:-1]
            y_hat = forward_prop[2]
            y_hat = softmax(y_hat)

            max_node = 0
            max_value = 0
            for node in range(len(y_hat)):
                if(y_hat[node]>max_value):
                    max_value = y_hat[node]
                    max_node = node

            if(max_node == train_labels[digit_index-1]):
                training_correct_count += 1





            y_hat_gradient = computeOutputGradient(y,y_hat)

            # If it is the first train_iter Initialize gradients
            gradients = backPropagate(x,theta, y_hat_gradient, y_hat, H_list)

            update_count += 1   # Keeps track of additions to gradient
            for grad in range(len(gradients)):
                if(len(gradients_sum)<len(gradients)):
                    gradients_sum.append(np.divide(gradients[grad],batch_size))
                else:
                    gradients_sum[grad] += np.divide(gradients[grad],batch_size)

            loss_sum += lossFunction(y,y_hat)


        assert(update_count == batch_size)
        theta = updateWeights(theta, gradients, learning_rates)
        accuracy_dict["train"].append(float(training_correct_count)/batch_size)
        update_count = 0



        if(train_iter%(batch_size*10)==0):
            learning_rates = [r/2 for r in learning_rates]



        if(train_iter%10==0):
            loss_list.append(loss_sum/batch_size)

            #input()
            print("train_iter:\t",train_iter)
            print("Assign Rand index to:",digit_index)
            print("Digit:", train_labels[digit_index-1])
            #print("Index:", digit_index)
            print("L Rate:\t", learning_rates)
            print("Loss:\t", loss_sum/batch_size)

            #Simple Error Calc
            correct_count = 0
            actual_values_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            predict_values_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            predict_correct_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            test_size = 5000
            for t in range(1,test_size+1):
                file_name = dir_path_test_data + "/" + str(t).rjust(5,"0") + ".jpg"
                image = mpimg.imread(file_name)



                # Change the image into a 784x1 vector and normalize
                x = image.flatten()
                x = np.divide(x,255)
                x = np.reshape(x, (784,1))

                y = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
                y[test_labels[t-1]] = 1

                test_forward = forwardPropagation(x,theta)
                y_hat_test = test_forward[2]

                max_node = 0
                max_value = 0
                for node in range(len(y_hat_test)):
                    if(y_hat_test[node]>max_value):
                        max_value = y_hat_test[node]
                        max_node = node

                actual_values_dict[test_labels[t-1]] += 1
                predict_values_dict[max_node] += 1
                #print()
                if(max_node == test_labels[t-1]):
                    correct_count += 1
                    predict_correct_dict[test_labels[t-1]] += 1


            print("Accuracy:\t", float(correct_count)/test_size)
            print("Actual:\t\t",actual_values_dict)
            print("Predicted:\t",predict_values_dict)
            print("Correct:\t",predict_correct_dict)
            accuracy_dict["test"].append(float(correct_count)/test_size)
            for d in range(10):
                digit_accurracies[d].append(float(predict_correct_dict[d])/actual_values_dict[d])

            print("=========================")



    end_time = time.time()
    print("Training Took:", end_time-start_time)


    plt.plot(range(0,len(loss_list)), loss_list)
    plt.title('Total Error')
    plt.xlabel('Iterations')
    plt.ylabel('% Error')
    plt.show()
    plt.plot(range(0,len(accuracy_dict["test"])), accuracy_dict["test"])
    plt.title('Total Accuracy Test')
    plt.xlabel('Iterations')
    plt.ylabel('% Accuracy')
    plt.show()
    plt.plot(range(0,len(accuracy_dict["train"])), accuracy_dict["train"])
    plt.title('Total Accuracy Training')
    plt.xlabel('Iterations')
    plt.ylabel('% Accuracy')
    plt.show()

    for d in range(10):
        plt.plot(range(0,len(digit_accurracies[d])), digit_accurracies[d])
        plt.title('Accuracy for Digit '+str(d)+ " Test Data")
        plt.xlabel('Iterations')
        plt.ylabel('% Error')
        plt.show()

    pickle_file = "nn_parameters" + str(accuracy_dict["test"][-1])[2:] + ".txt"
    filehandler = open(pickle_file,"wb")
    pickle.dump(theta, filehandler, protocol=2)
    filehandler.close()



if __name__ == '__main__':
    main()

###############################################################################
#***********************      Code Graveyard      *****************************
###############################################################################
"""



"""
