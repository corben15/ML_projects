'''
Nicholas Corbett
Deep Learning 6850 Programming Assignment 1

Implement in tensorflow he stochastic gradient desecent method with L2
regularization.

Initialize theta to a small value and iteratively update theta with appropriate
learning rate and regularization parameter lambda until convergence.

'''
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import pickle


'''
SOFTMAX
Description: The softmax is calculated by taking the dot product of a given
    weight vector times all the data points. Then e is raised to this
    scalar value. Then this is divided by the sum of all the weight vectors
     dotted with the data points a and this scalar value is again the exponent
     for e.

inputs: X - list of np.array's with shape (745x1) representing digits
        theta_list - list of np.array's with shape (745x1) representing the
            weight vectors for the individual classifiers
        k - an int from 0-4 representing the indice of the classifier to
            calculate the softmax for.

output: float
'''
def softmax(X,theta_list, k):
    #print("SOFTMAX FUNCTION")
    numerator_exponent = np.dot(np.transpose(X),theta_list[k])
    numerator = pow(math.e, numerator_exponent)
    denominator = 0
    for i in range(len(theta_list)):
        denominator_exponent = np.dot(np.transpose(X),theta_list[i])
        denominator += pow(math.e, denominator_exponent)

    return round(float(numerator),5)/round(float(denominator),5)

'''
GRADIENT
Description:

inputs: data - list of np.array's with shape (745x1) representing digits
        labels - which represent the actual digit values of the images in data
            Not necessary but used as a sanity check for the t matrix
        theta_list - list of np.array's with shape (745x1) representing the
            weight vectors for the individual classifiers
        k - int from 0-4 to indicate which gradient to calculate
        t - np.array with size of mxk where m is the number of digits in the
            stochastic batch and k is the number of classes.
output: gradient vector np.array with shape 785x1
'''
def calc_gradient(data, labels, theta_list, k,t):
    for m in range(len(data)):
        '''
        t1=0
        if(labels[m] == k+1):
            t1=1
        t2 = t[m][k]
        assert(t1==t2)
        '''
        if(m==0):
            #print("here1")
            gradient = (t[m][k] - softmax(data[m], theta_list, k))*data[m]
        else:
            #print("here2")
            gradient += (t[m][k] - softmax(data[m], theta_list, k))*data[m]
    gradient = np.negative(gradient)
    return gradient

'''
REGULARIZATION
Description:

inputs: lambda_ - float representing ruglarization parameters
        theta - np.array with shape 785x1 its the singular weight vector for
            a given classifier
output - regularization vector
'''
def calc_regularization(lambda_, theta):
    return 2*lambda_*theta

'''
update_weights
Description: The weights are updated by subtracting the gradient, which
    is the direction of steepest descent plus the regularization, times the
    leanrning rate which is the step size.

inputs: theta_list - list of np.array's with shape (745x1) representing the
            weight vectors for the individual classifiers
        learning_rate - float learning rate parameter
        gradient_list - list of np.array's with shape (785x1)
        lambda_ - float regularization parameter
'''
def update_weights(theta_list,learning_rate, gradient_list, lambda_):
    for i in range(len(theta_list)):
        theta_list[i] = theta_list[i] - learning_rate*(gradient_list[i]+calc_regularization(lambda_,theta_list[i]))
    return theta_list

def loss_function(X, theta_list,t):
    loss = 0
    for m in range(len(X)):
        for k in range(len(theta_list)):
            t1 = t[m][k]
            '''t2=0
            if(y_list[m] == k+1):
                t2=1
            assert(t1==t2)
            '''
            loss += t1*math.log(softmax(X[m],theta_list,k),math.e)
    return -loss

'''
I calculated the accuracy by counting the number of correct classifications
and dividing it by the total number of digits that belong to that class.
I didnt calculate the error i calculated the accuracy and then plotted
1-accuracy. Its the same thing.
'''
def calc_total_accuracyacy(theta_list, test_data, test_labels):
    number_correct = 0
    for i in range(len(test_data)):
        probabilities = []
        max_k = 0
        max_prob = 0
        for k in range(len(theta_list)):
            probabilities.append(softmax(test_data[i],theta_list,k))
            if(probabilities[k]>max_prob):
                max_prob = probabilities[k]
                max_k = k
        if(max_k+1 == test_labels[i]):
            number_correct += 1
    return float(number_correct)/len(test_data)

def calc_individual_accuracy(theta_list, test_data, test_labels):
    class_correct = np.array([0,0,0,0,0])
    classified = np.array([0,0,0,0,0])
    class_count = np.array([0,0,0,0,0])

    for i in range(len(test_data)):
        probabilities = []
        max_k = 0
        max_prob = 0
        for k in range(len(theta_list)):
            probabilities.append(softmax(test_data[i],theta_list,k))
            if(probabilities[k]>max_prob):
                max_prob = probabilities[k]
                max_k = k

        class_count[test_labels[i]-1] += 1
        classified[max_k] += 1

        if(max_k+1 == test_labels[i]):
            class_correct[max_k] += 1


    print("Correct:", class_correct)
    print("Classified:",classified)
    print("Total: ", class_count)
    return list(np.divide(class_correct,class_count))


def main():

    # Code to access files
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path_train_data = dir_path + "/data_prog2/train_data"

    start = time.time()
    # Develope Training sets
    train_labels_file = open(dir_path + "/data_prog2/labels/train_label.txt", 'r')
    train_labels = train_labels_file.read()
    train_labels = train_labels.split()
    train_labels = [int(digit) for digit in train_labels]
    train_data = []
    for i in range(25112):
        file_name = dir_path_train_data + "/" + str(i+1).rjust(5,"0") + ".jpg"
        image = mpimg.imread(file_name)
        # Change the image into a 784x1 vector and normalize
        x = image.flatten()
        x = np.divide(x,255)
        x = np.reshape(x, (784,1))
        # Add value of 1 for w0
        x = np.append(x,np.array([[1]]), axis=0)
        train_data.append(x)
    end = time.time()
    print("Train data processing time:",end-start)

    start = time.time()
    # Develope Test Sets
    dir_path_test_data = dir_path + "/data_prog2/test_data"
    test_data = []
    test_labels_file = open(dir_path + "/data_prog2/labels/test_label.txt", 'r')
    test_labels = test_labels_file.read()
    test_labels = test_labels.split()
    test_labels = [int(digit) for digit in test_labels]
    for i in range(4982):
        file_name = dir_path_test_data + "/" + str(i+1).rjust(4,"0") + ".jpg"
        image = mpimg.imread(file_name)

        x = image.flatten()
        x = np.divide(x,255)
        x = np.reshape(x, (784,1))
        # Add value of 1 for w0
        x = np.append(x,np.array([[1]]), axis=0)
        test_data.append(x)
    end = time.time()
    print("Test data processing time:",end-start)

    # Initialize hyper parameters
    lambda_ = .05           # Regularization parameter
    learning_rate = .01    # Learning Rate Parameter

    M = 25112               # M is the number of data points in training set
    K = 5                   # K is the number of classes/classifiers
    sbs = 10               # Stochastic Batch Size
    stoc_data = []

    # Initialize weights random numbers near zero
    theta_list = []
    for k in range(K):
        w = np.random.rand(784,1)
        w = w/10
        # Could also use normal to Initialize some of the weights to be negative
        # np.random.normal(0,.01,(784,1))
        w = np.append(w,np.array([[1]]), axis=0)
        theta_list.append(w)
    assert(len(theta_list)==K)

    individual_accuracys = []
    total_accuracys = []

    start = time.time()
    i = 0
    convergence_count = 0
    while(1):
        i = i%25112
        stoc_data.append(train_data[i])

        if(i%sbs == sbs-1):

            # Create t matrix
            y_list = train_labels[i-sbs+1:i+1]

            t = np.zeros((len(y_list),5))
            for k in range(sbs):
                t[k][y_list[k]-1] = 1


            assert(sbs == len(stoc_data))

            # Calculate the gradients
            gradient_list = []
            for k in range(len(theta_list)):
                gradient_list.append(calc_gradient(stoc_data,y_list,theta_list, k,t))

            # Reset the stochastic Dataset to empty
            stoc_data = []

            # Update theta_list (update the weight values)
            theta_list = update_weights(theta_list, learning_rate, gradient_list, lambda_)


            print("Iteration:",int(i/sbs))
            accuracy = calc_individual_accuracy(theta_list, train_data,train_labels)
            print("Individual Accuracy:",accuracy)
            total_accuracy = calc_total_accuracyacy(theta_list, train_data,train_labels)
            print("Accuracy:", total_accuracy)
            end = time.time()
            #print("Loss:",loss_function(stoc_data,theta_list,t) )
            print("Convergence Count:",convergence_count)
            print("Time:",end - start)
            print("\n\n\n")

            total_accuracys.append(total_accuracy)
            for a in range(len(accuracy)):
                individual_accuracys.append(accuracy[a])

            '''Originally trained until the accuracy was over 92% this value takes a
             while to get to but sometimes it is hit luckily. With a convergence test
             I can be more confident in the value '''
            #if(total_accuracy > .94):break
            # If the last 4 Accuracy values are within .0001 then stop training
            if( (len(total_accuracys) > 4) and abs(float(total_accuracys[-2] - total_accuracys[-1]) < .0001) ):
                convergence_count+=1
            else:
                convergence_count = 0
            if(convergence_count >= 4):
                break


        i+=1

    # Once the error values converge calculate the Testset Error
    print("Test Accuracy:",calc_total_accuracyacy(theta_list, test_data,test_labels))
    print("Test Individual Accuracy:", calc_individual_accuracy(theta_list, test_data,test_labels))

    # Plot Images of the weights
    for k in range(len(theta_list)):
        image = list(theta_list[k].flatten())
        image = image[0:len(image)-1]
        image = np.reshape(image,(28,28))
        imgplot = plt.imshow(image)
        title = "Image for Digit:" + str(k+1)
        plt.title(title)
        plt.colorbar()
        plt.show()

    # Dump Weights in files
    filehandler = open("multiclass_parameters.txt","wb")
    theta_list2 = []
    for theta in theta_list:
        theta_list2.append(list(theta.flatten()))
    pickle.dump(theta_list2, filehandler)
    filehandler.close()

    # Write weights to a file
    with open("multiclass_parameters1.txt","w") as file:
        for theta in theta_list:
            for weight in list(theta.flatten()):
                file.write(str(weight)+"\n")


    # Plot Total Errors
    iterations = range(len(total_accuracys))
    plt.plot(iterations,[1-i for i in total_accuracys])
    plt.title('Total Error')
    plt.xlabel('Iterations')
    plt.ylabel('% Error')
    plt.show()

    # Plot Individual errors
    for n in range(1,6):
        plt.plot(iterations,[1-i for i in individual_accuracys[n-1::5]])
        plt.title("Classifier "+str(n+1)+" Convergence")
        plt.xlabel('Iterations')
        plt.ylabel('% Error')
        plt.show()


if __name__ == '__main__':
    main()


###############################################################################
#***********************      Code Graveyard      *****************************
###############################################################################
