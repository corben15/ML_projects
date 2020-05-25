import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time

# If you are using your local machine uncomment the following code:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def loadData(file_name):
    data_file = open(file_name, "rb")
    train_x, train_y, test_x, test_y = pickle.load(data_file, encoding="bytes")
    data_file.close()
    return train_x, train_y, test_x, test_y

def main():
    # Reset Graph
    tf.reset_default_graph()
    # Load Data
    DATA_FILE = "cifar_10_tf_train_test.pkl"
    train_x,train_y, test_x, test_y = loadData(DATA_FILE)
    test_y_np = np.array(test_y)

    print("Train X size:\t", train_x.shape)
    print("Train Y size:\t", len(train_y))
    print("Test X size:\t", test_x.shape)
    print("Test Y size:\t", len(test_y))

    """
    imgplot = plt.imshow(data_list[0][0])
    plt.colorbar()
    plt.show()
    """

    # Hyper Parameters
    batch_size = 100
    num_epochs = 3000
    learning_rate = .005
    # Convolution Layer1
    filter_size1 = 5
    num_filters1 = 32
    # Convolution Layer2
    filter_size2 = 5
    num_filters2 = 32
    # Convolution Layer3
    filter_size3 = 3
    num_filters3 = 64

    # Dimensions of Data
    img_size = 32
    img_depth = 3           # number of channels in the image (red,blue,green)
    img_size_flat = 32*32*img_depth
    img_shape = (img_size,img_size,img_depth)
    num_classes = 10

    # Initializers
    xavier_init = tf.initializers.glorot_normal()
    #xavier_init = tf.contrib.layers.xavier_initializer()
    zero_init = tf.zeros_initializer()

    # Input Variables
    input_img = tf.placeholder(dtype=tf.uint8, shape=[None, img_size, img_size, img_depth], name="input_img")
    y = tf.placeholder(dtype=tf.int64, shape=[None], name="labels")

    # Normalization
    x = tf.image.convert_image_dtype(input_img,dtype="float32")
    x = tf.math.divide(x,255)
    mean = tf.math.reduce_mean(x,0)
    x = tf.math.subtract(x,mean)

    y_true = tf.one_hot(y, 10,dtype="float32")

    # Filters,Weights, and Biases
    F1_shape = [filter_size1,filter_size1,img_depth,num_filters1]
    F1 = tf.get_variable(shape=F1_shape, dtype='float32', initializer=xavier_init, name="filter1")
    F1_bias = tf.get_variable(shape=[num_filters1],dtype='float32', initializer=zero_init, name="filter_bias1")

    F2_shape = [filter_size2,filter_size2,num_filters1,num_filters2]
    F2 = tf.get_variable(shape=F2_shape, dtype='float32', initializer=xavier_init, name="filter2")
    F2_bias = tf.get_variable(shape=[num_filters2],dtype='float32', initializer=zero_init, name="filter_bias2")

    F3_shape = [filter_size3,filter_size3,num_filters2,num_filters3]
    F3 = tf.get_variable(shape=F3_shape, dtype='float32', initializer=xavier_init, name="filter3")
    F3_bias = tf.get_variable(shape=[num_filters3],dtype='float32', initializer=zero_init, name="filter_bias3")

    weights_fc = tf.get_variable(shape=[576,num_classes] , dtype="float32", initializer=xavier_init, name="weightsfc")
    bias_fc = tf.get_variable(shape=[10] , dtype="float32", initializer=zero_init, name="biasfc")

    # Forward Propagation
    conv_layer1 = tf.nn.leaky_relu(tf.nn.conv2d(x, filters=F1, strides=[1,1,1,1],padding="VALID") + F1_bias)
    pool1 = tf.nn.pool(conv_layer1, window_shape=[2,2],pooling_type="MAX", strides=[2,2], padding="VALID")

    conv_layer2 = tf.nn.leaky_relu(tf.nn.conv2d(pool1, filters=F2, strides=[1,1,1,1],padding="VALID") + F2_bias)
    pool2 = tf.nn.pool(conv_layer2, window_shape=[2,2],pooling_type="MAX", strides=[2,2], padding="VALID")

    conv_layer3 = tf.nn.leaky_relu(tf.nn.conv2d(pool2, filters=F3, strides=[1,1,1,1],padding="VALID") + F3_bias)

    # Vectorize Final Convolution
    conv_vector = tf.layers.flatten(conv_layer3)

    print(conv_layer1.get_shape())
    print(conv_layer2.get_shape())
    print(conv_layer3.get_shape())
    print(conv_vector.get_shape())

    # Fully Connected Layer
    logits = tf.matmul(conv_vector, weights_fc)+bias_fc
    softmax_op = tf.nn.softmax(logits)
    predict_lbl = tf.argmax(softmax_op, axis=1, name='predict_lbl')

    # Cost Function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                        logits=logits, name=None)
    correct_prediction = tf.equal(predict_lbl, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes", input_img)
    tf.add_to_collection("validation_nodes", predict_lbl)
    # start training
    saver = tf.train.Saver()

    # Plot Variables
    cost_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    test_accuracy_cls = {}

    start_time = time.time()
    # Initialize the Graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print("\n\n\n")
        sess.run(init)

        index = 0
        trained_set = set()
        for e in range(num_epochs):
            time.sleep(.1)

            indlimit = train_x.shape[0]-batch_size
            index = random.randint(0,indlimit)

            for i in range(index,index+batch_size):trained_set.add(int(i))

            x_batch = train_x[index: index+batch_size]
            y_batch = train_y[index: index+batch_size]

            permutation = np.random.permutation(len(y_batch))
            x_batch = x_batch[permutation,:]
            y_batch = np.asarray(y_batch)[permutation]

            sess.run(optimizer, feed_dict={input_img:x_batch, y:y_batch})
            # Store Values for plots
            cost_list.append(sess.run(cost, feed_dict={input_img:x_batch, y:y_batch}))
            train_accuracy_list.append(sess.run(accuracy, feed_dict={input_img:x_batch, y:y_batch}))





            if(e%100==0):
                print("Iteration:\t", e)
                print("Index Start:\t",index)

                print("Len Trained Set:", len(trained_set))
                predict_test = sess.run(predict_lbl, feed_dict={input_img:test_x})
                test_accuracy = np.sum(predict_test==test_y_np)/5000

                test_accuracy_list.append(test_accuracy)



                print("Test Accuracy:", test_accuracy)
                print()



        # this saver.save() should be within the same tf.Session() after the training is
        conv1_filters = sess.run(F1, feed_dict={input_img:x_batch})
        conv1_filter_images = ((conv1_filters + 0.1) * (1/0.3) * 255).astype('uint8')
        save_path = saver.save(sess, "my_model")


        for pred in range(len(predict_test)):
            if test_y_np[pred] not in test_accuracy_cls:
                test_accuracy_cls[test_y_np[pred]]={}
                test_accuracy_cls[test_y_np[pred]]["correct"] = 0
                test_accuracy_cls[test_y_np[pred]]["total"] = 0

            test_accuracy_cls[test_y_np[pred]]["total"] += 1
            if(test_y_np[pred]==predict_test[pred]):
                test_accuracy_cls[test_y_np[pred]]["total"] += 1

    print(test_accuracy_cls)

    end_time = time.time()
    print("Time Ellapsed:", end_time-start_time)

    plt.plot(range(0,len(train_accuracy_list)), train_accuracy_list)
    plt.title('Total Training Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()

    plt.plot(range(0,len(cost_list)), cost_list)
    plt.title('Total Error/Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    plt.plot(range(0,len(test_accuracy_list)), test_accuracy_list)
    plt.title('Total Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()
    # TODO: plot filters
    print(conv1_filter_images.shape)
    conv1_filter_images = conv1_filter_images.T
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.title("Filter "+str(i+1),fontsize=6)
        plt.axis("off")
        plt.imshow(conv1_filter_images[i].T)
    plt.show()



if __name__ == '__main__':
    main()

######################################################################
################# CODE GRAVEYARD ##########################
######################################################################
