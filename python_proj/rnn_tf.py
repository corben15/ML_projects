import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time
import os


# If you are using your local machine uncomment the following code:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def main():
    # Reset Graph
    tf.reset_default_graph()

    print()
    print()
    cwd = os.path.abspath(os.getcwd())
    data_file = open(cwd + '/youtube_train_data.pkl', 'rb')
    total_data, total_labels = pickle.load(data_file)
    data_file.close()

    print()
    print("Pre_Shuffle")
    print("Data Type: ", type(total_data))
    print("Data Shape: ", total_data.shape)
    print("Labels Shape:", total_labels.shape,"\n")

    match_data_point = np.copy(total_data[0])
    match_label_point = np.copy(total_labels[0])

    '''
    # Shuffle Data
    permutation = np.random.permutation(len(total_data))
    label_0_loc = int(np.where(permutation==0)[0])
    total_data = total_data[permutation,:]
    total_labels = total_labels[permutation,:]


    print("Post Shuffle")
    print("Data Type: ", type(total_data))
    print("Data Shape: ", total_data.shape)
    print("Labels Shape:", total_labels.shape)
    print()
    print()

    # Make sure labels still match up
    assert(np.array_equal(total_data[label_0_loc],match_data_point))
    assert(np.array_equal(total_labels[label_0_loc],match_label_point))
    '''

    # Split Data into Train and Test Sets
    train_data = total_data[:6000]
    train_labels = total_labels[:6000]
    test_data = total_data[6000:]
    test_labels = total_labels[6000:]


    # Hyper Parameters
    sequence_size = 10
    batch_size = 10
    learning_rate=.03
    pixel_dist_thresh = 10
    # Convolution Layer1 and Layer 2
    filter_size1 = 5
    num_filters1 = 32
    # Convolution Layer3
    filter_size3 = 3
    num_filters3 = 64

    # Dimensions of Data
    img_size = 64
    img_depth = 3           # number of channels in the image (red,blue,green)
    img_size_flat = 64*64*img_depth
    img_shape = (img_size,img_size,img_depth)


    # Initializers
    xavier_init = tf.initializers.glorot_normal()
    zero_init = tf.zeros_initializer()
    # Initializers
    xavier_init = tf.initializers.glorot_normal()
    zero_init = tf.zeros_initializer()


    #  Variables
    input_frames = tf.placeholder(dtype=tf.uint8, shape=[None, 10, 64, 64, 3], name='input_frames')
    y = tf.placeholder(dtype=tf.float32, shape=[None,10,7,2])

    # Normalization
    input_frames = input_frames/255
    input_frames = input_frames - tf.reduce_mean(input_frames, axis=(2, 3, 4), keepdims=True)
    input_frames = input_frames / tf.math.reduce_std(input_frames, axis=(2, 3, 4), keepdims=True)


    cnn_input = tf.reshape(input_frames, [tf.shape(input_frames)[0]*tf.shape(input_frames)[1],64,64,3])
    # Convotulion Filters,Weights, and Biases
    F1 = tf.get_variable(shape=[filter_size1,filter_size1,img_depth,num_filters1], dtype='float32', initializer=xavier_init, name="filter1")
    F1_bias = tf.get_variable(shape=[32],dtype='float32', initializer=zero_init, name="filter_bias1")
    F2 = tf.get_variable(shape=[filter_size1,filter_size1,num_filters1,num_filters1], dtype='float32', initializer=xavier_init, name="filter2")
    F2_bias = tf.get_variable(shape=[32],dtype='float32', initializer=zero_init, name="filter_bias2")
    F3 = tf.get_variable(shape=[filter_size3,filter_size3,num_filters1,num_filters3], dtype='float32', initializer=xavier_init, name="filter3")
    F3_bias = tf.get_variable(shape=[64],dtype='float32', initializer=zero_init, name="filter_bias3")

    # Forward Propagation
    conv_layer1 = tf.nn.leaky_relu(tf.nn.conv2d(cnn_input, filters=F1, strides=[1,1,1,1],padding="VALID") + F1_bias)
    pool1 = tf.nn.pool(conv_layer1, window_shape=[2,2],pooling_type="MAX", strides=[2,2], padding="VALID")
    conv_layer2 = tf.nn.leaky_relu(tf.nn.conv2d(pool1, filters=F2, strides=[1,1,1,1],padding="VALID") + F2_bias)
    pool2 = tf.nn.pool(conv_layer2, window_shape=[2,2],pooling_type="MAX", strides=[2,2], padding="VALID")
    conv_layer3 = tf.nn.leaky_relu(tf.nn.conv2d(pool2, filters=F3, strides=[1,1,1,1],padding="VALID") + F3_bias)

    # Vectorize Final Convolution
    conv_vector = tf.layers.flatten(conv_layer3)
    # Reshape for RNN
    rnn_input = tf.reshape(conv_vector, [tf.shape(input_frames)[0],10,7744])

    # Instantiate a LSTM cell
    num_units = 10
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    # Define your RNN network, the length of the sequence will be automatically retrieved
    h_val, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_input, dtype=tf.float32)

    weights_fc = tf.get_variable(shape=[num_units,14], dtype='float32', initializer=xavier_init, name="weights_fc")
    bias_fc = tf.get_variable(shape=[1,14],dtype='float32', initializer=zero_init, name="bias_fc")

    # Collection of the final output
    final_output = tf.zeros(shape=[tf.shape(input_frames)[0],0,7,2])
    for i in np.arange(sequence_size):
        temp = tf.reshape(h_val[:,i,:], [tf.shape(input_frames)[0],num_units])
        output = tf.matmul(temp, weights_fc) + bias_fc
        output = tf.reshape(output, [-1,1,7,2])
        final_output = tf.concat([final_output,output],axis=1)

    joint_pos = tf.identity(final_output, name='joint_pos')

    # Cost Function
    total_loss = tf.losses.mean_squared_error(labels=y, predictions=joint_pos)

    # Accuracy
    distances = tf.square(y - joint_pos)
    distances = tf.reduce_sum(distances, axis=3, keepdims=True)
    distances = tf.sqrt(distances)
    sum_distances = tf.reduce_sum(distances, axis=0)
    sum_distances = tf.reduce_sum(sum_distances,axis=0)/tf.cast(tf.shape(input_frames)[0], dtype=tf.float32)/10

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes",input_frames)
    tf.add_to_collection("validation_nodes",joint_pos)

    # Plot Variables
    loss_list_train = []
    loss_list_test = []
    total_accuracy_10_train_list = []
    total_accuracy_20_train_list = []
    total_accuracy_10_test_list = []
    total_accuracy_20_test_list = []
    individual_accuracy_10_test_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    individual_accuracy_20_test_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}

    train_distances = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    test_distances = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}

    # Start training
    saver = tf.train.Saver()
    # Initialize the Graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("\n\n")

        trained_set = set()
        for e in range(500):

            indlimit = train_data.shape[0]-batch_size
            index = random.randint(0,indlimit)

            for i in range(index,index+batch_size):trained_set.add(int(i))

            # Generate Random Batch
            x_batch = train_data[index: index+batch_size]
            y_batch = train_labels[index: index+batch_size]
            permutation = np.random.permutation(batch_size)
            x_batch = x_batch[permutation]
            y_batch = np.asarray(y_batch)[permutation]

            # Figure out layer shapes:
            c1 = sess.run(conv_layer1, feed_dict={input_frames:x_batch})
            p1 = sess.run(pool1, feed_dict={input_frames:x_batch})
            c2 = sess.run(conv_layer2, feed_dict={input_frames:x_batch})
            p2 = sess.run(pool2, feed_dict={input_frames:x_batch})
            c3 = sess.run(conv_layer3, feed_dict={input_frames:x_batch})
            print(x_batch.shape)
            print(c1.shape)
            print(p1.shape)
            print(c2.shape)
            print(p2.shape)
            print(c3.shape)
            return

            # Calc Predictions, Loss, and Back Propagate
            predictions = sess.run(joint_pos, feed_dict={input_frames:x_batch})
            tf_distances = sess.run(sum_distances, feed_dict={input_frames:x_batch, y:y_batch})
            loss = sess.run(total_loss, feed_dict={input_frames:x_batch, y:y_batch})
            sess.run(optimizer, feed_dict={input_frames:x_batch, y:y_batch})



            # Calc training accuracies
            distances = np.linalg.norm(predictions.reshape((-1,2)) - y_batch.reshape((-1,2)), ord=2, axis = 1)

            for i in range(7):
                train_distances[i].append(tf_distances[i][0])



            accuracy_10 = np.copy(distances)
            accuracy_10[accuracy_10<pixel_dist_thresh]=1
            accuracy_10[accuracy_10>pixel_dist_thresh]=0

            accuracy_20 = np.copy(distances)
            accuracy_20[accuracy_20<pixel_dist_thresh+10]=1
            accuracy_20[accuracy_20>pixel_dist_thresh+10]=0

            total_accuracy_10 = np.sum(accuracy_10)/accuracy_10.size
            total_accuracy_20 = np.sum(accuracy_20)/accuracy_20.size

            # Store train accuracy and train losses
            loss_list_train.append(loss)
            total_accuracy_10_train_list.append(total_accuracy_10)
            total_accuracy_20_train_list.append(total_accuracy_20)


            time.sleep(.1)
            if(e%20 == 0):
                print("Iteration:",e)
                #print("Total Accuracy 10:",total_accuracy_10)
                print("Error:", np.mean(np.linalg.norm(predictions.reshape((-1,2)) - y_batch.reshape((-1,2)), axis = 1)) )
                print("")

            # After 100 iterations test
            if(e%20 == 0):

                print("Loss:", loss)
                print("Total Accuracy 10:",total_accuracy_10)
                print("Total Accuracy 20:",total_accuracy_20)

                # Generate mini test batch
                test_batch_size = 500
                permutation = np.random.permutation(1000)
                shuffled_test_data = test_data[permutation]
                shuffled_test_labels = test_labels[permutation]
                test_batch_data = shuffled_test_data[:test_batch_size]
                test_batch_labels = shuffled_test_labels[:test_batch_size]


                predictions = sess.run(joint_pos, feed_dict={input_frames:test_batch_data})
                tf_distances = sess.run(sum_distances, feed_dict={input_frames:test_batch_data, y:test_batch_labels})
                loss = sess.run(total_loss, feed_dict={input_frames:test_batch_data, y:test_batch_labels})

                # Calc testaccuracies
                distances = np.linalg.norm(predictions.reshape((-1,2)) - test_batch_labels.reshape((-1,2)), ord=2, axis = 1)

                for i in range(7):
                    test_distances[i].append(tf_distances[i][0])


                accuracy_10 = np.copy(distances)
                accuracy_10[accuracy_10<pixel_dist_thresh]=1
                accuracy_10[accuracy_10>pixel_dist_thresh]=0

                accuracy_20 = np.copy(distances)
                accuracy_20[accuracy_20<pixel_dist_thresh+10]=1
                accuracy_20[accuracy_20>pixel_dist_thresh+10]=0

                total_accuracy_10 = np.sum(accuracy_10)/accuracy_10.size
                total_accuracy_20 = np.sum(accuracy_20)/accuracy_20.size

                # Store test accuracy and train losses
                loss_list_test.append(loss)
                total_accuracy_10_test_list.append(total_accuracy_10)
                total_accuracy_20_test_list.append(total_accuracy_20)

                individual_accuracy_10 = np.sum(accuracy_10.reshape(-1,7,1),axis=0)/test_batch_size
                individual_accuracy_20 = np.sum(accuracy_20.reshape(-1,7,1),axis=0)/test_batch_size

                # Store variables to plot
                loss_list_test.append(loss)
                total_accuracy_10_test_list.append(total_accuracy_10)
                total_accuracy_20_test_list.append(total_accuracy_20)


                for i in range(7):
                    individual_accuracy_10_test_dict[i].append(individual_accuracy_10[i][0])
                    individual_accuracy_20_test_dict[i].append(individual_accuracy_20[i][0])


        save_path = saver.save(sess, "my_model")


    # Plot image with Joint Marks
    imgplot = plt.imshow(x_batch[0][0])
    plt.scatter(y=predictions[0][0][:,1],x=predictions[0][0][:,0], marker='x', color='red')
    plt.show()

    for i in range(7):
        plt.plot(range(0,len(train_distances[i])), train_distances[i] )
    plt.legend()
    plt.title('Training Pixel Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Average Pixel Distance')
    plt.show()
    for i in range(7):
        plt.plot(range(0,len(test_distances[i])), test_distances[i] )
    plt.legend()
    plt.title('Test Pixel Distance')
    plt.xlabel('Iterations')
    plt.ylabel('Average Pixel Distance')
    plt.show()


    # Plot Training and Test Loss
    plt.plot(range(0,len(loss_list_train)), loss_list_train)
    plt.title('Total Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(range(0,len(loss_list_test)), loss_list_test)
    plt.title('Total Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    # Plot total training and test accuracy for 10 and 20 pixels
    plt.plot(range(0,len(total_accuracy_10_train_list)), total_accuracy_10_train_list)
    plt.title('Total Train Accuracy within 10 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()
    plt.plot(range(0,len(total_accuracy_20_train_list)), total_accuracy_20_train_list)
    plt.title('Total Train Accuracy within 20 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()
    plt.plot(range(0,len(total_accuracy_10_test_list)), total_accuracy_10_test_list)
    plt.title('Total Test Accuracy within 10 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()
    plt.plot(range(0,len(total_accuracy_20_test_list)), total_accuracy_20_test_list)
    plt.title('Total Test Accuracy within 20 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()


    # Plot individual Accuracy for 10 pixels and 20 pixels
    for i in range(7):
        plt.plot(range(0,len(individual_accuracy_10_test_dict[i])), individual_accuracy_10_test_dict[i] )
    plt.legend()
    plt.title('Individual Accuracy within 10 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()
    for i in range(7):
        plt.plot(range(0,len(individual_accuracy_20_test_dict[i])), individual_accuracy_20_test_dict[i] )
    plt.legend()
    plt.title('Individual Accuracy within 20 Pixels')
    plt.xlabel('Iterations')
    plt.ylabel('%Accuracy')
    plt.show()






if __name__ == '__main__':
    main()
