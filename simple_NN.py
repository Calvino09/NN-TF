import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import visualizations as vis
import argparse

import re
import logging

def simple_one_layer_NN():

    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.nn.softmax(tf.matmul(x, W) + b)

    # lazy evaluation
    #cross_entropy = -tf.reduce_sum(y*tf.log(pred))
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

    #xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred)
    #loss = tf.reduce_mean(xentropy, name='loss')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch  = int(len(train_data)/batch_size)

            for iteration in range(total_batch):
                x_batch = np.array(all_train[iteration * batch_size : min((iteration+1) * batch_size, len(train_data))])
                y_batch = np.array(all_label[iteration * batch_size : min((iteration+1) * batch_size, len(train_label))])

                _, c = sess.run([optimizer, cost], feed_dict = {x: x_batch, y: np.eye(10)[y_batch]})
                #print(c)
                avg_cost += c / total_batch

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        print(correct_prediction)

        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: np.array(validate_data), y: np.eye(10)[validate_label]}))


def five_NN_model_to_predict(learning_rate = 0.001, n_epochs = 100, batch_size = 100):
    # here we build a two layers NN model and test on validation set, you may improve it to a CV version
    # n_neurons_1 : number of neurons in the first layer
    # n_neurons_2  : number of neurons in the second layer
    # learning_rate : the learning rate of BGD
    # n_epochs : times of training the model
    # batch_size : since we adopted BGD, then we need to define the size of a size
    # initialize variables
    X = tf.placeholder(tf.float32, shape=(None, 28*28), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')

    
    # layer size
    
    L1 = 300
    L2 = 120
    L3 = 60
    L4 = 30
    L5 = 10
    
    # weights
    W1 = tf.Variable(tf.truncated_normal((28*28, L1),stddev = 0.01), name = 'layer_1')
    W2 = tf.Variable(tf.truncated_normal((L1, L2),stddev = 0.01), name = 'layer_2')
    W3 = tf.Variable(tf.truncated_normal((L2, L3),stddev = 0.01), name = 'layer_3')
    W4 = tf.Variable(tf.truncated_normal((L3, L4),stddev = 0.01), name = 'layer_4')
    W5 = tf.Variable(tf.truncated_normal((L4 , L5),stddev = 0.01), name = 'output_layer')

    # biases
    b1 = tf.Variable(tf.zeros([L1]), name='b_1')
    b2 = tf.Variable(tf.zeros([L2]), name='b_2')
    b3 = tf.Variable(tf.zeros([L3]), name='b_3')
    b4 = tf.Variable(tf.zeros([L4]), name='b_4')
    b5 = tf.Variable(tf.zeros([L5]), name='b_3')

    # the output of each layer
    Z1 = tf.nn.relu(tf.matmul(X,W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Z3 = tf.nn.relu(tf.matmul(Z2, W3) + b3)
    Z4 = tf.nn.relu(tf.matmul(Z3, W4) + b4)
    
    output = tf.matmul(Z4, W5) + b5

    # define loss function. Cross-entropy was adopted rather than MSE
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = output)
    loss = tf.reduce_mean(xentropy, name='loss')*100

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #training_op = optimizer.minimize(loss)
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # define accuracy
    correct = tf.nn.in_top_k(output, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(train_data) // batch_size):
                X_batch = np.array(all_train[iteration * batch_size : min((iteration+1) * batch_size, len(train_data))])
                y_batch = np.array(all_label[iteration * batch_size : min((iteration+1) * batch_size, len(train_label))])
                '''if (iteration + 1) * batch_size <= len(train_data):
                    X_batch = np.array(train_data[iteration * batch_size : iteration * batch_size + batch_size])
                    y_batch = np.array(train_label[iteration * batch_size : iteration * batch_size + batch_size])
                else:
                    X_batch = np.array(train_data[iteration * batch_size : ])
                    y_batch = np.array(train_label[iteration * batch_size : ])'''
                sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            # test error
            acc_test = accuracy.eval(feed_dict={X:np.array(validate_data),
                                               y:np.array(validate_label)})
            print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
        
        predict_output = sess.run(output,feed_dict={X:np.array(mnist_test)})
        return np.argmax(predict_output, axis= 1)


def cnn_model(learning_rate = 0.001, n_epochs = 50, batch_size = 100, drop_out = 0.75):
    # what args do we need ? - -|
    #NUM_ITERS=5000
    #DISPLAY_STEP=100
    #BATCH=100
    
    #
    # input layer               - X[batch, 28, 28]
    # 1 conv. layer             - W1[5, 5, 1, C1] + b1[C1]   pad = 2?
    #                             Y1[batch, 28, 28, C1]
    # 2 conv. layer             - W2[3, 3, C1, C2] + b2[C2]
    # 2.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 28x28-> 14x14
    #                             Y2[batch, 14,14,C2] 
    # 3 conv. layer             - W3[3, 3, C2, C3]  + b3[C3]
    # 3.1 max pooling filter 2x2, stride 2 - down sample the input (rescale input by 2) 14x14-> 7x7
    #                             Y3[batch, 7, 7, C3] 
    # 4 fully connecteed layer  - W4[7*7*C3, FC4]   + b4[FC4]
    #                             Y4[batch, FC4] 
    # 5 output layer            - W5[FC4, 10]   + b5[10]
    # One-hot encoded labels      Y5[batch, 10]
    
    # input 
    
    X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')
    
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32)
    
    # layer size, for cnn is conv depth (the number of detector)
    
    C1 = 4
    C2 = 8
    C3 = 16
    
    FC4 = 256  # fully connected layer
    
    # stride: 步幅, padding: 填充， "SAME"将detector结果填充为原向量维度
    stride = 1
    k = 2
    # conv 1
    W1 = tf.Variable(tf.truncated_normal((5,5, 1, C1),stddev = 0.01), name = 'conv_1')
    b1 = tf.Variable(tf.truncated_normal([C1], stddev = 0.01))
    
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding="SAME") + b1)   
    # conv 2 + maxpooling
    W2 = tf.Variable(tf.truncated_normal((3,3, C1, C2),stddev = 0.01), name = 'conv_2')
    b2 = tf.Variable(tf.truncated_normal([C2], stddev = 0.01))
    
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding="SAME") + b2)   
    Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
    
    # conv3 + maxpooling
    W3 = tf.Variable(tf.truncated_normal((3,3, C2, C3),stddev = 0.01), name = 'conv_3')
    b3 = tf.Variable(tf.truncated_normal([C3], stddev = 0.01))
    
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding="SAME") + b3)   
    Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
    
    # full connected 
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * C3])
    
    W4 = tf.Variable(tf.truncated_normal([7*7*C3, FC4], stddev=0.01, name = "full_connected"))
    b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.01))
    
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
    
    # calculate softmax mapping to 10 classification
    W5 = tf.Variable(tf.truncated_normal([FC4, 10], stddev=0.01))
    b5 = tf.Variable(tf.truncated_normal([10], stddev=0.01))
    
    Y5 = tf.nn.relu(tf.matmul(Y4, W5) + b5)
    
    Y = tf.nn.softmax(Y5)
    
    # loss function
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y, labels=y)
    loss = tf.reduce_mean(xentropy) * 100
    
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #training_op = optimizer.minimize(loss)
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # accuracy
    correct = tf.nn.in_top_k(Y, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    #correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # matplotlib visualization
    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
    allbiases  = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)


    # init 
    init = tf.global_variables_initializer()
    
    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()

    saver = tf.train.Saver()
    
    # run session
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for iteration in range(len(train_data) // batch_size):
                X_batch = np.array(all_train[iteration * batch_size : min((iteration+1) * batch_size, len(train_data))])
                y_batch = np.array(all_label[iteration * batch_size : min((iteration+1) * batch_size, len(train_label))])
                
                sess.run(training_op, feed_dict={X:np.reshape(X_batch, (len(X_batch), 28, 28, 1)), y:y_batch, pkeep: drop_out})
            
            acc_trn, loss_trn, w, b = sess.run([accuracy, loss, allweights, allbiases], feed_dict={X:np.reshape(X_batch, (len(X_batch), 28, 28, 1)), y:y_batch, pkeep: 1.0})
            
            acc_tst, loss_tst = sess.run([accuracy, loss], feed_dict={X:np.reshape(np.array(validate_data), (len(validate_data), 28, 28, 1)),
                                               y:np.array(validate_label), pkeep: 1.0})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(epoch,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)
            
            #acc_train = accuracy.eval(feed_dict={X:np.reshape(X_batch, (len(X_batch), 28, 28, 1)), y:y_batch, pkeep: 1.0})
            # test error
            #acc_test = accuracy.eval(feed_dict={X:np.reshape(np.array(validate_data), (len(validate_data), 28, 28, 1)),
            #                                   y:np.array(validate_label), pkeep: 1.0})
            #print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
        
        title = "MNIST_3.0 5 layers 3 conv. epoch={},batch_size={},learning_rate={},drop_out={}".format(n_epochs, batch_size, learning_rate, drop_out)
        vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,n_epochs)

        predict_output = sess.run(Y,feed_dict={X:np.reshape(np.array(mnist_test), (len(mnist_test), 28, 28, 1))})
        return np.argmax(predict_output, axis= 1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--nepochs', type=int, default = 50)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--drop-out', type=float, default=0.75)
    args = parser.parse_args()

    mnist_train = pd.read_csv("all/train.csv")
    mnist_test = pd.read_csv("all/test.csv")


    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    for train_index, validate_index in split.split(mnist_train, mnist_train['label']):
        train_data = mnist_train.loc[train_index]
        validate_data = mnist_train.loc[validate_index]

    train_label = train_data['label']
    train_data.drop('label', axis=1, inplace = True)
    train_label.reset_index(drop=True, inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    validate_label = validate_data['label']
    validate_data.drop('label', axis=1, inplace = True)
    validate_label.reset_index(drop=True, inplace=True)
    validate_data.reset_index(drop=True, inplace=True)

    all_train = np.concatenate([train_data, validate_data])
    all_label = np.concatenate([train_label, validate_label])

    #prediction = five_NN_model_to_predict()
    prediction = cnn_model(learning_rate = args.learning_rate, n_epochs = args.nepochs, batch_size = args.batch_size, drop_out = args.drop_out)

    df = pd.DataFrame({'ImageId': [i for i in range(1,len(prediction)+1)],
                  'Label': prediction})
    df.to_csv('./my_prediction.csv', index=None)
