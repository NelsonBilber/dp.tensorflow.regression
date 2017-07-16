# code from https://medium.com/@saxenarohan97/intro-to-tensorflow-solving-a-simple-regression-problem-e87b42fd4845

import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt


# Returns predictions and error
def calc(x, y, b, w):
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions)) #Computes the mean of elements across dimensions of a tensor.
    return [ predictions, error ]




def main():
    ################################################
    #  Train Data
    ################################################
    
    #get data
    total_features, total_prices = load_boston(True);

    #keep 300 samples for training
    train_features = scale(total_features[:300])
    train_prices = total_prices[:300]

    ################################################
    #  Validation Data
    ################################################

    #keep 100 samples for validation
    valid_features = scale (total_features[300:400])
    valid_prices = total_prices[300:400]

    ################################################
    #  Test Data
    ################################################

    #keep remaining samples as test set
    test_features = scale(total_features[400:])
    test_prices = total_prices[400:]


    ################################################
    # Tensors
    # A tensor is basically an n-dimensional array
    ################################################

    #generates a regularised set of numbers from the normal probability distribution
    w = tf.Variable(tf.truncated_normal([13,1], mean=0.0, stddev=1.0, dtype=tf.float64))
    #Random set of initial weights is considered a good practise in machine learning
    b = tf.Variable(tf.zeros(1, dtype = tf.float64))


    ################################################
    #  Cost Function
    ################################################
    y, cost = calc(train_features, train_prices, b, w)

    #Gradient descent only needs a single parameter, the learning rate, which is a scaling factor for the size of the parameter updates. The bigger the learning rate, the more the parameter values change after each step. If the learning rate is too big, the parameters might overshoot their correct values and the model might not converge. If it is too small, the model learns very slowly and takes too long to arrive at good parameter values.

    # Feel free to tweak these 2 values:
    learning_rate = 0.0001
    # But how can we change our parameter values to minimize the loss? This is where TensorFlow works its magic. Via a technique called auto-differentiation it can calculate the gradient of the loss with respect to the parameter values. This means that it knows each parameter’s influence on the overall loss and whether decreasing or increasing it by a small amount would reduce the loss. It then adjusts all parameter values accordingly, which should improve the model’s accuracy. After this parameter adjustment step the process restarts and the next group of images are fed to the model.
    # groups of values to feed the system
    epochs = 30000
    points = [[], []] # You'll see later why I need this


    ################################################
    #  Apply a gradient Descend
    ################################################

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(init)
        for i in list(range(epochs)):
            sess.run(optimizer)
        
            if i % 10 == 0.:
                points[0].append(i+1)
                points[1].append(sess.run(cost))
        
            if i % 100 == 0:
                print(sess.run(cost))

        #plt.plot(points[0], points[1], 'r--')
        #plt.axis([0, epochs, 50, 600])
        #plt.show()
    
        valid_cost = calc(valid_features, valid_prices, b, w)[1]
    
        print('Validation error =', sess.run(valid_cost), '\n')
    
        test_cost = calc(test_features, test_prices, b, w)[1]
        print('Test error =', sess.run(test_cost), '\n')

    print ("End simple regression")

if __name__ == "__main__":
    main()
