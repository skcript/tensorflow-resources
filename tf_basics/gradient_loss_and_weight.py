# Basics of Neural Networking
import numpy as np

no_inputs = 5
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])
weight = 1
bias = 0.0
learning_rate = 0.01

# The basic formula is Y = ( w * X ) + B
# Y is the outputs
# w is the weight
# X is the inputs
# B is the bias

for i in range(10):
    print 'Step', i, ':'
    output = x_data * weight + bias
    # [ 0.3  0.6  0.9  1.2  1.5]

    difference = y_data - output

    gradient = 0
    for i in range(no_inputs):
        gradient += difference[i] * x_data[i]

    gradient /= no_inputs

    loss = (difference * difference) / no_inputs
    print 'Loss:', np.sum(loss)

    weight = weight + (learning_rate * gradient)

    print 'Weight: ', weight, '\n'
