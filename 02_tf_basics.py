# Basics of Tensorflow

import tensorflow as tf

x_input = tf.constant([1, 2, 3, 4])
# print x_input
# Tensor("Const:0", shape=(4,), dtype=int32)

sess = tf.Session()
print sess.run(x_input)
