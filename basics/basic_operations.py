from __future__ import print_function

import tensorflow as tf

# Start tf session
sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a+b

# Print out operation everything is operation
print(a)
print(b)
print(c)

print(a+b)


# Print out the result of operation
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
print(sess.run(a+b))
