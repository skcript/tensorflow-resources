from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from scipy import diag
from numpy.random import multivariate_normal

# What is Multivariate normal distribution?

# In probability theory and statistics, the multivariate normal distribution
# or multivariate Gaussian distribution, is a generalization of the one-dimensional
# (univariate) normal distribution to higher dimensions.

# The multivariate normal distribution is often used to describe,
# at least approximately, any set of (possibly) correlated real-valued random
# variables each of which clusters around a mean value.

# https://en.wikipedia.org/wiki/Multivariate_normal_distribution

# METHOD: Classification
# Classify any random point to one of three different classes.
alldata = ClassificationDataSet(2, 1, nb_classes=3)

# DATASET: Multivariate Gaussian Distribution
# Produce a set of points in 2D belonging to three different classes.

# Assumed means of three different classes
means = [(-1,0), (2,4), (3,1)]
# Assumed covariance of three different classes
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]

# Gather a dataset of 400 values, where each value randomly belongs to one class
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass], cov[klass])
        alldata.addSample(input, [klass])

# Randomly split the dataset into 75% training and 25% test data sets.
tstdata, trndata = alldata.splitWithProportion( 0.25 )

# For neural network classification, it is highly advisable to encode classes
# with one output neuron per class.
# Note that this operation duplicates the original targets and stores
# them in an (integer) field named ‘class’.
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

# Explore trndata and tstdata
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# Now build a feed-forward network with 5 hidden units.
# The input and output layer size must match the dataset’s input and target dimension.
# You could add additional hidden layers by inserting more numbers giving the desired layer sizes.
fnn = buildNetwork(trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer)

# Set up a trainer that basically takes the network and training dataset as input.
# We are using a BackpropTrainer for this.
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

# Start the training iterations.
for i in range(20):
    trainer.trainEpochs(5)

# Evaluate the network on the training and test data.
trnresult = percentError( trainer.testOnClassData(),
                        trndata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=tstdata),
                        tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
      "train error: %5.2f%%" % trnresult, \
      "test error: %5.2f%%" % tstresult

# To classify new data, just use the activate method
# self.fnn.classify(data)
