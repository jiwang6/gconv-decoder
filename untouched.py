# %% [markdown]
# *Erik J. Bekkers and Maxime W. Lafarge, Eindhoven University of Technology, the Netherlands*
# 
# *8 June 2018*
# 
# ***
# 
# *This DEMO was tested on a laptop with*:
# - *Windows as OS*
# - *Jupyter Notebook (version 5.5.0)*
# - *Python (version 3.5.5)*
# - *TensorFlow-GPU (versions 1.1 and higher)*
# - *An NVIDIA Quadro M1000M GPU*
# - *The following additional libraries installed for this demo to run: sklearn, scipy, and matplotlib*

# %% [markdown]
# # Basic usage of the se2cnn library

# %% [markdown]
# This jupyter demo will contain the basic usage examples of the se2cnn library with applications to digit recognition in the MNIST dataset. The se2cnn library contains 3 main layers (check the useage via *help(se2cnn.layers.z2_se2n)*):
# - **z2_se2n**: a lifting layer from 2D tensor to SE(2) tensor
# - **se2n_se2n**: a group convolution layer from SE(2) tensor to SE(2) tensor
# - **spatial_max_pool**: performs 2x2 spatial max-pooling of the spatial axes
# 
# The following functions are used internally, but may be of interest as well:
# - **rotate_lifting_kernels**: rotates the raw 2D lifting kernels (output is a set of rotated kernels)
# - **rotate_gconv_kernels**: rotates (shift-twists) the se2 kernels (planar rotation + orientation shift)
# 
# 
# 
# In this demo we will construct the following network:
# 
# | layer nr. | Layer                          | Tensor shape  |
# | --------- | ------------------------------ | ------------- |
# | 0         | input                          | (32 x 32 x 1) |
# | --------- | --------------------------------------------------- | -------------------------- |
# | 1         | 5x5 lifting convultion (+ReLU) | (28 x 28 x Ntheta x Nc)  |
# | 1         | 2x2 max pooling                | (14 x 14 x Ntheta x Nc)  |
# | --------- | --------------------------------------------------- | -------------------------- |
# | 2         | 5x5 group convultion (+ReLU)   | (10 x 10 x Ntheta x 2*Nc)  |
# | 2         | 2x2 max pooling                | (5 x 5 x Ntheta x 2*Nc)  |
# | --------- | --------------------------------------------------- | -------------------------- |
# | 3         | merge orientation+channel dim  | (5 x 5 x Ntheta 2*Nc) |
# | 3         | 5x5 2D convultion (+ReLU)      | (1 x 1 x 4*Nc)  |
# | --------- | --------------------------------------------------- | -------------------------- |
# | 4         | 1x1 2D convultion (+ReLU)      | (1 x 1 x 128)  |
# | --------- | --------------------------------------------------- | -------------------------- |
# | 5         | 1x1 2D convolution: the output layer | (10) |
# 
# Here Ntheta is the number of orientation samples to discretize the SE(2) group, Nc is the number of channels in the lifting layer. The first two layers are **roto-translation covariant**, meaning that the feature vectors rotate according to rotations of the input patterns (e.g. basic features do not need to be learned for each orientation). In layer 3 the orientation axis is merged with the channel axis, followed by 2D convolutions, making the subsequent layers **only translation covariant**. Here we made this choice due to the fact that in the MNIST dataset the characters always appear under the same orientation, and the task does not need to be rotation invariant.
# 
# There are several options to make a network completely roto-translation invariant:
# 1. For example one could stick with (5x5 and 1x1) SE(2) group convolutions all the way to the output layer, which would then provide a length 10 feature vector for each orientation. On can then simply do a maximum projection over the orientations (tf.reduce_max) to get the maximal response for each bin, followed by a softmax.
# 2. One coulde reduce the patch size to [1 x 1 x Ntheta x Nc] via group convolutions, apply a maximum projection over theta, and perform 1x1 (fully connected) 2D convolutions all the way to the end.

# %% [markdown]
# ***
# # Part 1: Load the libraries

# %% [markdown]
# ## Load the libraries

# %%
# Impor tensorflow and numpy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m
import time
# For validation
from sklearn.metrics import confusion_matrix
import itertools

# For plotting
from matplotlib import pyplot as plt

# Add the library to the system path
import os,sys
se2cnn_source =  os.path.join(os.getcwd(),'..')
if se2cnn_source not in sys.path:
    sys.path.append(se2cnn_source)

# Import the library
import se2cnn.layers

# %% [markdown]
# ## Useful functions

# %% [markdown]
# ### The se2cnn layers

# %% [markdown]
# For useage of the relevant layers defined in se2cnn.layers uncomment and run the following:

# %%
# help(se2cnn.layers.z2_se2n)
# help(se2cnn.layers.se2n_se2n)
# help(se2cnn.layers.spatial_max_pool)

# %% [markdown]
# ### Weight initialization

# %% [markdown]
# For initialization we use the initialization method for ReLU activation functions as proposed in:
# 
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

# %%
# Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
def weight_initializer(n_in, n_out):
    return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
    )

# %% [markdown]
# ### Confusion matrix plot

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %% [markdown]
# ### Size of a tf tensor

# %%
def size_of(tensor) :
    # Multiply elements one by one
    result = 1
    for x in tensor.get_shape().as_list():
         result = result * x 
    return result 

# %% [markdown]
# ***
# # Part 2: Load and format the MNIST dataset

# %% [markdown]
# The MNIS dataset consists of 28x28 images of handwritten characters. We are going to classify each input image into 1 of the 10 classes (the digits 0 to 9).

# %%
mnist = tf.keras.datasets.mnist

(train_data, train_labels), (eval_data, eval_labels) = mnist.load_data()

# %%
bigtot = 0

for image in train_data:
    # flatten the image
    image = image.flatten()

    # number of zeros in the image
    num_zeros = np.count_nonzero(image == 0)

    bigtot += num_zeros

print(1- bigtot/(60000*784))

# %% [markdown]
# By default the data is formatted as flattened arrays. Here were format them as 2D feature maps (with only 1 channel).

# %%
# Reshape to 2D multi-channel images
train_data_2D = train_data.reshape([len(train_data),28,28,1]) # [batch_size, Nx, Ny, Nc]
eval_data_2D = eval_data.reshape([len(eval_data),28,28,1])

# Plot the first sample
plt.plot()
plt.title('Digit = %d' % train_labels[0])
plt.imshow(train_data_2D[0,:,:,0])

# %% [markdown]
# We would like to have the patches to be of size 32x32 such that we can reduce it to 1x1 via 5x5 convolutions and max pooling layers. So, here we pad the images on the left and right with zeros.

# %%
train_data_2D = np.pad(train_data_2D,((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))
eval_data_2D = np.pad(eval_data_2D,((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

# %% [markdown]
# ***
# # Part 3: Build a graph (design the G-CNN)

# %% [markdown]
# ## Build a graph

# %%
graph = tf.Graph()
graph.as_default()
tf.reset_default_graph()

# %% [markdown]
# ## Settings

# %% [markdown]
# Kernel size and number of orientations

# %%
Ntheta = 12 # Kernel size in angular direction
Nxy=5       # Kernel size in spatial direction
Nc = 4      # Number of channels in the initial layer

# %% [markdown]
# ### Placeholders

# %%
inputs_ph = tf.placeholder( dtype = tf.float32, shape = [None,32,32,1] )
labels_ph = tf.placeholder( dtype = tf.int32, shape = [None,] )

# %% [markdown]
# ### Prepare for the first layer

# %%
tensor_in = inputs_ph
Nc_in = 1

# %% [markdown]
# ### Save the kernels to a library for later inspection

# %%
kernels={}

# %% [markdown]
# ## Layer 1: The lifting layer

# %%
with tf.variable_scope("Layer_{}".format(1)) as _scope:
    ## Settings
    Nc_out = Nc

    ## Perform lifting convolution
    # The kernels used in the lifting layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Lifting layer
    tensor_out, kernels_formatted = se2cnn.layers.z2_se2n(
                            input_tensor = tensor_in,
                            kernel = kernels_raw,
                            orientations_nb = Ntheta)
    # Add bias
    tensor_out = tensor_out + bias
    
    ## Perform (spatial) max-pooling
    tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
tensor_in.get_shape()

# %% [markdown]
# ## Layer 2: SE2-conv, max-pool, relu

# %%
with tf.variable_scope("Layer_{}".format(2)) as _scope:
    ## Settings
    Nc_out = 2*Nc

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Ntheta,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Ntheta*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # The group convolution layer
    tensor_out, kernels_formatted = se2cnn.layers.se2n_se2n(
                            input_tensor = tensor_in,
                            kernel = kernels_raw)
    tensor_out = tensor_out + bias
    
    ## Perform max-pooling
    tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
tensor_in.get_shape()

# %% [markdown]
# ## Layer 3: 2D fully connected layer (5x5)

# %%
# Concatenate the orientation and channel dimension
tensor_in = tf.concat([tensor_in[:,:,:,i,:] for i in range(Ntheta)],3)
Nc_in = tensor_in.get_shape().as_list()[-1]

# 2D convolution layer
with tf.variable_scope("Layer_{}".format(3)) as _scope:
    ## Settings
    Nc_out = 4*Nc

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [Nxy,Nxy,Nc_in,Nc_out],
                        initializer=weight_initializer(Nxy*Nxy*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_raw

# %%
tensor_in.get_shape()

# %% [markdown]
# ## Layer 4: Fully connected layer (1x1)

# %%
# 2D convolution layer
with tf.variable_scope("Layer_{}".format(4)) as _scope:
    ## Settings
    Nc_out = 128

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Nc_in,Nc_out],
                        initializer=weight_initializer(1*1*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))
    # Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_raw

# %%
tensor_in.get_shape()

# %% [markdown]
# ## Layer 5: Fully connected (1x1) to output

# %%
with tf.variable_scope("Layer_{}".format(5)) as _scope:
    ## Settings
    Nc_out = 10

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Nc_in,Nc_out],
                        initializer=weight_initializer(1*1*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))

    
    ## Convolution layer
    tensor_out = tf.nn.conv2d(
                        input = tensor_in,
                        filter=kernels_raw,
                        strides=[1, 1, 1, 1],
                        padding="VALID")
    tensor_out = tensor_out + bias
    
    ## The output logits
    print (tensor_out.get_shape())
    logits = tensor_out[:,0,0,:]
    predictions = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits)
    
    ## Save the kernels for later inspection
    kernels[_scope.name] = kernels_raw

# %%
logits.get_shape()

# %% [markdown]
# ## Define the loss and the optimizer

# %%
# Cross-entropy loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph, logits=logits)

# %%
#-- Define the l2 loss 
weightDecay=5e-4
# Get the raw kernels
variables_wd = tf.get_collection('raw_kernels')
print('-----')
print('RAW kernel shapes:')
for v in variables_wd: print( "[{}]: {}, total nr of weights = {}".format(v.name, v.get_shape(), size_of(v)))
print('-----')
loss_l2 = weightDecay*sum([tf.nn.l2_loss(ker) for ker in variables_wd])

# %%
# Configure the Training Op (for TRAIN mode)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train_op = optimizer.minimize(
    loss=loss + loss_l2,
    global_step=tf.train.get_global_step())

# %% [markdown]
# ***
# # Part 4: Train and test the G-CNN

# %% [markdown]
# ## Begin session

# %%
#-- Start the (GPU) session
initializer = tf.global_variables_initializer()
session = tf.Session(graph=tf.get_default_graph()) #-- Session created
session.run(initializer)

# %% [markdown]
# ## Optimization

# %% [markdown]
# In each epoch we pass over all input samples in batch sizes of batch_size

# %%
batch_size=100
n_epochs=10

# %% [markdown]
# Loop over the input stack in batch of size "batch_size".

# %%
# check tf can see the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
for epoch_nr in range(n_epochs):
    loss_average = 0
    data = train_data_2D
    labels = train_labels
    # KBatch settings
    NItPerEpoch = m.floor(len(data)/batch_size) #number of iterations per epoch
    samples=np.random.permutation(len(data))
    # Loop over dataset
    tStart = time.time()
    for iteration in range(NItPerEpoch):
        feed_dict = {
                inputs_ph: np.array(data[samples[iteration*batch_size:(iteration+1)*batch_size]]),
                labels_ph: np.array(labels[samples[iteration*batch_size:(iteration+1)*batch_size]])
                }
        operators_output = session.run([ loss , train_op ], feed_dict)
        loss_average += operators_output[0]/NItPerEpoch
    tElapsed = time.time() - tStart
    print('Epoch ' , epoch_nr , ' finished... Average loss = ' , round(loss_average,4) , ', time = ',round(tElapsed,4))

# %% [markdown]
# ## Validation

# %%
batch_size = 1000
labels_pred = []
for i in range(round(len(eval_data_2D)/batch_size)):
    [ labels_pred_batch ] = session.run([ predictions ], { inputs_ph: eval_data_2D[i*batch_size:(i+1)*batch_size] })
    labels_pred = labels_pred + list(labels_pred_batch)
labels_pred = np.array(labels_pred)

# %% [markdown]
# Compare the first 10 results with the ground truth

# %%
print(labels_pred[0:10])
print(eval_labels[0:10])

# %% [markdown]
# The accuracy (average nr of successes)

# %%
((labels_pred - eval_labels)**2==0).astype(float).mean()

# %% [markdown]
# Total nr of errors

# %%
((labels_pred - eval_labels)**2>0).astype(float).sum()

# %% [markdown]
# Error rate

# %%
100*((labels_pred - eval_labels)**2>0).astype(float).mean()

# %% [markdown]
# Plot a confusion matrix to see what kind of errors are made

# %%
cm = confusion_matrix(eval_labels, labels_pred)
plot_confusion_matrix(cm, range(10))


