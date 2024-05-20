# %%
# Impor tensorflow and numpy
#import tensorflow as tf
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
se2cnn_source =  os.path.join(os.getcwd(),'../se2cnn/')
if se2cnn_source not in sys.path:
    sys.path.append(se2cnn_source)

# Import the library
import se2cnn.layers

# %%
# check tf can see the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
# Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
def weight_initializer(n_in, n_out):
    return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
    )

def size_of(tensor) :
    # Multiply elements one by one
    result = 1
    for x in tensor.get_shape().as_list():
         result = result * x 
    return result 

# %% [markdown]
# ## Load and Format Dataset

# %%
code_distance = 5
error_rate = 180

observations = 70000

dataset = np.load(f'high-level/test-datasets/HL_data_{code_distance}_{error_rate}_{observations}.npy')

nontest, test = np.split(dataset, [60000])

nontest_data_unzipped = nontest[:, :code_distance**2 *2]
# last 4 columns are the labels
nontest_labels= nontest[:, -4:]

test_data_unzipped = test[:, :code_distance**2 *2]
test_labels= test[:, -4:]



# %%
from Datavis import visualize_observation

# visualizing observation and label
# red: x-error, yellow: z-error, green: x and z error, blue: stabilizer activation

observation_index = 0

visualize_observation(nontest[observation_index][:-4])
print(nontest[observation_index][-4:])


# %% [markdown]
# ## Zip Together Syndromes

# %%
# new np ndarrays
nontest_data = []

test_data = []

for observation in nontest_data_unzipped:

    # split in half
    observation = observation.reshape(code_distance *2, code_distance)
    first_half = observation[:code_distance]
    second_half = observation[code_distance:]
    # zip them together
    observation = np.array(list(zip(first_half, second_half))).flatten()


    nontest_data.append(observation)

for observation in test_data_unzipped:
    # split in half
    observation = observation.reshape(code_distance *2, code_distance)
    first_half = observation[:code_distance]
    second_half = observation[code_distance:]
    # zip them together
    observation = np.array(list(zip(first_half, second_half))).flatten()

    test_data.append(observation)

nontest_data = np.array(nontest_data)
test_data = np.array(test_data)

print(nontest_data.shape)
print(test_data.shape)


# %% [markdown]
# ## Closed Surface Padding

# %%
temp_nt = nontest_data
temp_t = test_data

nontest_data = []
test_data = []

kernellll = 3

pad_size = kernellll - 1

for observation in temp_nt:
    # reshape to 2D
    observation = observation.reshape(2*code_distance, code_distance)
    # left pad is last two columns of observation
    left_pad = observation[:,-pad_size:]
    # right pad is first two columns of observation
    right_pad = observation[:,:pad_size]    
    # pad the observation
    observation = np.hstack((left_pad, observation, right_pad))

    bottom_pad = observation[:pad_size,:]
    top_pad = observation[-pad_size:,:]

    observation = np.vstack((top_pad, observation, bottom_pad))

    # flatten the observation
    observation = observation.flatten()

    nontest_data.append(observation)

for observation in temp_t:
    # reshape to 2D
    observation = observation.reshape(2*code_distance, code_distance)

    # left pad is last two columns of observation
    right_pad = observation[:,:pad_size]
    # right pad is first two columns of observation
    left_pad = observation[:,-pad_size:]

    # pad the observation
    observation = np.hstack((left_pad, observation, right_pad))

    bottom_pad = observation[:pad_size,:]
    top_pad = observation[-pad_size:,:]

    observation = np.vstack((top_pad, observation, bottom_pad))

    # flatten the observation
    observation = observation.flatten()

    test_data.append(observation)

nontest_data = np.array(nontest_data)
test_data = np.array(test_data)


# %%
# Reshape to 2D multi-channel images
nontest_data_2D = nontest_data.reshape(len(nontest_data), code_distance*2 + 2 * pad_size, code_distance + 2* pad_size, 1) # data shape here
test_data_2D = test_data.reshape(len(test_data), code_distance*2 + 2*pad_size, code_distance + 2*pad_size, 1)

print(f"nontest_data_2D.shape = {nontest_data_2D.shape}")


# %%
print(nontest_data[0])

print(nontest_data_2D[0,:,:,0])

# %% [markdown]
# # one-hot encode all labels

# %%
nontest_labels_encoded = np.array([int(''.join(map(str,single_label)),2) for single_label in nontest_labels])
test_labels_encoded = np.array([int(''.join(map(str,single_label)),2) for single_label in test_labels])

print(nontest_labels[:5])
print(nontest_labels_encoded[:5])

print(test_labels[:5])
print(test_labels_encoded[:5])



# %%

print(test_labels[:5])
# head of labels
print(test_labels_encoded[:5])


# %%
# histogram of nontest_labels_encoded
plt.title(f"Histogram of labels, L={code_distance}, p=0.{error_rate}")
plt.hist(nontest_labels_encoded, bins=16)
# show all x ticks
plt.xticks(np.arange(16))
#plt.savefig(f"images/histogram_L_{code_distance}_p_{error_rate}.png")

plt.show()




# %% [markdown]
# ## Define GCNN Structure

# %%
Ntheta = 4 # Kernel size in angular direction
Nxy = kernellll     # Kernel size in spatial direction
Nc = 4     # Number of channels in the initial layer

graph = tf.Graph()
graph.as_default()
tf.compat.v1.reset_default_graph()

inputs_ph = tf.placeholder( dtype = tf.float32, shape = [
    None, 
    nontest_data_2D.shape[1], 
    nontest_data_2D.shape[2], 
    nontest_data_2D.shape[3]] )
labels_ph = tf.placeholder( dtype = tf.int32, shape = [None,] )


tensor_in = inputs_ph
Nc_in = 1

kernels={}

# %%
print(inputs_ph.shape)
print(labels_ph.shape)

# %% [markdown]
# ### Layer 1

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
    #tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)

    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(tensor_in.get_shape())

# %% [markdown]
# ### Layer 2

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
    
    #tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)

    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)
    
    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(tensor_in.get_shape())

# %% [markdown]
# ### Layer 3

# %%
with tf.variable_scope("Layer_{}".format(3)) as _scope:
    ## Settings
    Nc_out = 4*Nc

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

    #tensor_out = se2cnn.layers.spatial_max_pool( input_tensor=tensor_out, nbOrientations=Ntheta)
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)
    
    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(tensor_in.get_shape())

# %% [markdown]
# ### Layer 4

# %%
with tf.variable_scope("Layer_{}".format(4)) as _scope:
    ## Settings
    Nc_out = 8*Nc

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
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)
    
    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(tensor_in.get_shape())

# %% [markdown]
# ### Layer 5

# %%
with tf.variable_scope("Layer_{}".format(5)) as _scope:
    ## Settings
    Nc_out = 16*Nc

    Ny = tensor_in.get_shape().as_list()[1]
    Nx = tensor_in.get_shape().as_list()[2]

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Ntheta,Nc_in,Nc_out],
                        initializer=weight_initializer(Ny*Nx*Ntheta*Nc_in,Nc_out))
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
    
    ## Apply ReLU
    tensor_out = tf.nn.relu(tensor_out)
    
    ## Prepare for the next layer
    tensor_in = tensor_out
    Nc_in = Nc_out
    
    ## Save kernels for inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(tensor_in.get_shape())

# %% [markdown]
# ### Layer 6

# %%
# create mapping matrix
identity = np.identity(4)
# swap 2n and 3rd rows
identity[[1, 2]] = identity[[2, 1]]
rearr = np.kron(identity, identity)


with tf.variable_scope("Layer_{}".format(6)) as _scope:
    ## Settings
    Nc_out = 16

    ## Perform group convolution
    # The kernels used in the group convolution layer
    kernels_raw = tf.get_variable(
                        'kernel', 
                        [1,1,Ntheta,Nc_in,Nc_out],
                        initializer=weight_initializer(1*1*Ntheta*Nc_in,Nc_out))
    tf.add_to_collection('raw_kernels', kernels_raw)
    bias = tf.get_variable( # Same bias for all orientations
                        "bias",
                        [1, 1, 1, 1, Nc_out], 
                        initializer=tf.constant_initializer(value=0.01))

    
    ## Convolution layer
    tensor_out, kernels_formatted = se2cnn.layers.se2n_se2n(
                            input_tensor = tensor_in,
                            kernel = kernels_raw)
    tensor_out = tensor_out + bias

    print(tensor_out.get_shape())
    
    ## The output logits
    logits = tensor_out[:,0,0,:,:] # reduces shape to tensor 

    print(logits.get_shape())
    
    logits = logits[:,0,:] + logits[:,1,:] @ rearr + logits[:,2,:] + logits[:,3,:] @ rearr

    predictions = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits)
    
    ## Save the kernels for later inspection
    kernels[_scope.name] = kernels_formatted

# %%
print(logits.get_shape())

# %% [markdown]
# ## Loss Function and Optimizers

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

# %%
#-- Start the (GPU) session
initializer = tf.global_variables_initializer()
session = tf.Session(graph=tf.get_default_graph()) #-- Session created
session.run(initializer)

# %%
batch_size=100
n_epochs=20

# %%
average_losses = {}

for epoch_nr in range(1, n_epochs+1):
    loss_average = 0
    data = nontest_data_2D
    labels = nontest_labels_encoded
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

    average_losses[epoch_nr] = loss_average

    if epoch_nr < 10:
        print('Epoch ' , epoch_nr , ' finished... Average loss = ' , round(loss_average,4) , ', time = ',round(tElapsed,4))
    else:
        print('Epoch' , epoch_nr , ' finished... Average loss = ' , round(loss_average,4) , ', time = ',round(tElapsed,4))

# %% [markdown]
# ## Validation

# %%
import time
batch_size = 1000
labels_pred = []
# start time
tStart = time.time()
for i in range(round(len(test_data_2D)/batch_size)):
    [ labels_pred_batch ] = session.run([ predictions ], { inputs_ph: test_data_2D[i*batch_size:(i+1)*batch_size] })
    labels_pred = labels_pred + list(labels_pred_batch)
# end time
tElapsed = time.time() - tStart
print('Time elapsed for prediction = ',round(tElapsed,4),' seconds')
labels_pred = np.array(labels_pred)

# %%
print(labels_pred[0:10])
print(test_labels_encoded[0:10])

# %% [markdown]
# Accuracy

# %%
((labels_pred - test_labels_encoded)**2==0).astype(float).mean()

# %% [markdown]
# Total Errors

# %%
((labels_pred - test_labels_encoded)**2>0).astype(float).sum()

# %% [markdown]
# Error Rate

# %%
100*((labels_pred - test_labels_encoded)**2>0).astype(float).mean()

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=f'Confusion Matrix for L={code_distance}, p=0.{error_rate}',
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
    # make bigger
    plt.gcf().set_size_inches(8, 8)

    #plt.savefig(f"cms/cm_L_{code_distance}_p_{error_rate}.png")

    
    plt.show()

# %%
cm = confusion_matrix(test_labels_encoded, labels_pred)
plot_confusion_matrix(cm, range(16))
# save to file


# %%
# calculate the per class f1 score

def f1_score(cm):
    f1 = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = np.sum(cm[:,i]) - tp
        fn = np.sum(cm[i,:]) - tp
        f1.append(tp/(tp + 0.5*(fp + fn)))
    return f1

f1_scores = f1_score(cm)

print(f1_scores)

# save to file
with open(f"f1-scores/f1_L_{code_distance}_p_{error_rate}.txt", "a") as f:
    f.write(str(f1_scores))


# %%
# plot the f1 score
plt.bar(range(16), f1_score(cm))
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.title(f'F1 Score per Class, L={code_distance}, p=0.{error_rate}')
plt.xticks(range(16))
plt.savefig(f"images/f1_score_L_{code_distance}_p_{error_rate}.png")
plt.show()

# %%
# plot losses
plt.plot(average_losses.keys(), average_losses.values())
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title(f'Average Loss per Epoch, L={code_distance}, p=0.{error_rate}')
plt.xticks(range(1, n_epochs+1))
plt.savefig(f"images/loss_L_{code_distance}_p_{error_rate}.png")


