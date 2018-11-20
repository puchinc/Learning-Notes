""""""""""""
" APPENDIX "
""""""""""""

# Loss Functions
* Cross-Entropy
    L(y, y') = -y * log(y') + (1 - y) * log(1 - y')
             = - ∑ p(y) * log(p(y'))

分类问题，都用 onehot + cross entropy
training 过程中，分类问题用 cross entropy，回归问题用 mean squared error。
training 之后，validation / testing 时，使用 classification error，更直观，而且是我们最关注的指标。



deep = Flatten()(deep)
deep = Dropout(p_dropout)(deep)
deep = Dense(deep_output_size)(deep)
deep = BatchNormalization()(deep) # solve gradient vanish
deep = Activation('relu')(deep) # solve gradient vanish

# Batch Normalization
Machine learning methods tend to work better when their input data consists of uncorrelated features with zero mean and unit variance. 
It is possible that this normalization strategy could reduce the representational power of the network, since it may sometimes be optimal for certain layers to have features that are not zero-mean or unit variance. To this end, the batch normalization layer includes learnable shift and scale parameters for each feature dimension.

# Activation Functions
* Sigmoid [0, 1] can lead us to gradient decent problem where the updates are so low.
    A = 1 / (1 + np.exp(-z)) # Where z is the input matrix
* Tahn [-1, 1], tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.
    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
* RELU = max(0,z) # so if z is negative the slope is 0 and if z is positive the slope remains linear.


# CONVOLUTION NEURAL NETWORK
Advantages:
1. Parameter sharing.
A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.
2. sparsity of connections.
In each layer, each output value depends only on a small number of inputs which makes it translation invariance.

## Padding
The general rule now, if a matrix nxn is convolved with fxf filter/kernel and padding p give us n+2p-f+1,n+2p-f+1 matrix.
* Same input output size: P = (f-1) / 2
* Convolution with stride s: N * N -> (N+2P-F)/S + 1 * (N+2P-F)/S + 1
* p = (n * s - n + f - s) / 2, When s = 1 ==> P = (f-1) / 2

## Pooling
Example of Max pooling on 3D input:
Input: 4x4x10
Max pooling size = 2 and stride = 2
Output: 2x2x10

