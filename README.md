# conv-net
### *A NumPy implementation of the famed Convolutional Neural Network*

## Overview
CNNs are well-known for their ability to recognize patterns present in images, and so the problem of choice is fashion classification using grayscale images from Fashion MNIST dataset. Both the training instances and the labels will be read as NumPy arrays. Each picture is 28x28 pixels. The dataset consists of a training set of 60.000 examples and a test set of 10.000 examples. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. Each training and test instance is assigned to one of the following labels: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag and 9 Ankle boot. We choose to perform a supervised learning based classification task. To do so, we aim to learn to predict clothes based on 28x28 grayscale images using a CNN classifier. 

Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

<p align="center">
<img width="80%" alt="fashion_MNIST" src="https://teaching.pages.centralesupelec.fr/deeplearning-lectures-build/data/00-pytorch-fashionMnist/fashionMNIST_samples.png">
</p>

## How Convolutional Neural Networks learn
### Convolutions

CNNs make use of **filters** (also known as kernels), to detect what features, such as edges, are present throughout an image. A filter is just a matrix of values, called weights, that are trained to detect specific features. The filter moves over each part of the image to check if the feature it is meant to detect is present. To provide a value representing how confident it is that a specific feature is present, the filter carries out a **convolution operation**, which is an element-wise product and sum between two matrices.

<p align="center">
<img width="50%" alt="convolution" src="https://miro.medium.com/proxy/0*dRD6PhKOnnCIhz15.jpg">
</p>

When the feature is present in part of an image, the convolution operation between the filter and that part of the image results in a real number with a high value. If the feature is not present, the resulting value is low.

So that the CNN can learn the values for a filter that detect features present in the input data, the filter must be passed through a non-linear mapping. The output of the convolution operation between the filter and the input image is summed with a bias term and passed through a non-linear activation function. The purpose of the activation function is to introduce non-linearity into our network. Since our input data is non-linear (it is infeasible to model the pixels that form a handwritten signature linearly), our model needs to account for that. To do so, we use the Rectified Linear Unit (ReLU) activation function:

<p align="center">
<img width="40%" alt="ReLU" src="https://miro.medium.com/proxy/1*oePAhrm74RNnNEolprmTaQ.png">
</p>

As you can see, the ReLU function is quite simple; values that are less than or equal to zero become zero and all positive values remain the same. Usually, a network utilizes more than one filter per layer. When that is the case, the outputs of each filter`s convolution over the input image are concatenated along the last axis, forming a final 3D output.

#### The Code
Using NumPy, we can program the convolution operation quite easily. The convolution function convolves all the filters over the image. At each step, the filter is multipled element-wise (*) with a section of the input image. The result of this element-wise multiplication is then summed to obtain a single value using NumPy`s sum method, and then added with a bias term.

```python
def convolution(input_image_matrix, kernel_weights_matrix, bias, stride=1):
    """
    Convolve `kernel_weights_matrix` over `input_image_matrix` using stride `stride`
    """
    (no_channels_out, no_channels_in_from_kernel, kernel_size, _) = kernel_weights_matrix.shape  # filter dimensions
    no_channels_in_from_image, initial_h, initial_w = input_image_matrix.shape  # image dimensions
    output_h = int((initial_h - kernel_size) / stride) + 1  # calculate output dimensions
    output_w = int((initial_w - kernel_size) / stride) + 1
    assert(no_channels_in_from_image == no_channels_in_from_kernel,
           "Filter and Image dimension mismatch. Expected {} but instead got {}."
           .format(no_channels_in_from_image, no_channels_in_from_kernel))
    output_matrix = np.zeros((no_channels_out, output_h, output_w))

    # convolve the filter over every part of the image, adding the bias at each step.
    param_dict = {"f": kernel_size, "filt": kernel_weights_matrix, "s_x": stride, "s_y": stride,
                  "image": input_image_matrix, "in_dim_w": initial_w, "in_dim_h": initial_h, "n_c": no_channels_out,
                  "out": output_matrix, "bias": bias}
    output_matrix = apply(param_dict, do_one_conv)

    return output_matrix
```

```python
def apply(param_dict, function):
    """
    Parametrized convolution execution body that applies a function (sliding window) over the whole input matrix.
    """
    kernel_size = param_dict["f"]
    for curr_c in range(param_dict["n_c"]):
        curr_y = out_y = 0
        while curr_y + kernel_size <= param_dict["in_dim_h"]:
            curr_x = out_x = 0
            while curr_x + kernel_size <= param_dict["in_dim_w"]:
                param_dict["out"] = function(curr_x, curr_y, curr_c, out_x, out_y, param_dict)
                curr_x += param_dict["s_x"]
                out_x += 1
            curr_y += param_dict["s_y"]
            out_y += 1
    return param_dict["out"]


def do_one_conv(curr_x, curr_y, curr_channels, out_x, out_y, param_dict):
    """
    Parametrized function that executes one convolution over a certain window in the input matrix.
    """
    kernel_size = param_dict["f"]

    param_dict["out"][curr_channels, out_y, out_x] = \
        np.sum(param_dict["filt"][curr_channels] *
               param_dict["image"][:, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size]) + \
        param_dict["bias"][curr_channels]
    return param_dict["out"]
```

The ```filt``` input is initialized using a standard normal distribution and ```bias``` is initialized to be a vector of zeros. After one or two convolutional layers, it is common to reduce the size of the representation produced by the convolutional layer. This reduction in the representation`s size is known as **downsampling**.

### Downsampling

To speed up the training process and reduce the amount of memory consumed by the network, we try to reduce the redundancy present in the input feature. There are a couple of ways we can downsample an image, but for this post, we will look at the most common one: max pooling. In max pooling, a window passes over an image according to a set stride (how many units to move on each pass). At each step, the maximum value within the window is pooled into an output matrix, hence the name **max pooling**.

In the following visual, a window of size `kernel_size=2` passes over an image with a `stride=2`.  `kernel_size` denotes the dimensions of the max pooling window (red box) and `stride` denotes the number of units the window moves in the x and y-direction. At each step, the maximum value within the window is chosen:

<p align="center">
<img width="60%" alt="max_pooling" src="https://miro.medium.com/max/1058/0*wH3GmU0JP9zQeODt.gif">
</p>

#### The Code

```python
def maxpool(input_image_matrix, kernel_size=2, stride=2, kernel_weights_matrix=None, bias=None):
    """
    Downsample `image` using kernel size `f` and stride `s`
    """
    no_channels, initial_h, initial_w = input_image_matrix.shape
    output_h = int((initial_h - kernel_size) / stride) + 1
    output_w = int((initial_w - kernel_size) / stride) + 1

    output_matrix = np.zeros((no_channels, output_h, output_w))
    param_dict = {"f": kernel_size, "filt": None, "s_x": stride, "s_y": stride, "image": input_image_matrix,
                  "in_dim_w": initial_w,
                  "in_dim_h": initial_h, "n_c": no_channels, "out": output_matrix, "bias": None}
    output_matrix = apply(param_dict, do_one_pool)
    return output_matrix
```

```python
def do_one_pool(curr_x, curr_y, curr_channels, out_x, out_y, param_dict):
    """
    Parametrized function that executes one max pooling over a certain window in the input matrix.
    """
    kernel_size = param_dict["f"]
    param_dict["out"][curr_channels, out_y, out_x] = np.max(
        param_dict["image"][curr_channels, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size])
    return param_dict["out"]
```

After multiple convolutional layers and downsampling operations, the 3D image representation is converted into a feature vector that is passed into a Multi-Layer Perceptron, which merely is a neural network with at least three layers. This is referred to as a **Fully-Connected Layer**.

### Fully-Connected Layer

In the fully-connected operation of a neural network, the input representation is flattened into a feature vector and passed through a network of neurons to predict the output probabilities. The following image describes the flattening operation:

<p align="center">
<img width="40%" alt="fully_connected_layer" src="https://miro.medium.com/max/1400/1*Lzx2pNLpHjGTKcofsaSH1g.png">
</p>

The rows are concatenated to form a long feature vector. If multiple input layers are present, its rows are also concatenated to form an even longer feature vector. The feature vector is then passed through multiple dense layers. At each dense layer, the feature vector is multiplied by the layer`s weights, summed with its biases, and passed through a non-linearity.

#### The Code

```python
class Flatten(Layer):
    def name(self):
        return "Flatten"

    def __init__(self):
        super(Flatten, self).__init__()
        self.original_shape = None

    def forward(self, x):
        (nf2, dim2, _) = x.shape
        self.original_shape = x.shape
        flat_x = x.reshape((nf2 * dim2 * dim2, 1))
        return flat_x

    def backprop(self, dx, x):
        dx = dx.reshape(self.original_shape)  # reshape fully connected into dimensions of pooling layer
        dx = self.activation.backprop_activation(dx, x)
        return dx, None, None
```

The following image visualizes the fully connected operation and dense layers:

<p align="center">
<img width="80%" alt="fully_connected" src="https://miro.medium.com/max/1400/1*6qFBUoRbA6X7aAVjq-6SfA.png">
</p>

### Output Layer

The output layer of a CNN is in charge of producing the probability of each class (each digit) given the input image. To obtain these probabilities, we initialize our final Dense layer to contain the same number of neurons as there are classes. The output of this dense layer then passes through the **Softmax activation function**, which maps all the final dense layer outputs to a vector whose elements sum up to one:

<p align ="center">
<img src="https://render.githubusercontent.com/render/math?math=\sigma (x_{j}) = \frac{e^{x_{j}}}{\sum_{i} e^{x_{i}}}">
</p>

where <img src="https://render.githubusercontent.com/render/math?math=x"> denotes each element in the final layer`s outputs.

#### The Code

```python
def _softmax(x):
    out = np.exp(x)
    return out / np.sum(out)
```

### The Network

Given the relatively low amount of classes (10 in total) and the small size of each training image (28x28), a simple network architecture was chosen to solve the task ofclothes classification. The network uses two consecutive convolutional layers followed by a max pooling operation to extract features from the input image. After the max pooling operation, the representation is flattened and passed through a Multi-Layer Perceptron (MLP) to carry out the task of classification.

## The Implementation

### Step 1: Getting the data

The Fashion MNIST training and test data can be obtained from *https://github.com/zalandoresearch*. The files store image and label data as tensors, so the files must be read through their bytestream. We define two helper methods to perform the extraction:

#### The Code

```python
def extract_data(filename, num_images, IMAGE_WIDTH):
    """
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w],
    where `m` is the number of training examples.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    """
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
```

### Step 2: Initialize parameters

We first define methods to initialize both the filters for the convolutional layers and the weights for the dense layers. To make for a smoother training process, we initialize each filter with a mean of 0 and a standard deviation of 1.

#### The Code

```python
def initializeFilter(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01
```

### Step 3: Define the backpropagation operations

To compute the gradients that will force the network to update its weights and optimize its objective, we need to define methods that backpropagate gradients through the convolutional and max pooling layers.

#### The Code

```python
def convolutionBackward(dconv_prev, input_image_matrix, kernel_weights_matrix, stride):
    """
    Backpropagation through a convolutional layer.
    """
    (no_channels_out, no_channels_in_from_kernel, kernel_size, _) = kernel_weights_matrix.shape
    (no_channels_in_from_image, source_h, source_w) = input_image_matrix.shape
    # initialize derivatives
    dout = np.zeros(input_image_matrix.shape)
    dfilt = np.zeros(kernel_weights_matrix.shape)
    dbias = np.zeros((no_channels_out, 1))
    param_dict = {"conv_in": input_image_matrix, "dbias": dbias, "dconv_prev": dconv_prev, "dfilt": dfilt, "dout": dout,
                  "f": kernel_size, "filt": kernel_weights_matrix, "n_c": no_channels_out, "orig_dim_h": source_h,
                  "orig_dim_w": source_w, "s": stride}
    dict_out = apply_bkwd(param_dict, do_one_conv_bkwd)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias
	
	
	def maxpoolBackward(dpool, input_image_matrix, kernel_size, stride):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the
    original maxpooling during the forward step.
    """
    (no_channels, source_h, source_w) = input_image_matrix.shape

    dout = np.zeros(input_image_matrix.shape)

    dict_in = {"dout": dout, "dconv_prev": dpool, "f": kernel_size, "n_c": no_channels, "conv_in": input_image_matrix,
               "orig_dim_h": source_h, "orig_dim_w": source_w, "s": stride, "dfilt": None, "dbias": None}
    dict_out = apply_bkwd(dict_in, do_one_pool_bkwd)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias
```

### Step 4: Building the network

In the spirit abstraction, we now define a method that combines the forward and backward operations of a convolutional neural network. It takes the network`s parameters and hyperparameters as inputs and spits out the gradients:

#### The Code

```python
def build_model(num_classes=10, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8, save_path='params.pkl'):
    model = SequentialModel()
    # Initializing all the parameters

    # INITIALIZE CONV & FC LAYERS WEIGHTS (co, ci, kh, kw) & BIASES
    conv_1 = Conv2D(out_dim=num_filt1, in_dim=img_depth, kernel=(f, f))
    conv_1.set_activation(relu)
    model.add(conv_1)

    conv_2 = Conv2D(out_dim=num_filt2, in_dim=num_filt1, kernel=(f, f))
    conv_2.set_activation(relu)
    model.add(conv_2)

    pooled = MaxPool(kernel=(2, 2))
    pooled.set_activation(idx)
    model.add(pooled)

    flatten = Flatten()
    flatten.set_activation(idx)
    model.add(flatten)

    dense3 = Dense(out_dim=128, in_dim=800)
    dense3.set_activation(relu)
    model.add(dense3)

    dense4 = Dense(out_dim=num_classes, in_dim=128)
    dense4.set_activation(softmax)
    model.add(dense4)

    optimizer = AdamOptimizer(num_classes=num_classes, img_dim=img_dim)
    optimizer.set_loss(categoricalCrossEntropy)
    optimizer.addCallbacks({DumpModelPickleCallback.get_name(): DumpModelPickleCallback(save_path=save_path)})
    optimizer.setFrequency(1)  # execute callbacks every n-th epoch

    model.set_optimizer(optimizer)

    return model
```

### Step 5: Training the network

To efficiently force the network`s parameters to learn meaningful representations, we use the Adam optimization algorithm. The first one refers to the images being multiplied with the model’s parameters in order to obtain a prediction for each image. The algorithm will see the training examples and compare the predictions to the ground truth labels. Furthermore, the model learns by using gradient-based learning. It learns the weights of convolutional kernels that extract the features and it also learns weights after backward propagation of errors through the cross-entropy loss:

<p align ="center">
<img src="https://render.githubusercontent.com/render/math?math=CE = - \sum_{x} p(x) log q(x)">
</p>

where <img src="https://render.githubusercontent.com/render/math?math=p(x)"> is true class distribution and <img src="https://render.githubusercontent.com/render/math?math=q(x)"> is predicted class distribution.

#### The Code

```python
    def train(self, model, train_data):
        cost = []

        print("LR:" + str(self.lr) + ", Batch Size:" + str(self.batch_size))

        for indx, epoch in enumerate(range(self.num_epochs)):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + self.batch_size] for k in range(0, train_data.shape[0], self.batch_size)]

            t = tqdm(batches)
            for x, batch in enumerate(t):
                params, cost = self.adamGD(batch, self.num_classes, self.img_dim, self.img_depth, model, cost)

                t.set_description("Cost: %.2f" % (cost[-1]))

        if indx % self.frequency == 0:
            to_save = [params, cost]
            self.callbacks["SaveModel"](to_save)

        return cost

    def adamGD(self, batch, num_classes, dim, n_c, model, cost):
        lr = self.lr
        beta1 = self.beta1
        beta2 = self.beta2
        """
        Update the parameters through Adam gradient descent.
        """
        global grads
        X = batch[:, 0:-1]  # get batch inputs
        X = X.reshape(len(batch), n_c, dim, dim)
        Y = batch[:, -1]  # get batch labels

        cost_ = 0
        batch_size = len(batch)

        diff_grad = None
        params = model.params()
        for _ in range(3):
            weights = []
            for w_b in params:  # FULL PYTHON LIST
                if w_b is not None:
                    weights.append(np.zeros(w_b.shape))
                else:
                    weights.append(None)
            if diff_grad is None:
                diff_grad = [weights]
            else:
                diff_grad.append(weights)

        # full forward run
        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

            # Collect Gradients for training example
            probs, feed_results = model.full_forward(x)
            loss_value = self.loss(probs, y)  # categorical cross-entropy loss value
            grads_w, grads_b = model.full_backprop(probs, y, feed_results)

            grads = []
            grads.extend(grads_w)
            grads.extend(grads_b)
            for xx in range(model.no_layers() * 2):
                if grads[xx] is not None:
                    diff_grad[0][xx] += grads[xx]
                else:
                    diff_grad[0][xx] = None

            cost_ += loss_value

        # backprop
        for my_i in range(8):
            if diff_grad[0][my_i] is None or diff_grad[1][my_i] is None or diff_grad[2][my_i] is None:
                continue
            diff_grad[1][my_i] = beta1 * diff_grad[1][my_i] + (1 - beta1) * diff_grad[0][my_i] / batch_size  # momentum update
            diff_grad[2][my_i] = beta2 * diff_grad[2][my_i] + (1 - beta2) * (diff_grad[0][my_i] / batch_size) ** 2  # RMSProp update
            # combine momentum and RMSProp to perform update with Adam
            params[my_i] -= lr * diff_grad[1][my_i] / np.sqrt(diff_grad[2][my_i] + 1e-7)

        cost_ = cost_ / batch_size
        cost.append(cost_)

        model.set_params(params)
        return model.params(), cost
```

## Start

For starting the development server of the project maintained in Github you should install all necessary dependencies using:

`$ pip install -r requirements.txt`

Afterwards, you can train the network using the following command:

`$ python3 train_cnn.py 'param.pkl'`

Upon changing any of the source files the application will automatically perform a rerun operation.

## Usage

The proposed ML toolkit can be used in many different scenarios. One can find it easy to configure it as we want by adding any number of layers of choice into a classification or regression task.

## Integrate

After the CNN has finished training, a .pkl file containing the network`s parameters is saved to the directory where the script was run. The trained parameters in the GitHub are under the name ``params.pkl``. The proposed library is versatile because it's easy to add several layers and combine them in order to obtain what we want for any task and to integrate it in a PyTorch application.