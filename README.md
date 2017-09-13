
# Keras Convolutional Neural Network with Python
<p align="center">
  <img width="460" height="300" src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Keras_Logo.jpg">
</p>

Welcome to another tutorial on Keras. This tutorial will be exploring how to build a Convolutional Neural Network model for Object Classification. Let's get straight into it!

```
Note: For learners who are unaware how Convolutional Neural Newtworks work, here are some excellent links on 
the theoretical aspect:
- https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
- https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721
- https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
```
As I did in my previous tutorial I will start by talking about Keras, you can skip it and go straight to the implementatation
if you understand what keras is. 

So what exactly is Keras? Let's put it this way, it makes programming machine learning algorithms much much easier. It simply runs atop Tensorflow/Theano, cutting down on the coding and increasing efficiency. In more technical terms, Keras is a high-level neural network API written in Python. 

## Implementation
### Imports
Every Machine learning heavy Python program starts off by imports. Here are the required imports for CNN:
```python
1    from keras.models import Sequential
2    from keras.layers import Dropout, Dense, Flatten
3    from keras.optimizers import SGD
4    from keras.layers.convolutional import Conv2D, MaxPooling2D
5    from keras.utils import np_utils as u
6    from keras.datasets import cifar10
```
**Line 1-6** all represent functions from Keras:
* Sequential: Creates a linear stack of layers
* Drouput: Ensures minimum overfitting. it does this my selecting random nodes and setting them to 0
* Dense: This essentially is the output layer. It performs the output = activation(dot(input, weights) + bias)
* Flatten: This rolls out our array into 2 dimensions, [numberOfData, features]
* SGD: Stochastic Gradient Descent, this is the optimizer
* Conv2D: This is the convolution layer
* MaxPooling2D: This function performs max pooling 
* np_utils: Some tools to allow us to format our data
* cifar10: This is the dataset we will be using
### Data
Before we can start constructing our CNN we need to load our data:
```python
1    #Lets start by loading the Cifar10 data
2    (X, y), (X_test, y_test) = cifar10.load_data()
```
**Line 2** will download the data and return two tuples, training set and testing set. So we can go ahead and save them into (X,y) and
(X_test, y_test)
```
Note: On a slow computer the cifar10 dataset can take a very long time to download when initiated with Python. 
I recommend downloading manually using website https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
Then head to the .keras/ directory under your username and save the file in the datasets folder with the name
cifar-10-bathces-py.tar.gz. You can then continue normally running the Python program
```
### Formatting the Data
Like always Keras requires a unique format in order to process the data. Therefore we need to format our data
```python
1    #Keep in mind the images are in RGB
2    #So we can normalise the data by diving by 255
3    #The data is in integers therefore we need to convert them to float first
4    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0
5    #Then we convert the y values into one-hot vectors
6    #The cifar10 has only 10 classes, thats is why we specify a one-hot
7    #vector of width/class 10
8    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)
```
**Line 4** Takes our training data and our test data and normalises them. Since they represent colour images, we can divide by 255.
astype converts the integers into floats. 

**Line 8** This is our training labels and test labels. It converts them into one-hot vectors. A one hot vector is an array of 0s and 1s.
Since we have 10 classes our array will be of lenght 10. For example, if our third class is airplanes then the one hot vector for
the airplane data would be [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

### Building the CNN Model
We are done pre-processing our data. Now we can build our CNN model for training!
```python
1     #Now we can go ahead and create our Convolution model
2     model = Sequential()
3     #We want to output 32 features maps. The kernel size is going to be
4     #3x3 and we specify our input shape to be 32x32 with 3 channels
5     #Padding=same means we want the same dimensional output as input
6     #activation specifies the activation function
7     model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
8                  activation='relu'))
9     #20% of the nodes are set to 0
10    model.add(Dropout(0.2))
11    #now we add another convolution layer, again with a 3x3 kernel
12    #This time our padding=valid this means that the output dimension can
13    #take any form
14    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
15    #maxpool with a kernet of 2x2
16    model.add(MaxPooling2D(pool_size=(2, 2)))
17    #In a convolution NN, we neet to flatten our data before we can
18    #input it into the ouput/dense layer
19    model.add(Flatten())
20    #Dense layer with 512 hidden units
21    model.add(Dense(512, activation='relu'))
22    #this time we set 30% of the nodes to 0 to minimize overfitting
23    model.add(Dropout(0.3))
24    #Finally the output dense layer with 10 hidden units corresponding to
25    #our 10 classe
26    model.add(Dense(10, activation='softmax'))
27    #Few simple configurations
28    model.compile(loss='categorical_crossentropy',
29              optimizer=SGD(momentum=0.5, decay=0.0004, metrics=['accuracy']))
```              
**Line 2** This initialises our model. Essentially creates an "empty template" of our model

**Line 7** Our first layer will be a convolution layer. We specify some parameters, 32 represents the number of output feature maps, (3, 3) is the kernel size, our input shape is 32x32 with 3 channels (RGB). If padding is set to same then that means we require the same output spatial dimensions as input. Essentially padding just adds a layer of 0s to make up for the "loss in data". We do this so we can preserve as much information about the early layer as possible.
Finally our activation layer is set to "relu"
```
Note: If your keras json file has not been eddited then your data_format parameter would be set to channel_last. 
This means that the input_shape=(32, 32, 3) is correct. If you're data_format=channel_first then your input_shape changes
to input_shape(3, 32, 32)
```
**Line 10** We drop/set 20% of our nodes to zero to minimize overfitting

**Line 14** We add another convolution layer. This type we do not require input_shape as it has already been specified in the first layer.
Once again, we want 32 output feature maps and computer with 3x3 kernel. This time our padding is set to "valid". This means that we don't want any padding, the output will be whatever it will be. Once again, our activation function is "relu"

**Line 16** This performs maxpooling with a kernel size of 2x2

**Line 19** Before we could put our data through our output/dense layer we need to flatten our data we have only 2 dimensions.
The 2 dimensions being [full batch, features]. The length of the features will be height\*width of the data produced after te convolution layer\*32 being the number of feature maps

**Line 21** We then put our data through the Dense layer with 512 hidden units and the activation function relu".

**Line 23** Then we perform the droupout function on 30% of the CNN nodes to prevent overfitting

**Line 26** Finally we put it through another Dense layer this time with 10 unit outputs (representing the 10 different classes) using the "softmax" activation function

We can now move onto the configurations

**Line 28** We compile our model with the categorical_crossenrtopy loss and the SGD optimizer.
The SGD optimizer has several parameters. The momentum parameter is essentially used for a faster convergence of the loss function. Sometimes gradient descent oscillates when gradients are too steep, this will also cause slow weight updates but if you add a fraction of the previous update to the current, the convergence is faster. In order to have a high momentum term you must decrease the learning rate or it would cause error. Decay represent the learning rate decay after every update. This helps in reaching convergence faster as well.
You need to have metrics enabled in order to get accuracy scores for evaluation. Metric also shows you the accuracy while training.
### Training and Saving
We've built the model, done our configuration therefore we can now start training!
```python
1    #Run the algorithm!
2    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
3          batch_size=512)
4    #Save the weights to use for later
5    model.save_weights("cifar10.hdf5")
6    #Finally print the accuracy of our model!
7    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))
```
**Line 2** This line runs our model. Out of 50000 we take a consecutive 512 batches and run them 25 times each. (Batch size = 512, epoch = 25)

**Line 5** We can save our weights if we want to.

**Line 7** Finally, we display our accuracy after evaluating our test set.
## Conclusion and Development
And we are done with our very own CNN! Here are additional features and other ways you can improve your CNN:
- For prediction you could simple use the model.predict_classes(X[0:1]) to classify your image (To see if it works properly)
- When using dropout the weights can be suddenly put into a very bad situation causing them to fluctuate etc. So we can have another parameter in our Dense and Conv2D layers, kernel_constraint. We use this to set constraints on our weights, e.g. non-negativity. we do this by kernel_constraint=maxnorm(desiredValue). The maxnorm constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value. (from keras.constraints import maxnorm)
- In our compilation line we could have added another paramter called nestrov momentum. Nestrov=false is dafult but can be set to true to make converging faster. Here is more information on nestrov http://cs231n.github.io/neural-networks-3/#sgd. (nestrov=True)
- For large datasets and having parameters like momentum active, having low batch size can cause errors. Anything lower than a 512 batch size would cause a warning such as method on batch end is slow compared to batch update. To fix this, simply increase batch size.
- To further develop your CNN you could have more layers, a deeper CNN which would allow for a higher accuracy etc. You could train for more epochs.

Thats all! For any questions or bugs do not hesitate to contact me!

  
