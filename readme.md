# Udemy Course: Tensorflow 2.0: Deep Learning and Artificial Intelligence

* [Course Link](https://www.udemy.com/course/deep-learning-tensorflow-2/)

## Section 1 

## Section 2: Google Colab

### Lecture 6. Uploading your own data to Google Colab

* The Codelab (Jupyter Notebook) way
	* we can run unix commands in colab notebooks useing the ! infront
	* we can use `!wget http://<link to dataset>` to upload a dataset to colab
	* then with `!ls` we can locate it
	* to check if the data file has a header row `!head <datafile>`
	* we then import pandas and use it to create a dataset from the csv
```
import pandas as pd
df = pd.read_csv('arrhythmia.data',header=None)
```
	* we extract the first 6 features and give them header
```
data = df[[0,1,2,3,4,5]]
data.columns = ['age','sex','height','weight','QRS duration','P-R interval']
```
	* we printout a plot with feature histograms
```
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15,15]
data.hist();
```
	* we can easily do a dataset scatterplot with pandas
```
from pandas.plotting import scatter_matrix
scatter_matrix(data);
```
* The tf.Keras way (using tensorflow)
	* install and import tf2
```
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
```
	* set the url and use keras to get it from
```
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
tf.keras.utils.get_file('auto_mpg.data',url)
```
	* keras default folder for datasets is `/root/.keras/datasets/`
	* after checking the file with `!head` we make a dataset out of it
```
df = pd.read_csv('/root/.keras/datasets/auto_mpg.data',header=None,delim_whitespace=True)
df.head()
```
* The Manual Upload way using Colab
	* with the code below we create an upload button  in notebook to upload local files 
```
from google.colab import files
uploaded = files.upload()
```
	* the file is uploaded as a a python dictionary
	* the file now resided in colab working dir
	* many times we want to test various algos.
	* we then want to write data loading or prep code in one notebook
	* then somehow use that code in our algo notebooks to keep project DRY
	* we upload py script using colab, we import a method from the script and run it
```
from google.colab import files
uploaded = files.upload() #fake_util.py
from fake_util import MY_function
my_function()
```
* Access files from GDrive in Colab
	* we import and mount gdrive (/content is our pwd)
```
from google.colab import drive
drive.mount('/content/gdrive')
```
	* we can view our drive `!ls gdrive/'My Drive'`

### Lecture 7. Where can I learn about Numpy, Scipy, Matplotlib, Pandas, and
Scikit-Learn?

* we have done in it JPortillas courses
* tutor offers a free course [Numpy Stack](https://www.udemy.com/course/deep-learning-prerequisites-the-numpy-stack-in-python/)

## Section 3: Machine Learning and Neurons

### Lecture 8. What is Machine Learning?

* ML boils down to a geometry problem
* Linear Regression is line or curve fitting. SO some say its a Glorified curve-fitting
* Linear Regression becomes more difficult for humans as we add features or dimensions or planes or even hyperplanes
* Regression becomes more difficult for humans  when problems are not linear
* classification and regression are examples of Supervised learning
* in regression we try to make the line as close as possible to data points
* in classification we try to use aline that separates points in different classes correclty
* ML cares about data. their meaning makes no difference

### Lecture 9. Code Preparation (Classification Theory)

* Usual ML Steps
	* Load in data
	* instantiate teh model
	* train (fit) the model
	* evaluate results
* the difference of TF and Skikit-Learn is that in TF we build the model. in Scikit-Learning we get it ready
* In TF we care about the Architecture of the model (go from input to prediction)
* We care bout the internals of training a model
	* Build the cost/loss function
	* Do Gradient Descent to minimize cost
* We do all this using TF API. all this are portable in other Languages like JS.
* Our classification model solves the y=mx+b problem
* for 2D classification we arrange the equation as 'w1x1+w2x2+b=0'
* if the previous equation is formed as 'w1x1+w2x2+b=a' and a>=0 we predict class 1 if a<0 class 0
* mathematicaly this can be expressed as the step function 'y=u(a)'
* in DL we preffer smoother functions (sigmoid) 'y=sigma(a)'
* we normally interpret this as 'the probability that y=1 given x'
* to make the prediction we round. e.g. if p(y=1|x) >=50% predict 1 else 0
* the S shaped curve is called sigmoid 'p(y-1|x)=sigma(w1x1+w2x2+b)
* the sigmoid is called logistic regression as sigmoid = logistic function
* if we have >2 inputs
	* p(y=1|x) = sigma(wTx+b) = sigma(S[d=1->D]wdxd+b
	* we use the dot product of W and X matrices
* In TF2 we need to find inthe API the function that does the math for our math problem
* the wTx+b is performed in a TF layer called Dense `tf.keras.layers.Dense(output_size)`
* to solve the problem with TF2 we need 2 layers, an input (with size equal to our feats) and a dense implementing  the sigmoid with output 1 (the class)
```
model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(D,)),
	tf.keras.layers.Dense(1,activation='sigmoid')
```
* next step is training - fitting the model.
	* we need to define a cost/loss function to be used
	* we need to optimize using gradiaent descent
	* GD has many flavors. 'adam' the most typical in DL
	* we also need to provide a metric to measure the model
	* accuracy (correct/total) is the most used in classification
```
model.compile(optimizer="adam",
	loss="binary_crossentropy",
metrics=["accuracy"])
```
* after we compile the model we need to fit it(train) with data
	* train is a 2 step repetitive process
	* train with train data
	* test with test data
	* repeat with new params for number of epochs to get  better fit
```
r = model.fit(X_train,y_train,
	validation_data=(X_test,y_test),
	epochs=100)
```
* finally we plot metrics to evaluate results visualy
```
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
```

### Lecture 10. Classification Notebook

* a breast cancer classification problem using TF2
```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Load in the dataset from Sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```
* data is of type `sklearn.utils.Bunch` a dictionary like object where keys act like attributes
* `data.data` contains the featues table and `data.target` the classification results
```
#split the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.33)
N,D = X_train.shape
# scale the data to 0-1 so that sigmoid can work (model)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
* note that for test data we dont fit and transform so as to avoid cheating the model
```
# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(D,)),
  tf.keras.layers.Dense(1,activation="sigmoid")                              
])
model.compile(optimizer="adam",
  loss="binary_crossentropy",
  metrics=["accuracy"])
# Train the model
r = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)
# Evaluate the model . evaluate() returns loss and accuracy
print("Train score: ",model.evaluate(X_train,y_train))
print("Test score: ",model.evaluate(X_test,y_test))
# Plot what was returned by model.fit()
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label="loss")
plt.plot(r.history['val_loss'],label="val_loss")
```
* val_loss = validation loss (test data)

### Lecture 11. Code Preparation (Regression Theory)

* loading input data usually involves preperocessing
* instnatiate teh model
* train the model
* evaluate the model
* when we do regression we dont need activation. there is no default value
* we use a Dense layer without activation `tf.keras.Dense(1)` 1 stands for one output
* in regression the most basic optimizer is SGD (stochastic gradient descent)
* in regression cost method is usually MSE (mean squared error)  MSE = 1/N * S[i=1->N](yi-yi^)2
```
moel.compile(optimizer=tf.keras.optimizers.SGD(0.001,0.9),loss='mse'
```
* to imptrove fitting and cost fluctuation we decrease learning rate as epochs progress (learning rate scheduling)
* accuracy is not applicable in regression, we use R^2
* we ll try to proove Moores Law. this low is about exponential growth. this is non linear
* Exponential Growth C=A0r^t
	* C = Count (output val)
	* A0 = initial Value of C when t=0
	* t = time(input var)
	* r = rate of growth
* Using log we can turn it linear: logC = logr*t+logA0

### Lecture 12. Regression Notebook

* we will make amodel to prove moores law
* note that TF and keras expect 2D arrays as input
* se that we can pass in callbacks in the train process to be called in each epoch
```
# load in the data
data = pd.read_csv('moore.csv',header=None).values
data[0]
X = data[:,0].reshape(-1,1) # make it a 2D array of size NxD where D=1
Y = data[:,1]
# Plot the data. (its exponential?)
plt.scatter(X,Y)
# since we want a linear model. we use the log
Y = np.log(Y)
plt.scatter(X,Y)
# We also center the X data so the values are not too large
# we could scale it too but then we'd ave to reverse the transformation later
X = X - X.mean()
# We create the Tensorflow model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)                                    
])
model.compile(optimizer=tf.keras.optimizers.SGD(0.001,0.9),loss='mse')
# 0.001 is learnign rate, 0.9 is momentum
# model.compile(optimizer='adam',loss='mse')
# Learning Rate Scheduler
def schedule(epoch, lr):
  if(epoch >=50):
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
# Train the model
r = model.fit(X,Y,epochs=200,callbacks=[scheduler])
# Plot the Loss
plt.plot(r.history['loss'],label='loss')
# Get the Slope of the line
# THe slope of the line is related to the doubling rate of transistor count
print(model.layers) # note: there is only 1 layer here. Input layer does not count
print(model.layers[0].get_weights())
# weight (W) is a 2D array and bias,scaler (B) is a 1D array
# the slope of the line is
a = model.layers[0].get_weights()[0][0,0]
```
* W.shape is (D,M)
* b.shape 9s (M,)
* D = input size, M = Output size
* Based on our original exponential grownth function C +Aor^t and its conversion to liear with log (we saw it above)
	* y = logC
	* a = logr
	* x=t
	* b=logA0
* r = e^a

### Lecture 13. The Neuron

* Logistic Regression can be considered as a Neuron
* Linear and Logistic Regression both use the same line equation y=mx+b
* m is the weight for input x.
* in real problems we have multiple feats so we end up to a Linear Equation y=wix1+w2x2+b
* this ends up to solving a Linear System. Thats a Neuron
* a biological Neuron has imputs that take signals from other neurons with the dendrites.
* sensors lead to electrical signals sent to the brain
* the neuron decides to pass or not the aggregated signal
* some connections are strong and some weak. some are ecitatory (+) and some inhibitory (-)
* when the aggregated signals exceed a threshold (~ 20mV)the neuron fires a strong signal downstream
* this happens in the trigger zone (Hillcock) and generates an Action Potential (~100mV)
* the signal travels through the myelin sheath. the neuron after firing returns to rest
* the post synaptic potentials received by the dendrits have a amplitude of ~10mV and a raise time of ~10ms.
* the action potential generated by the neuron has a rapid raise time ~1ms reaching +30mV after the threshold potential is reached ~ -40mV.
* then the neuronreturns to resting potential  (-70mV) after the refactory period
* this working principle is called All or Nothing much like the Binary Classification
* The Neuron fires or not
* in a Computational Neuron the threshold is -b when wTx + B >0 the neuron fires (prediction 1)
* this is the dot product or inner product or the possibility of y=1 whatever the x

### Lecture 14. How does a model "learn"?

* The goal of Linear Regression is to do Line Fit
* we need an error function to find the best fit. eg with minimum MSE (mean squared error)
* MSE = 1/N S[i=1->N](yi-^yi)^2 (^yi=mxi+b) ^yi is the prediction
* to Minimize MSE (cost) we use calculus. finding where the derivative (slope) is zero equals finding a min /max
* derivative is the slope. so we need to find where m is 0
* usually we have multiple parameters so we need the partial derivatives of each
* the multidimensional equivalent of a derivative is gradient
* to solve for model params (wights) we find the gradient and set it to 0
* TF2 saves the day. we dont have to calculate gradients. It uses automatic differentiation
*	* TF automatically finds the gradient of loss
	* TF uses the gradients to train our model
* if we just have to solve an equation. why training is iterative?? why check for convergence?
* most of the time we cannot just solve the for gradient = 0. only for linear regression we can
* For Logistic Regression and other problems we need to do Gradient Descent.
	* in keras this is done in the fit method.
	* w,b are randomly initialized. for each epoch: w = w - htta * GradwJ, b = b - htta*GradbJ
	* htta is the learning rate.
* Learing Rate is a hyperparameter. it is mostly optimized by trial and error.
	* we check the loss per iteration
	* normally we choos epowers of 10: 0.1, 0.001, 0.0001 etc
	* a good learning rate leads to an reverced exponential curve
	* a large learnign rate will lead to noize or bounds around convergence

### Lecture 15. Making Predictions

* using a trained model for predictions is what makes it useful
* we see the process for linear classification with test data
```
# Make Predictions
P = model.predict(X_test)
print(P) # they are outputs of the sigmoid, interpretted as probabilities (py=1|x)
# Round to get the actual Predictions
# Note! has to be flattened since the targtets are size(N,) while the predictions ar3e size (N,1)
import numpy as np
P = np.round(P).flatten()
print(P)
 Calculate the accuracy, compare it to evaluate() output
print('Manuallty calculated accuracy:', np.mean(P==y_test))
print('Evaluate output:', model.evaluate(X_test, y_test))
```
* for linear regression
```
# Make sure the line fits out data
Yhat = model.predict(X).flatten()
plt.scatter(X,Y)
plt.plot(X, Yhat)
# Manual Calculation
# get the weights
w,b = model.layers[0].get_weights()
# Reshape X because we flatteneed it earlier
X = X.reshape(-1,1)
# dimension : (N x 1) x (1 x 1) + (1) -> (N x 1)
Yhat2 = (X.dot(w)+b).flatten()
# dont use == for floating points
np.allclose(Yhat,Yhat2)
```

### Lecture 16. Saving and Loading a Model

```
# Let's now save our model to a file
model.save('linearclassifier.h5')
# Let's load the model and confirm that it still works
# note! there is a bug in Keras where load/save only works if you DONT use the Input() layer explicitly
# make sure you define the model with only Dense(1, Input_shape=(D,))
model = tf.keras.models.load_model('linearclassifier.h5')
print(model.layers)
model.evaluate(X_test,y_test)
```
* to get the model from colab: CLICK right arrow on left top corner => Files => Download it
* or use python script
```
# Download the file - requires Google Chrome
from google.colab import files
files.download('linearclassifier.h5')
```
## Section 4: Feedforward Artificial Neural Networks

### Lecture 17. Artificial Neural Networks Section Introduction

* ANNs (Artificial Neuron Networks)
* Feedforward is the most basic type of NNs
* Neurons in our Brain communicate forming a Network. it makes who we are
* NNs are what we use for AI. trying to build an artificial brain
* The Human brain has an amount of neurons way mopre than the most complex NN ever built
* NNs are a series of layers connected to each other.
* in the brain neurons can connect back to an earlier neuros to create a recurrent connection
* with the feedforward NNs we will do multiclass classification

### Lecture 18. Forward Propagation

* to make a prediction we need to keep forward direction in an NN (input -> output)
* different neurons find different features
* for computational model we assume neurons are same
* same inputs can be fed to multiple neurons, each calculating something different
* neuron inone layer can act as inputs to another layer
* for a single neuron model we expressed it sigma(wTx+b)
* for multiple neurons in a layer: zj = sigma(wjTx+bj),for j=1..M (M neurons in the layer)
* if we consider z as a vector we can express it z=sigma(WTx+b)
	* z is vector of size M (column vector of size Mx1
	* x is a a vector of size D (column vector of size Dx1
	* W is a matrix of size DxM (transposed MxD)
	* b is a vector of size M
	* sigma() is an element wise operation
* for each layer we have such an equation. for an L layer network we have 
	* p*y=1|x) = sigma(W^(L)T*z(L-1) +b(L)) for binary classification
	* for linear regression p*y=1|x) = W^(L)T*z(L-1) +b(L) no sigmoid
* Each neural network layer is a 'feature transformation
* Each Layer learns increasingly complex features (e.g facial recognition)

### Lecture 19. The Geometrical Picture

* The neuron is interpretable
	* large weight important feat
	* small weights non-important feat
* the linear model (neuron) is not very expressive
* as we said to mak4e the problem more complicated we can
	* add more dimensions (feats) to the linear problem
	* make the pattern non linear (non linear separable)
* the neuron model expresses a line or multidimensional plane
* there is no setting of w and b to get a curve
* One way to solve non-linear problems is feature engineering
	* say salary is a quadratic function of years of experience. Yhat = ax^2 + bx + c
	* this is still linear regression. x1 = x x2=x^2 => yhat = w1x1 + w2x2 + b
* The problem with Feature engineering is there are too many possibilities
* Another way to solve non-linear problems is by repeating a single neuron
	* each neuron computes a different non-linear feat of the input
	* its non-linear because of the sigmoid
* a linear boundary in geometry takes the form of wTx+b (single neuron)
* a 2-layer neural network boundary takes the form of
	* W^(2)Tsigma(W^(1)Tx+b^(1))+b^(2)
	* we cannot simplify this to alinear form
* if we had no sigmoid we could reduce it to linear form: W'Tx+b'
	* W' = W^(2)TW^(1)T
	* b'= W^(2)Tb^(1)+b^(2)
* With neurons we get Automated feature engineering. 
* Ws and bs are random and found iteratively using gradient descent
* No actual domain knowledge is needed.
* This is the power of DL. it allows non-experts to build very good models
* Checkout Tensorflow Playground in web

### Lecture 20. Activation Functions

* we have seen only the sigmoid activation function: sigma(a) = 1/(1+exp(-a))
	* it maps vals to 0..1 
	* it mimics the biological neuron
	* it makes our NN decision boundary non linear
* Sigmoid is problematic and not used very often
	* we want to have all our inputs standardized. centered around 0 and all in same range
	* sigmoid is problematic. its output is centered on 0.5 and not 0
	* neurons must be uniform. the ones output is the nexts input.
* Hyperbolic Tangent (tanh) solves this: tanh(a) = (exp(2a) - 1)/(exp(2a)+1)
	* same shape as sigmoid but goes from -1 to +1
	* tanh is still problematic
* The problem with both is the Vanishing gradient problem
	* we use gradient descen t to train the model
	* this requeires finding the gradients of w parameters
	* the deeper the network the more terms have to be multiplied by the gradient due to the chain rule of calculus
	* a neural network is a chain of composite functions
	* its output is essential sigmoid(sigmoid(sigmoid...)))
	* we end up multiplying the derivative of the simoid over and over
	* the derivative of the sigmoid is like a very flat bell shaped curve with low peak ~ 0.25
	* it is essentially 0 in its most part. if we multiply small numbers we get even smaller
	* the further we go from the output the less contribution from layers to the output
	* the gradient vanishes the furtheOld ner back we go to the network
	* layers further back are not trained at all
* One old-shool solution from Geoff Hinton was the 'greedy layer-wise pretraining'
	* train each layer greedly one after the other
* other old school solutions: RBMs(restricted boltzman machines, Deep Boltzman machines)
* The solution was simple. dont use activation functions with vanishing gradients
* use the ReLU(rectifier linear unit) like the zener diode output: R(z) = max(0,z)
	* in ReLU the gradient on the less side <0 is already 0 (vanished)
	* this is a phenomenon of dead neurons. neurons that their input is small are not trained
* ReLU works. rights side is enough to do the job
* Solutions to solve the Dead Neuron problem: 
	* Leaky ReLU (LReLU) => small positive slope for negative inputs. still non linear
	* ELU (exponential linear unit) f(x) = x > 0 ? x : a(exp(x) -1)
	* Softplus: f(x) = log(1+ exp(x)) like a smooth curved ReLU.
* Authors claim ELU speeds up learning and imporves accuracy. negative values possible. mean close to 0 (unlike ReLU)
* Both Softplus and ELU have vanishing gradients on left. but so does ReLU and works
* Softplus and ReLU cant be centered around 0 (range 0 +inf)
* it does not matter
* the default in most models is ReLU. still if our results are not good enough we should experiment
* use the computer and experiment
* ReLU might be even more biological plausible than sigmoid. 
* action potentials are same regardless of stimuli
* their frequenry changes with stimuli intensity (voltage in receptors)
* the RELU is like encoding the action potential frequency.
	* zero minimum (no action potential)
	* as the input increases in value also the output increases (in frequency) having larger effect downstream
* Relationship between action potential frequency and stimulus intensity is non linear
* it can be modeled as log() or root functions. (like decibel scale)
* what we model is stimulus intensity vs action potential spikes /sec
* The most recent and accurate activation method of Biological neurons is the BRU (Bionodal Root Unit)
* its math equation is: f(z) = z >=0 ? (r^2z+1)^1/r - 1/r : exp(rz) -1/r
* Not yet adopted by the DL community

### Lecture 21. Multiclass Classification

* for output layer sigmoid is the right choice for binary classification
* for hideen layers choose ReLU
* For Multiclass classification suppose we have reached the Final Layer
	* a^(L) = W^(L)Tz^(L-1)+b^(L)
	* a^(L) is a vector of size K(for a K-Class classifier)
	* how do we turn this vector into a set of probabilities for each of the K classes?
* Requirements for a Probaility
	* we need a probaility distribution over K values p(y-k|x) >=0 and <=1
	* probabilities must sum to 1 S[k=1->K]p(y=k|x) = 1
* Softmax meets both requirements
	* exp is always positive
	* the denominatoor is the sum of all possible values of the numerator
* In tensorflow `tf.keras.Dense(K,activation='softmax')`
* Softmax is not meant for hidden layers unlike ReLU/sigmoid/tanh
* So to recap:
	* Regression: activation? none/Identity
	* Binary Classification: activation? sigmoid
	* Multiclass Classification: activation? Softmax

### Lecture 22. How to Represent Images

* NNs are most powerful with unstructured data: images, sound,text
* we ll see how to classify images
* we represent iages as matrix. 
* each element has 3 nums, is the 3 colors content that makes the pixel (RGB)
* so a color image is actually a 3D tensor HxWxC C=3 (RGB channels)
* color in nature is continuous (analog) has infinite vals
* 8bits precision is good enough to quantify colors. 16mil xolora
* raw images can be big. we compress them for storage (JPEG)
* Grayscale Images are 2D. they have 1 channel for color
* matplotlib imshow plots images. default cmap(colormap) is heatmap
* to pass in a NN we scale the image 8bit values from ints 8-255 to 0..1 vals (floats)
* these vals although not centered around 0 are OK. they express probability
* VGG network is famous in Computer Vision. 
	* (image classivication.object detection et al). 
	* it won ImageNet contest
	* VGG does not scale. it subtracts the mean across each channel. 
	* values are centered around 0 with range 0-255
	* if we use tfs built-in VGG model. we need to preprocess the image with `tf.keras,applications.vgg16.preprocess_input`
* NNs expect an input X of shape NxD (N=samples,D=feats)
* An image has dimensions of HxWxC. it does not have D feats
* A full dataset of images is a 4D tensor of dimensions NxHxWxC
* To convert an image to a Feature Vector we flatten it out.with Flatten(). the image becomes NxD

### Lecture 23. Code Preparation (ANN)

* Steps we ll take:
	* Load the data (MNIST dataset)
	* Build the model (Multiple Dense layers, Output Layer. multiclass lg regression (softmax) 
	* Train the model
	* Evaluate the model
	* Make Predictions
* Load Data
	* data already in `tf.keras.datasets`
	* images 28x28 = 784 pixels or feats flattened out
	* `(x_train,y_train),(x_test,y_test) = mnist.load.data()`
	* Pixels vals will be scaled to 0..1
	* tf.keras will do the flattening for us. we dont have to do it
* Instantiate the model
```
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(128,activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10,activation='softmax')
])
```
* dropout is a way to reduce bias in our model
* train the model
```
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
model.fit(x_train,y_train)
```
* Cross-Entropy Loss
	* we have K output probabilities for the K classes
	* one=hot encoding the target means 9 is repressented as yOH=[0,0,0,0,0,0,0,0,0,1]
	* Loss = -S[k=1->K]ykOHlog(yhatk) where yhatk=p(y=k|x)
	* if our prediction is perfect y=1 => Loss = -1*log1=0
	* if our prediction is totally wrong y=0 => Loss = -1*log0 = inf
* Why the cross -entropy loss is sparce???
	* to calculate cross-entropy both the one hot encoded targets and output probs must be arrays of K
	* this is not optimal as target is just an int
	* a on-hot array is sparce (9/10) vals are 0
	* for all 0s we get 0, its redundant
* Solution? Sparce-categorical -cross-entropy
	* we consider a non -one-hot encoded target y=k*
	* we need only the log of the k*th entry of the prediction
	* we can index the prediction reducing computation by factor K
	* it works because by default lables are 0based ints
	* Loss = -log(yhat[k*])
* Last step: evaluat ethe model and predict
```
model.evaluate(X,Y)
model.predict(X)
```

### Lecture 24. ANN for Image Classification

* we write down the code in Colabs
```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Load in the data
mnist =tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test  = x_train / 255.0, x_test / 255.0
print('x_train.shape:',x_train.shape)
# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10,activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
# Train the model
r = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)
# Plot the loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
# Evaluate the model
print(model.evaluate(x_test,y_test))
```
* we see a common problem in ML. the more we train the better we get with the train dataset without gain on the test set.
* its called overfitting
* we copy a useful script to printout confusion matrix
```
# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='Confustion matrix',cmap=plt.cm.Blues):
  ###
  # This function prints and plots the confustion matrix
  # Normalization can be applied by setting 'normalize=True' 
  ###
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confustion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()  / 2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
      plt.text(j,i, format(cm[i,j],fmt),
        horizontalalignment='center',
        color='white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test,p_test)
plot_confusion_matrix(cm,list(range(10)))

# Do the results make sense?
# its easy to confuse 9<->4, 9<->7, 2<->7 etc
```
* show examples of misclassified data
```
# Show some misclassified examples

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i],cmap='gray')
plt.title('True label: %s Predicted: %s' % (y_test[i],p_test[i]));
```

### Lecture 25. ANN for Regression

* we cp a notebook in colab to do the job using synthetic data we will create
```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Other Imports
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Make the dataset
N = 1000
X = np.random.random((N,2)) * 6 - 3 # uniform distribution between (-3,+3)
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1]) # this implements the function y = cos(2xi) + cos(3x2)
# Plot it
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
# plt.show()
# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(2,),activation='relu'),
  tf.keras.layers.Dense(1)
])
# Compile and fit
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt,loss='mse')
r = model.fit(X,Y,epochs=100)
# Plot the loss
plt.plot(r.history['loss'],label='loss')
# plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
# surface plot
line = np.linspace(-3,3,50)
xx,yy = np.meshgrid(line,line)
Xgrid = np.vstack((xx.flatten(),yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0],Xgrid[:,1],Yhat,linewidth=0.2,antialiased=True)
# plt.show()
```
* we see that fixed learnign rate is suboptimal approach. we need to schedule it
* its impressive how we aproximate so close a cosine function without applying any method
* to plot 3d
	* we create the meshgrid
	* we geenrate numbers with linspace
	* we call meshgrid on the points
	* we flatten the array for ML
	* we do prediction of Y
	* we pot the surface on the 3d axis system
```
# Can it extrapolate?
# Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
# surface plot
line = np.linspace(-5,5,50)
xx,yy = np.meshgrid(line,line)
Xgrid = np.vstack((xx.flatten(),yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0],Xgrid[:,1],Yhat,linewidth=0.2,antialiased=True)
# plt.show()
```
* we see a problem with extrapolation
* we need a periodic activation method in our NN to fix that

## Section 5: Convolutional Neural Networks

### Lecture 26. What is Convolution? (part 1)

* 