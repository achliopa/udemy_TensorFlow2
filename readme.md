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

### Lecture 7. Where can I learn about Numpy, Scipy, Matplotlib, Pandas, andScikit-Learn?

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

* Convolution is the result of passing a filter (kernel) over a data series
* in DSP is like passing a step function over a sample stream
* In image prcessign is passing a kernel over an image
* the math symbol of comvolution is the star (asterisc)
* we can think it as feature transformation
* blurring an image is by applying a gausian filter over it
* the result is calculated by the inner product of the kernel with the affected area
* If input length = N kernel length = K , output length = N-K+1
* convolution pseudocode
```
input_image,kernel
output_height = input_height -kernel_height +1
output_width = input_width -kernel_width +1
output_image = np.zeros((output_height, output_width)
for i in range(0,output_height):
	for j in range(0,output_width):
		for ii in range(0,kernel_height):
			for jj in range(0,kernel_width):
				output_image[i,j]+=input_image[i+ii,j+jj]*kernel[ii,jj]		
```
* images by default are not square, 
* kernels are almost always square by convention
* Convolution equation: (A * w)ij=S[i'=1->K]S[j'=1->K]A(i+i',j+j')w(i',j')
* TF does the calculation for us
* More formal equation: Input=Y,Filter=X,Output=Z
	* Z(i,j)=S[m=1->M]S[n=1->N]X(m,n)Y(i-m,j-n)
* As we use GRadient Descent it does not matter if in our filters convolution we use + or -
* Scipy has a convolution method `from scipy.signal import convolve2d`
	* the result is different as it does proper convolution with - and not the DL version with +
* to produse the DL version with scipy we need to flip the filter
	* `convolve2d(A,np.fliplr(np.flipud(w)), mode='valid')`
* What we actually do in DL is called 'cross-correlation'
* mode= valid is usedbecause the moveement of the filter is always bvounded by the limits of the image. 
* the output is always smaller thant the input
* if we want to have same size we can use padding (zeros)
* even with single padding we lose context. 
* we can have full padding : output_lenght = N+K-1 to dont miss out info
* The 3 Modes of convolution
	* Valid: Output_size=N-K+1 : Typical usage
	* Same: Output_size=N : Typical usage (no padding)
	* Full: Output_size=N+K-1 : ATypical usage

### Lecture 27. What is Convolution? (part 2)

* Vectorization of operations is desirable because numpy vectors are way faster than for loops
* we can use it to simplify convolution using inner product to look like: aTb = S[i=1->N]aibi =|a||b|cos(theta ab)
* Dot product is called cossine similarity or cosine distance
* for angle=0 cosine is 1 for angle =90deg cosine is 0 for angle=180deg cosine is -1
* cosine is an expression of similarity in vectors
* Pearson Correlation is almost same as cosine but with mean subtraction
* Pab = S[i=1->N](ai-a_)(bi-b_)/((sqrt(S[i=1->N](ai-a_)^2)(sqrt(S[i=1->N](bi-b_)^2))
* Dot product is a correlation measure
	* high positive correlation-> dot product large and positive
	* high negative correlation-> dot product large and negative
* Convolution  is a pattern finder

### Lecture 28. What is Convolution? (part 3)

* Equivalence of convolution in matrix multiplication
* 1D convolution is same as 2D without the 2nd index bi = S[i'=1->K]ai+i'wi'
* by repeating same filter again and again inside a matrix we can do convolution doing matrix multiplication
* problem is it takes too much space.
* sometimes instead of doing matric multiplication we can replace it with convolution
* in a = WTx what if instead of a full weight matrix W we used 2 weights repeated
* we could do convolution and save on Ram and time
* typical input vectors in modern CNNs are sized at scale of 10^5 feats.
* matrix multiplication can explode in complexity
* a dense layer would treat a feat in a different position of an image as different

### Lecture 29. Convolution on Color Images

* Color images are 3D objects
* in 3D is like convoluting a cube through a box of same depth.
* we actually add an index and a sumamtion in comvolution but we now the K = 3
* Input is HxWx3, Kernel is KxKx3, output image (H-K+1)x(W-K+1) 3rd dimension vanishes.
* Shape of bias term
	* in a dense layer, if WTx is avector of size M , b is also of size M
	* in a conv layer b does not have the same shape as W*x 
	* technically this is not allowed by rules of matrix arithmentic
	* the rules of broadcasting (in Numpy code) allow it
	* if W*x has shape of HxWxC2 b is a vector of size C2 (one scalar per feature map)
* By convolution vs matrix multiplication we save massively
* since convolution is a part of the neural network layer. its easy to think how filters will be found
* conv is a pattern finder/shared-param matrix multiplication / feature transformer
* W fill be found  through training with gradient descent.

### Lecture 30. CNN Architecture

* Modern CNNs originate from LeNet (Yann LeCun)
* A typical Architecture is Conv->Pool->Conv->Pool->Conv->Pool->Dense->Dense->Output
* Stage 1 of Conv->Pool is like a Feature transformer to feed feats in the ANN of stage 2
* Stage 2 is a non-linear classifier
* Pooling is another name for Downsampling (making a smaller image out of a bigger one)
* if a poolsize is 2 an 100x100image becomes 50x50 (Downsample by 2)
* there are 2 types of pooling. max pooling and avg pooling. max pooling is more common and faster
* Why we do it? a) we hav eless data to process downstream, b) we dont care where the feat ccured, just that it did
* We call this 'Translational Invariance'. we humans do the same
* the input for pooling is the pattern finder matrix. max pooling preserves context. if it was found the info will persist downstream
* Pooling boxes can overlap (this is called stride)stride controls pixel shifting between subling next subsampling operation
* after each conv/pool step image shrinks but filter size stays the same
* initially filter looks for tiny patterns (lines,curves) then as image shrinks it looks for increasingly complex features
* while we lose spatial info in the proces we get feature info as we get more feature maps
* CNN standard conventions to start with
	* small filters relative to image: 3x3, 5x5, 7x7
	* repeat convolution->pooling pattern
	* increase # of feature maps 32->64->128->128
	* read papers on the subject
* we can avoid pooling by doing strided convolution
* neighbor pixels are typically highly correlated
* when image comes out of 1st stage is 3d we need to reshape it before feeding it to 2nd stage using Flatten() layer
* Global Max pooling layer can fix the problem of differnet sized images (eg from internet)
* we do global max pooling before we feed the FNN 
* if our images are too small we get error when we apply global max pooling

### Lecture 31. CNN Code Preparation

* Step 1: Load in the data (Fashion MNIST and CIFAR-10)
	* Fashion MNIST (28x28 grayscale clothes) 10 classes
	* CIFAR-10 32x32x3 color objects 10classes
* Step 2: Build the model
	* conv->pool->Conv-->pool->FNN
	* Learn Functional API
* Step 3: Train the model
* Step 4: Evaluate model
* Step 3: Make Predictions
* Later on: Data Augmentation. produce more images from original dataset (rotate etc, mirror)
* Loading the Data:
	* CIFAR-10: `tf.keras.datasets.cifar10.load)data()`
	* Fashion MNIST: `tf.keras.datasets.fashion_mnist.load_data()`
	* `(x_train,y_train),(x_test,y_test)=load_data()`
* CNN expects NxHxWxC input. Fashion MNIST is Nx28x28. we need to reshape it to Nx28x28x1.
* CFAR-10 labels are Nx1 we need to flatten() them
* Build the model
* We will use Keras functional API
* Keras is an API spec independent of TF
* Keras inventor works for Google who created TF. so its is built in
* if we use Keras library separate from backend we might have version mismatch
* Keras the TF module works by default 
* Keras Standard API

```
model = Sequential([
	Input(shape=(D,)),
	Dense(128, activation='relu'),
	Dense(K,activation='softmax')
])
...
model.fit()
...
```

* Keras Functional API

```
i = Input(shape=(D,))
x = Dense(128, activation='relu)(i)
x = Dense(K,activation='softmax')(x)
model = Model(i,x)
...
model.fit()
...
```

* we see that keras objects work as functions
* Keras Functioanl API: looks cleaner, easier to create model branches, models with multiple inputs outputs
* eg we can feed the input in 2 layers in parallel
* CNN with functional API

```
i = Input(shape=x_train[0].shape)
x = Conv2d(32, (3,3), strides=2, activation='relu')(i)
x = Conv2d(64, (3,3), strides=2, activation='relu')(x)
x = Conv2d(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(K, activation='softmax')(x)
model = Model(i,x)
```

* conv is 2d because color channel is not a spatial dimension
* speech would require a 1d convolution
* video would require a 3d convolution
* we can use 3d convolution to operate on 3d objects (voxels)
* in conv layers we specify also padding mode (valid,same.,full)
* first params is the # of feature maps to generate
* using Dropouts between Convolutions is a controversial issue. mostly wrong

### Lecture 32. CNN for Fashion MNIST

* set the colab notebook for FashionMNIST 

```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Additional Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout
from tensorflow.keras.models import Model
# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape:', x_train.shape)
# the data is only 2d
# convolutional layer expects HxWxC
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)
# number of classes
K = len(set(y_train))
print('number of classes: ',K)
# build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i,x)
# Compile and fit
# note: make sure we are using a GPU enabled notebook for this
model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
r = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=15)
# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
```

* we see that the model is overfitting
* model is getting more confident on its incorrect predictions
* we plot the confusion matrix diagram

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
# label mapping
labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split()
# Show some misclassified examples

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i].reshape(28,28),cmap='gray')
plt.title('True label: %s Predicted: %s' % (labels[y_test[i]],labels[p_test[i]]));
```

### Lecture 33. CNN for CIFAR-10

* we repeat the process for CIFAR-10
```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Additional Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
# Load in the data
cifar10 = tf.keras.datasets.cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train,y_test = y_train.flatten(),y_test.flatten()
print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
# number of classes (find number of unique elements using set)
K = len(set(y_train))
print('number of classes: ',K)
# build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i,x)
# Compile and fit
# note: make sure we are using a GPU enabled notebook for this
model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
r = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=15)
# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
```
* again we are overfitting

### Lecture 34. Data Augmentation

* data augmentation helps us improve our model
* we produce multiple samples from one sample by manipulating (*without losing the feat we look for)
* it adds translational invariance
* deep learning keeps improving because data keeps increasing
* the more data we invent the more space it takes
* google colab wont have the space either
* it can be doen automatically
* Keras API delivers
* we need to know about generators and iterators
* in python2 for loops when we use `for i in range(10)` it creates the range
* if we use `xrange()` in python2  python does not create the list at all
* in python 3 this is default for `range()` if we `print(range(10)` we get `range(0,10)` which is not a list
* in python we can create a random generator using yield without ever creating a list in memory

```
def my_random_geenrator():
	for _ in range(10):
		x = np.random.randn()
		yield x
```

* Keras does the same to do data augmentation on the fly, the pseudocode would be:

```
def my_image_augmentation_generator():
	for x_batch,y_batch in zip(x_train,y_train):
		x_batch = augment(x_batch)
		yield x_batch,y_batch
```

* The code for the autogenerator in Keras is

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=True
)

train_generator = data_generator.flow(
	x_train,y_train,batch_size)

steps_per_epoch = x_train.shape[0] // batch_size
r_model.fit_generator(
	train_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=50)
```

* other args we can use 'rotation_range,vertical_flip,shear_range,zoom_range,brightness_range'
* be careful to keep semantical context
* when using a generator we need to set the steps per epoch.
* otherwise generating will keep doing its job indefinitely...
* fit_generator also returns history for plotting

### Lecture 35. Batch Normalization

* We will see the intermediate normalization/standarization layer used in CNNs
* we have seen how important is to normalize/standardize data before passing them into linear/logistic regression
* the problem is only input data get normalized. after passing from adense layer data is no longer normalized
* We can solve this with Batch Normalization
	* In TF with `model.fit()` we do batch gradient descent

```
for epoch in range(epochs):
	for x_batch,y_batch in next_batch(x_train,y_train):
		w <- w - learning_rate * grad(x_batch,y_batch)
```

* In our models we inject Layers that do Batch Normalization
	* the layer looks at each batch of data, 
	* calculates the mean and standard deviation on the fly
	* standardazes based on that z=(x-mean)/stddev
	* How do we know the normalization is good??? we dont
	* Batch normalization fixes this. it starts with mean and stdev based on data. 
	* then it shifts them to optimal with gradient descent.
	* z = (x-meanB)/stddevB y = zgamma+betta (gamma,betta are leatrnt)
* Batch Normalization acts as a regularizer, prevents overfitting like Dropout
* How Batch Norm does Regularization?
	* every batch is different so we get different batch mean and stddev
	* there is no true mean and dev for the whole dataset
	* this acts as noise, using noise in the model it makes it resilient to noise.
	* when we fit to the noise we overfit
* Batch norm is not applied between dense layers. there we use dropout
* It is used between Convolution layers in CNNs

### Lecture 36. Improving CIFAR-10 Results

* we will improve the CIFAR10 results by applying the new techiques we learned
* our new model

```
# build the model using the functional API
# inspired by VGG
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x=  MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x=  MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3),activation='relu',padding='same')(i)
x = BatchNormalization()(x)
x=  MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i,x)
```

* we fit with generator

```
# Fit with data augmentation
# if we run this after starting previous fit it will continue where it left off
batch_size = 12
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=True
)

train_generator = data_generator.flow(
	x_train,y_train,batch_size)

steps_per_epoch = x_train.shape[0] // batch_size
r = model.fit_generator(
	train_generator,
  validation_data=(x_test,y_test),
	steps_per_epoch=steps_per_epoch,
	epochs=50)
```

## Section 6: Recurrent Neural Networks, Time Series, and
Sequence Data

### Lecture 37. Sequence Data

* Time Series AKA Sequence = Any continuous-valued measurement taken periodically (e.g sensor data, stock data)
* Forecasting can have multiple benefits
* e.g Weather is dynamc and difficult to predict
* speech and audio are time series
* Text is an example where DL excels. in classical ML we use the 'Bag of Words' feature vector
* Bag of Words
  * e.g document classification
  * create a long feature vector (one netry for each word in English vocabulary)
  * Inside this vector we put a count of how many times each word appeared in the text
  * our data set will compose of each documents feature vector and its lable
  * when we only use counts we lose contex of the order of the words
* Sequence
  *  1D time series
  *  just like with linear regressin we can expand the concept to multiple dimensions
  *  For non-sequential (e.g tabular) data we have an NxD matrix
* Shape of Sequence
  * length? N? D? N=#samples D=#features
  * T = sequence length (T for time)
  * Our data is represented as a 3D array of size NxTxD
* Example: Location Data
  * model employee path to work recording GPS data from cars
  * N: one sample would be one person's single trip to work
  * D: =2 (GPS will record Latitude and Longitude pairs)
  * T: the number of (lat,lng) measurements taken from start to finish of a single trip
* Variable Length Sequence
  * each trip has different number of samples T
  * TF/Keras work with equal sized Tensors
* Example: Stock Prices
  * just a single value: D=1
  * suppose we use a time window of 10 samples T=10 to predict the next sample
  * N = number of time windows in the time series (say L samples) N=L-T+1
  * more like 1D convolution
  * if we want to measure 500 stocks? D = 500 . if T=10 a sample will be TxD = 5000
* Example. Neural Interfaces
  * D = # of electrodes in the brain
  * Predict the leteer we want to type using 1sec of measurements with sampling rate of 1sample/ms
  * T = 1000
  * N = #of letters our test subject tried to type
  * D = # of electrodes
* Why NxTxD??
  * all ML libraries in Python conform with this standard
  * we use to put the number of features last
* Variable length sequences
  *  in the past we used variable length sequences in RNN. 
  *  this is very compicated to work with and are inefficient data structs
  *  NxTxD is a single array (numpy arrays are fast)
  *  if T depends on which sample, we might use T(n) instead. we would have to use a list (inefficient)
* In TF and Keras we use constant length sequences if we dont want to use custom code
* We use padding with 0s. NN thinks all NxTxD array is full of legit data so we waste resources

## Lecture 38. Forecasting

* many do it in a way that looks nice but does not make sense
* for us Forecsting means to predict the next values (multiple) of a time series
* Number of future steps we want to predict is called the horizon 
  * e.g predict demand for next 3-5 days for product manufacturing
  * hourly weather for next 7 days 7x24 = 168
* it makes virtually no sense to predict just one step ahead
* Simplest way to do a forecast given an 1-D time series?? Linear Regression
  * Most 'time series analysis' involves only linear regression
  * If our data is NxTxD and linear regression expects only NxD, how it works??
  * For 1-D time series D=1 (superfluous) so it is an NxT array if we flatten NxTx1
  * To do linear regression we pretend T is D
* Example.
  * time series of length 10 [1,2,3,2,1,2,3,2,1,2]
  * predict next val using past 3 vals
  * input matrix (x) has shape Nx3, target(Y) has shape N = num of time windows that fit in the time series
  * because we wnat to predict next val we cannot use all  = 10-3+1 = 8 but 7 (we can think of it as winodw is size 4 including the target)
  * This is called an 'AutoRegressive Model (AR)' in statistics xhatt= w0+w1xt-1+w2xt-2+w3xt-3
* How we forecast with AR?
  * put Xtest into `model.predict(Xtest)` to yield prediction for x11, x12,x13
  * this is wrong. forecasting is about predicting multiple steps ahead. 
* In order to predict multiple steps ahead we must use our earlier predictions as inputs
* Beacuse we must use our own predictions we cant just do `model.predict()` in one step
```
x = last_values_of_train_set
predictions=[]
for i in range(length_of_forecast):
  x_next = model.predict(x)
  predictions.append(x_next)
  x = concat(x[1:], x_next)
```

* Can we apply this rule to make more powerful predictor?
* A limitation of linear regression is that prediction can be a linear function of inputs
* ANN is more powerful using same interface
* Linear Regression Model
```
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i,x)
```

* ANN non-linear
```
i = Input(shape=(T,))
x = Dense(10,activation='relu')(i)
x = Dense(1)(x)
model = Model(i,x)
```

### Lecture 39. Autoregressive Linear Model for Time Series Prediction

* we will write down a notebook for an AR model using a synthetic dataset
```
# Install and import TF2
!pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)
# Additional imports
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Make the original data
series = np.sin(0.1*np.arange(200)) #+ np.random.randn(200)*0.1

# Plot it
plt.plot(series)

### Build the dataset
# lets see if we cn use T past values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series) -T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1,T)
Y = np.array(Y)
N = len(X)
print("X.shape",X.shape,"Y.shape",Y.shape)

# Try the autoregressive linear model
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i,x)
model.compile(
    loss="mse",
    optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
    X[:-N//2],Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:],Y[-N//2:]),
)
```

* we train in the first half of the dataset and validate on the second half of the dataset
* loss is almost 0
* next we will do a 'wrong' forecast using true targets
* NOTE! `model.predict()` returns NxK output .for us N = 1 and K=1
```
# "Wrong" forecast using true targets

validation_target = Y[-N//2:]
validation_predictions = []

# index of first validation input
i = -N//2

while len(validation_predictions) < len(validation_target):
  p = model.predict(X[i].reshape(1,-1))[0,0] #1x1 array => scalar
  i += 1
  # update the predictions list
  validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* results are correct
* next we forecast the correct way using only self-predictions for making predictions)

```
# Forecast future values (use only self-predictions for making predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] #!-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1,-1))[0,0] #1x1 array => scalar

  # update the predictions list
  validation_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x,-1)
  last_x[-1] = p
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```
* again the prediction is perfect (no noise signal)
* we add noise uncommenting `#+ np.random.randn(200)*0.1` and rerun
* wrong forecast gives almost perfect result. but its misleading
* the proper forecast predicts the main pattern (sinusoidal) but it filters out all the noise

### Lecture 40. Proof that the Linear Model Works

* How linear regression can forecast a non-linear signal like sinusoidal
* It is possible to predict perfectly a clear sine wave using only 2 previous values
* this is called the AR(2) model
* no bias term is needed x(t)=w1x(t-1)+w2x(t-2)
* if we want to write a timeseries as a function of time it looks like x(t)=sin(ωt)
* we used that whenwe made our time series `np.sin(0.1*np.arange(200))` where `np.arange(200)` are the values of t and 0.1 the angular frequency
* We can now prove our AR(2) modl can learn the sine wave
* we plug the sine wave function to the recurrence relation sin(ω(t+1)) = w1sin(ωt) +w2sin(ω(τ-1))
* derivation is easier when we shift by 1 starting at t+1
* Fibonacci is another example of a recirrence relation where weight are 1 Fib(n)=Fib(n-1)+Fib(n-2)
* then we multiply by ω: sin(ω(t+1))=w1sin(ωt)+w2.sin(ω(t-1)) => sin(ωt+ω)=w1sin(ωt)+w2.sin(ωt-ω) => sin(ωt+ω) -w2.sin(ωt-ω)=w1sin(ωt) 
* we know sin(a+b)+sin(a-b)=2cos(b).sin(a) smae like ours where w1 = 2cos(b) and w2 = -1 ωτ=a ω=b

### Lecture 41. Recurrent Neural Networks

* We approached CNNs with ANN in mind flattening images and treating them as feature vectors
* We also aproached time sequence data pretending T is D (for 1D sequence) and passing it in a NxT matrix as Input for tabular data model
* what if D>1?
* Why not flattne it out makeing our TxD series a single feature vector? we ve done this with images already
* 1st problem: Full matric Multiplication Takes Up Space. In CNNs we took advantage of data structure to use convolution
* RNNs do the same unlike ANNs that are generic
* How we can exploit the structure of the sequence?
  * we take inpiration from forecasting
  * we know that to predict an x value past values are useful
  * what if we apply this to the hidden layer feature vectors?
  * hidden vector is calculated from input: h = σ(WhTx+bh)
  * output is calculated from hidden vector: yhat = σ(WoTx+bo)
  * To make an RNN we make the hidden feature vector depend on previous hidden state (its previous value)
  * recurrent loop implies a time delay of 1
  * linear regression forecasting model: output is linear function of inputs
  * now: hidden layer is a non-linear function of input and past idden state?what kind of non-liear function? a neuron
* RNN equation:
  * Typically use 1 hidden layer (unlike CNN)
  * h(t) is a non linear function of h(t-1) and x(t)
  * Elman Unit or Simple Recurent Unit: ht=σ(WxhTxt+WhhTht-1+bh) x:input,h:hidden,o:output,xh:input-to-hidden,hh:hidden-to-hidden(recursive)
  * yhat=σ(WoTht+bo)
* How do we calculate output prediction given a sequence of inputs
  * Sequence of input vectors: x1,x2,...,xT , Shape(xt) = D
  * From x1 we get h1 and then yhat1 => h1 = σ(WxhTx1+WhhTh0+bh) h1 depends on h0 => yhat1 => σ(WoTh1+bo)
  * h0 is the initial hidden state, can be a learned param.in TF is not learnable. just assume its 0
  * Next we repeat for x2 to get h2 and then yhat2 => h2 = σ(WxhTx2+WhhTh1+bh) => yhat2 => σ(WoTh2+bo)
  * we repeat to hT and yhatT. it becomes clear that each yhatT depends on x1,x2...,xt but not xt+1,...,xT
* why we have a yhat for each time step?? if we do forecasting we care about next value
* for these problems all yhats except from last are ignored. Keep only yhatT=f(x1,x2,..,xT)
* there are some cases when we want to keep all the yhats. e.g when we doNeural machine  Translation. (both input and output series are meaningful sentences)
* For ANN or CNN output is the probaility of y to be in each category given the input p(y=k|x)
* Machine Translation is classification because the target is a word (a class in the diccionary)
* What is the given in this case?? p(yt=k|?)
* An unrolled RNN is a NN rolled out in time states 1,2,...T where each hidden layer h connects to nest time step h layer
* its clear that the given in an RNN are all the x p(yT=k|x1,x2...,xT)
* The Relationship to Markov Models become apparent
  * The Markov assumption is that the current value depends only on the immediate previous value p(xt|xt-1,xt-2,..,x1) = p(xt|xt-1)
  * thats absurd. if the current word is the how can we forecast the next word?
* An RNN on the other hand is much more powerful as it will do the forecast based on ALL previous words
* We put down some Pseudocode to show RNNs
```
Wxh = input to hidden layer
Whh = hidden to hidden layer
bh = hidden bias
Wo = hidden to output weight
bo = output bias
X = TxD input matrix

tanh hidden activation
softmax output activation

Yhat = []
h_last = h0
for t in range(T):
  h_t = tanh(X[t].dot(Wx) + h_last.dot(Wh) +bh)
  yhat = softmax(h_t.dot(Wo)+bo)
  Yhat.append(yhat)
  # update h_last
  h_last = h_t
```

* There is biological inspiration in RNNs. there is no reason for neuron to go only in 1 direction
* Hopfield networks are loop neuron networks in Brain. one way to train them is Hebbian Learning
  * neurons that fire together wire together, and neurons that fire out of sync fail to link 
* also in electronics RNNs have similarity with resgisters and memory
* What are the savings of RNN against an ANN.
  * e.g CNNs have 'shared weights' to take advantage of data structure.
  * in RNNs also we apply the same Wxh for each x(t) and the same Whh to get from h(t-1) to h(t)
  * the savings are again huge Wxh = DxM Whh = MxM Wo = MxK

### Lecture 42. RNN Code Preparation

* we will do the same forecasting excercise we did for Autoregressive linear model but now with a Simple RNN
* Steps:
* Load in the data (fix data shape for RNN: NxTxD)
  * supervised learnign dataset
  * sequence of length T
  * sine wave with/without noise
  * Linear regression expect 2D array NxT
  * RNN expects 3D => NxTx1
* Build the model
  * `SimpleRNN(5, activation='relu)` layer.
* Train the model
  * same as AR 
* Evaluate the model
  * same
* Make predictions (again be careful with shapes)
  *  input shape will be NxTxD => output NxK (N=1,D=1,K=1 for prediction)
  *  a single time-series input will be an 1-D array of length T 
  *  `model.predict(x.reshape(1,T,1)[0,0]` to make it scalar as output is 2D NxK

### Lecture 43. RNN for Time Series Prediction

* we cp the notebook to test Simple RNN
* first part is same with AR
* we test first sine without noise
* we build the dataset
* * SimpleRNN default activation is tanh unless defined otherwise
```
### Build the dataset
# lets see if we cn use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) -T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1,T,1) # now the dat should be N x T X D
Y = np.array(Y)
N = len(X)
print("X.shape",X.shape,"Y.shape",Y.shape)
# Try the Simple RNN model
i = Input(shape=(T,1))
x = SimpleRNN(5)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss="mse",
    optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
    X[:-N//2],Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:],Y[-N//2:]),
)
```

* loss is good
* we do 1 step forecast like before is perfect
* we do forecast using the RNN with default params

```
# Forecast future values (use only self-predictions for making predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] #!-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1,-1,1))[0,0] #1x1 array => scalar

  # update the predictions list
  validation_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x,-1)
  last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* its not perfect like AR model. RNN is mor flexible we have some shifting
* if we set in SimpleRNN activation=None our forecast is perfect like with AR. without activation method the model reduces to a linear model
* we add noise keeping activation=None
* 1 step forecast but multistep one is bad.same for tanh activation
* with relu activation is even worse in multistep forecast. loks like copying previous value

### Lecture 44. Paying Attention to Shapes

* we will go through a notebook that emphasizes the importance of shapes in RNNs

```
# Things we should know and memorize

# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units (neurons)
# K = number of output units (neurons)
```

* we make data and build the model when we see explicitly the shapes
```
# Make some data
N = 1
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)
# Make an RNN

M = 5 # number of hidden units

i = Input(shape=(T,D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)

model = Model(i,x)
# Get the output
Yhat = model.predict(X)
print(Yhat)
print('Yhat.shape', Yhat.shape)
```

* we use our model to make a prediction. the output is NxK (i,2)
* we want to try to replicate the output by geting the weights
```
# see if we can replicate this output
# get the weights first
model.summary()
# see what's returned
model.layers[1].get_weights()
# Check their shapes
# Should make sense
# First output is input > hidden
# Second output is hidden > hidden
# Third output is bias term (vector of length M)
a,b,c = model.layers[1].get_weights()
print(a.shape,b.shape,c.shape)
```

* we get `(3, 5) (5, 5) (5,)` or inout > hidden (D,M) hidden > hidden (M,M) hidden>output(M,) It makes sense

```
Wx,Wh,bh = model.layers[1].get_weights()
Wo,bo = model.layers[2].get_weights()
# we manualy calculate the output
h_last = np.zeros(M) # initial hidden state
x = X[0] # the one and only sample
Yhats = [] # where we store the outputs

for t in range(T):
  h = np.tanh(x[t].dot(Wx)+h_last.dot(Wh)+bh)
  y = h.dot(Wo)+bo # we only care about this value on the last iteration
  Yhats.append(y)
  #important: assign h to h_last
  h_last = h

#print the final output and confirm
print(Yhats[-1])
```

* the output is the same with our prediction 

### Lecture 45. GRU and LSTM (pt 1)

* Modern RNN Units: 
	* LSTM (LongShort-Term Memory)
	* GRU (Gated Recurrent Unit)
* GRU is like a simplified version of the LSTM (less params and more efficient)
* Why we need them?
  * Think of the vanishing gradients problem,
  * The output prediction of SimpleRNN is a huge composite function,depending on x1,x2,...,xT
* We see that derivatives (like Wxh) appears several times
* we need to find which derivatives appear all the time in the yhat equation
* eg we need gradient WchTxt for all inputts x1->T (all timesteps). the final gradient will be a function of all these individual derivs
* an other parameter we need to consider is how deep nested a term is. e.g x1 is the most deeply nested of all
* from ANNs we know that composite functions turn into multiplications in the derivative form. more nested more multiplications
* so RNNS are very prone to the vanishing gradient problem. RNNss cant learn from inputs far back due to vanishing gradient.
* using RELU like in ANNs does not cut it
* in RNNs we have to use GRU and LSTM units (neurons)
* LSTMs were created in 1997. GRUs in 2014 and is simpler and more efficient
* Gated Recirrent Unit (GRU)
  *  has the same IOs (API) with Simple RNN x(t),h(t-1) => GRU => h(t)
  *  Simple RNN: ht = tanh(WxhTxt + WhhTht-1 + bh)
  *  GRU: update gate vector zt = σ(WxzTxt + WhzTht-1 + bz)  size=M 
  *  GRU: reset gate vector rt = σ(WxrTxt + WhrTht-1 + br)  size=M
  *  GRU: hidden state ht = (1-zt)(o)ht-1 + zt(o)tanh(WxhTxt+WhhT(rt(o)ht-1) + bh) size=M
  *  M is a hyperparameter (number of hidden units/feats)
  *  shape of weights from x(t) = DxM
  *  shape of weights from h(t) = MxM
  *  bias is size=M
  *  (o) is elemnt-wise multiplication x[0]y[0], x[1]y[1] etc
  *  update gate vector : should we take the new value for h(t) or keep the old one h(t-1)? => carry on past states z(t)->0 remember z(t)->1 forget h(t-1)
  *  it is a logistic regression (neuron). binary classification which val to choose for h(t)
  *  reset gate vector: a neuron , a switch for remembering/foget h(t-1) but different parts of it compared to zt.

### Lecture 46. GRU and LSTM (pt 2)

* GRU solution to SimpleRNN vaishing gradient problem: make hidden state a weighted sum of the previous state and new state thus allowing to remember old state
* gates are binary classifiers doing logistic regression aka neurons
* GRUs heve less params than LSTM thus perform better.
* Cutting edge research favors LSTMs [Paper1](https://arxiv.org/abs/1703.03906)[Paper2](https://arxiv.org/abs/1805.04908)
* LSTM is like GRU but with more states and more gates thus more complex
* LSTM has dfferent API than GRU or SIMPLERNN x(t),h(t-1),c(t-1)=>LSTM=>h(t),c(t)
* it has an additional term or state c(t) or the Cell State
* We usually ignore it like GRU intermediate vectors z,r we calculate it but dont use it
* In TF LSTM unit is by default simplified to output only h(t) but can be configured to output c(t) as well
* LSTM equations:
  *   Forget gate vector (a neuron/binary classifier) : ft = σ(WxfTxt + WhfTht-1 + bf)  size=M 
  *   Input/Update gate vector (a neuron/binary classifier) : it = σ(WxiTxt + WhiTht-1 + bi)  size=M
  *   Output gate vector (a neuron/binary classifier) : ot = σ(WxoTxt + WhoTht-1 + bo)  size=M (controls which parts of cell state go to output)
  *   Cell state  (weighted sum) : ct = ft(o)ct-1 + it(o)fc(WxcTxt + WhcTht-1 + bc) <- SimpleRNN term with activation function fc (usually tanh)
  *   Hidden State  : ht = ot(o)fh(ct) <- activation function fh (usually tanh)
  *   fc and fh are tanh by default in Tensorflow and Keras. we can change them both e.g to relu with activation argument,
  *   to change them individually we habv eto write our own LSTM in TF but then it wont be GPU optimized
* Options for RNN Units
  * for each x1,x2,...,xT we will calculate h1,h2,...,hT
  * SimpleRNN,GRU<LSTM return only hT by default
  * we may want all of h1,h2..hT to get yhat1,yhat2,yhatT for our outut predictions
```
i = Input(shape=(T,)) # size NxTxM
x = Dense(K)(x) # size NxTxK
```
* if we set `return_state=True` in LSTM we get cT oT and hT where oT=hT

### Lecture 47. A More Challenging Sequence

* we will see the LSTM in action in a notebook using a synthetic signal to compare its results with SimpleRNN and AR

```
# Additional imports
from tensorflow.keras.layers import Input,Dense,Flatten,SimpleRNN,GRU,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make the synthtic original data
series = np.sin((0.1*np.arange(400))**2) #x(t)=sin(ωt^2)
# plot it
plt.plot(series)

### build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T) # make it N x T
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

# try autoregressive linear model
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i,x)
model.compile(
    loss="mse",
    optimize=Adam(lr=0.01),
)

# train the model
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:], Y[-N//2:]),
)
```

* we see that AutoRegression has a high loss on this fuzzy signal
```
# One-step forecast using true targets
# Note: even the one-step forecast fails badly
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title('Linear Regression Predictions')
plt.legend()
# "Wrong" forecast using true targets

validation_target = Y[-N//2:]
validation_predictions = []

# index of first validation input
i = -N//2

while len(validation_predictions) < len(validation_target):
  p = model.predict(X[i].reshape(1,-1))[0,0] #1x1 array => scalar
  i += 1
  # update the predictions list
  validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* multistep forecast is even worse
```
# Forecast future values (use only self-predictions for making predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] #!-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1,-1))[0,0] #1x1 array => scalar

  # update the predictions list
  validation_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x,-1)
  last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* we now try an RNN and repeat predictions
```
### Now Try an RNN/LSTM model
X = X.reshape(-1,T,1) # make it NxTxD

# make the RNN
i = Input(shape=(T,D))
x = SimpleRNN(10)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss="mse",
    optimizer=Adam(lr=0.05),
)

# train the RNN
r = model.fit(
    X[:-N//2],Y[:-N//2],
    batch_size=32,
    epochs=200,
    validation_data=(X[-N//2:],Y[-N//2:]),
)
  validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* multistep forecast is even worse
```
# Forecast future values (use only self-predictions for making predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] #!-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1,-1))[0,0] #1x1 array => scalar

  # update the predictions list
  validation_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x,-1)
  last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* we now try an RNN and repeat predictions
```
### Now Try an RNN/LSTM model
X = X.reshape(-1,T,1) # make it NxTxD

# make the RNN
i = Input(shape=(T,D))
x = SimpleRNN(10)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss="mse",
    optimizer=Adam(lr=0.05),
)

# train the RNN
r = model.fit(
    X[:-N//2],Y[:-N//2],
    batch_size=32,
    epochs=200,
    validation_data=(X[-N//2:],Y[-N//2:]),
)
```

* RNN does better in one step pred, ok in multi-step (cannot follow freq )
```
# One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.title('many-to-one RNN')
plt.legend()

# multi-step forecast
forecast = []
input_ = X[-N//2]
while len(forecast) <len(Y[-N//2:]):
  # reshape the input to NxTxD
  f = model.predict(input_.reshape(1,T,1))[0,0]
  forecast.append(f)

  #make a new input with the latest forecast
  input_ = np.roll(input_,-1)
  input_[-1] = f

plt.plot(Y[-N//2:], label='targets')
plt.plot(forecast, label='forecast')
plt.title('RNN forecast')
plt.legend()
```
* RNN does better, forecast matches frequency nicely
* we replace inmodel SimpleRNN with  LSTM and rerun
```
# x = SimpleRNN(10)(i)
x = LSTM(10)(i)
```

* we see that LSTMs are not so good either
* LSTMs work better for long term dependencies

### Lecture 48. Demo of the Long Distance Problem

* we build the dataset
```
### build the dataset
# this is a nonlinear and long-distance dataset
# (actually, we will test long-distance vs short-istance patterns)

# start with a small T and increase it later
T = 10
D = 1
X = []
Y = []

def get_label(x, i1,i2,i3):
  # x = sequence
  if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
    return 1
  if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
    return 1
  if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
    return 1
  if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
    return 1
  return 0

for t in range(5000):
  x = np.random.randn(T)
  X.append(x)
  y = get_label(x, -1, -2, -3) # short distance
  # y = get_label(x, 0, 1, 2) # short distance
  Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(Y)
```

* we put a pattern that is used for classification once in the end of the sequence (RNN  does not need long term memory to remember the pattern) and once in the distant past (RNN must have long termn memory to remember it)
* we try linear model. we expect a fail as its a nonlinear classification
```
# Try a linear model first - note: its a classification problem now
i = Input(shape=(T,))
x = Dense(1,activation='sigmoid')(i)
model = Model(i,x)
model.compile(
    loss="binary_crossentropy",
    optimize=Adam(lr=0.01),
    metrics=['accuracy']
)

# train the model
r = model.fit(
    X, Y,
    epochs=100,
    validation_split=0.5,
)
```

* loss and accuracy are bad
* we try a simple RNN
```
### Now Try a simple RNN
inputs = np.expand_dims(X,-1)

# make the RNN
i = Input(shape=(T,D))

# method1
x = SimpleRNN(10)(i)
# x = LSTM(5)(i)
# x = GRU(5)(i)

# method2
# x = LSTM(5, return_sequences=True)(i)
# x = GlobalMaxPool1D()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i,x)
model.compile(
    loss="binary_crossentropy",
    # optimizer='rmsdrop',
    # optimizer='adam',
    optimizer=Adam(lr=0.01),
    # optimizer=SGD(lr=0.1,momentum=0.9),
    metrics=['accuracy'],
)

# train the model
r = model.fit(
    inputs, Y,
    epochs=200,
    validation_split=0.5,
)
```

* RNN solves the problem well as the feature is not in distant past
* we make the dataset such as its a long distanc eproblem
* SimpleRNN can not solve it as we have a vanishing gradient problem
* we replace SimpleRNN layer with LSTM and train again. now we see good results
* we increase sequence length from 10 to 20 and rerun. LSTM solves it with 200 epochs but its getting difficult
* we try GRU instead of LSTM for 400 epochs. GRU has problem
* we set sequence length T=30 and try LSTM. it cannot cut it now to find a pattern 30 steps in the past
* we try a different approach with LSTM and Max Pooling
* we try `x = LSTM(5, return_sequences=True)(i)` which for input TxD returns TxM as it returns all hidden states. h1 to hT. 
* so we need max pooling to get back to size M `max{h(1)...h(T)}`
* simple LSTM with Return Sequences False returns size M as it returns only h(T)
* `GlobalMaxPooling2D` goes from HxWxC to C
* `GlobalMaxPooling1D` goes from TxM to M
* we will try LSTM with `return_sequences=True` to return the intermediate hidden states
```
### Now try LSTM with Global Max pooling
inputs = np.expand_dims(X,-1)

# make the RNN
i = Input(shape=(T,D))

# method2
x = LSTM(5, return_sequences=True)(i)
x = GlobalMaxPool1D()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i,x)
model.compile(
    loss="binary_crossentropy",
    # optimizer='rmsdrop',
    # optimizer='adam',
    optimizer=Adam(lr=0.01),
    # optimizer=SGD(lr=0.1,momentum=0.9),
    metrics=['accuracy'],
)

# train the model
r = model.fit(
    inputs, Y,
    epochs=100,
    validation_split=0.5,
)
```
* it improves results a lot
* return_sequence=False
  * After RNN unit h(T):M
  * Output: K
* return)sequence=True
  * Input: TxD
  * After RNN Unit h(1),h(2)...h(T): TxM
  * After global max pooling max{h(1),h(2)...h(T)}:M
  * Output: K
* For Images we use GlobalMaxPooling2D: input: HxWxC => output: C
* For sequences we use GlobalmaxPooling1D: input: TxM => output: M
* To extend our Exercise we can spread out the relevant points 'pattenrs'
* this simulates language; we cant parse the meaning of a sentence without considering the reationships between words whih are far apart
* Also adding GlobalmaxPooling to SimpleRNN or GRU
* Stacking RNN layers
  * the input into a 2nd LSTM unit will be h(1),h(2)...h(T) from the first LSTM unit like x(1),x(2)...x(T) are inputs to first LSTM

### Lecture 49. RNN for Image Classification (Theory

* We take a dumb approach considering tabular data (survey answers)
* these feats made up the input feature vector
* for images we flattended the pixels making it a feature vector.
* we pretend each pixel is the answer to a survey question
* for time series we pretend each value in teh sequence is the answer to a survey question
* a multidimensional times series is a TxD matrix (each column is a time series)
* consider a black and white image MNIST, fashionMNIST. its a HxW matrix (2-D). as it is also 2D we can pretend the image is a time series
* RNN can be perceived as an image scanner, it looks at the image each row at the time
* CODE prep
  * load in the data (MNIST): X is of size NxTxD (T=D=28)
  * instantiate the model: LSTM=> Dense (10,activation='softmax')
  * fit/plot the loss

### Lecture 50. RNN for Image Classification (Code)

```
# Load in the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:",x_train.shape)
# Build the model

i = Input(shape=x_train[0].shape)
x = LSTM(128)(i)
x = Dense(10,activation='softmax')(x)

model = Model(i,x)
# compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=10)
```

* it works ok as it captures long distance relationships.
* its long distance as patterns (feats) can apear anywhere in the image (sequence of pixels)

### Lecture 51. Stock Return Predictions using LSTMs (pt 1)

* many people claim to predict stock prices using LSTMs
* wrong claims can spread in the internet
* we load 5 year Starbux stockprices in a dataframe, transform ans scale the data for RNN and use LSTMs
```
# we load inb CSV with SBUX stockprices from 2013-2018
df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')
# Start by doing the wrong thing - trying to predict the price itself
series = df['close'].values.reshape(-1,1)
# Normalize the data
# The boundary is just an approximation
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()
### Build the dataset
# let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X).reshape(-1,T,1) # Now the data should be NxTxD
Y = np.array(Y)
N = len(X)
print("X.shape",X.shape,"Y.shape",Y.shape)
### Try Autoregressive RNN model
i = Input(shape=(T,1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(
    loss='mse',
    optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:],Y[-N//2:]),
)
# One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
```

* one step forecasting seems unrealistically good
* we try multi-step forecasting
```
# Forecast future values (use only self-predictions for making predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] #!-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1,T, 1))[0,0] #1x1 array => scalar

  # update the predictions list
  validation_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x,-1)
  last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
```

* it fails BADLY. so in one step forecasting its just copying previous value
* time series like images are neighbor correlated data

### Lecture 52. Stock Return Predictions using LSTMs (pt 2)

* one step prediction on stock prices is misleading and unconventional
* what is more conventional is the stock return (net profit/loss) for a given time (say day or week)
* R = (Vfinal - Vinitial)/Vinitial
```
# calculate returns by first shifting the data
df['PrevClose'] = df['close'].shift(1) # move everything up 1

# so now its like
# close  / prev. close
# x[2] x[1]
# x[3] x[2]
# x[4] x[3]
# ...
# x[t] x[t-1]
# the the return is
# x[t] -x[t-1] / x[t-1]
df['Return'] = (df['close'] - df['PrevClose']) / df['PrevClose']
# we check distribution of vals
df['Return'].hist()
# they are concentrated so we need to normalize them
series = df['Return'].values[1:].reshape(-1,1)
# normalize the data
# note: I didn't think about where the true boundary is, this is just aprox
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
scaler = scaler.transform(series).flatten()
```

* we follow same steps as before to prove that the model cannot do much execpt one step prediction.

### Lecture 53. Stock Return Predictions using LSTMs (pt 3)

* we will use all data: open,close,high,low,volume (D=5)
* we ll try to predict if the price will go up or down (binary classification)
* regression is harder than classification
```
# Now turn the full data into numpy arrays

# Not ye in the final 'X' format
input_data = df[['open','high','low','close','volume']].values
targets = df['Return'].values
# Now make the actual data which will go into the neural network
T = 10 # the number of time steps to look at to make a prediction for the next day
D = input_data.shape[1]
N = len(input_data) - T # (e.g if T=10 and ou have 11 data pointsthen we only have 1)
# Normalize the inputs
Ntrain = len(input_data) * 2 // 3
scaler = StandardScaler()
scaler.fit(input_data[:Ntrain+T])
input_data = scaler.transform(input_data)
# Setup X_train and y_train

X_train = np.zeros((Ntrain, T, D))
y_train = np.zeros(Ntrain)

for t in range(Ntrain):
  X_train[t,:,:] = input_data[t:t+T]
  y_train[t] =(targets[t+T]>0)
# Setup X_test and y_test
X_test = np.zeros((N-Ntrain, T, D))
y_test = np.zeros(N-Ntrain)

for u in range(N-Ntrain):
  # u counts from 0...(N - Ntrain)
  # t counts from Ntrain...N
  t = u + Ntrain
  X_test[u,:,:] = input_data[t:t+T]
  y_test[u] = (targets[t+T] > 0)
# Make the RNN
i = Input(shape=(T,D))
x = LSTM(50)(i)
x = Dense(1,activation='sigmoid')(x)
model = Model(i,x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy'],
)
# train the RNN
r = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=300,
    validation_data=(X_test,y_test),
)
```

* model is overfitting on noise
* accuracy is no better than random guessing
* one step prediction of stock price seems accurate
* one step prediction of stock return is more realistic but less accurate\
* 5 cols for binary classification is terrible
* trying to predict stock prices from stock prices is flawed
* stock prices is result of occurence in real world
* stock depends on investors emotions. how the company appears in media, nw investors etc

## Section 7: Natural Language Processing (NLP)

### Lecture 54. Embeddings

* we used RNNs for models on sequence data
* text is also sequence data but not continuous. they are categorical objects
* we cannot use SImpleRNNs because text entries are not numerical
* if we try to hot encode words using an array the size of the vocabulary (V) we end up with a hugely sparce vector for each word
* if we have a sequence of T words which get one-hot encoded in size V vectors. the sentence becomes TxV matrix
* V can be as big as 1million. 
* This creates a problem as to do classification we look for structures in input features aka clustering
* Embeddings is a better solution.
  * assign each word to a D-dimensional vector (not hot encoded)
* One hot endoding an integer k and multiplying that by a matrix is the same as selecting the k row of the matrix
* if  we index the weight table directly selecting the kth row: W[k] is much more efficient
* the Embedding Layer:
  * Step1: convert words into integers (index): `['i','like','cats'] => [50,25,3]`
  * Step2: use integers to index the word embedding matrix to get word vectors for each wordmining `[50,25,3]=>[[0.3,-0.5],[1.2,0.7],[-2.1,0.9]]` Tlength array -> TxD matrix
* Tensorflow Embedding
  * Convert sentences into sequences of word indexes `['i','like','cats'] => [50,25,3]` unique integers lik in classification
  * Embedding layer maps sequence of integers into sequence of word vectors `['i','like','cats'] => [50,25,3]` Tlength array -> TxD matrix
* How we find weights? 
  * we know that the word vectors must have some structure. words are clustering by meaning in dimensional space
  * like in CNNs the weights are found automatically when we call model.fit()
  * sometimes we use pretrained word vectors (trained with some other algorithm) 
  * we freeze the embedding layer weights so noly other layers are trained with fit. 
  * and build a model like: Input => Embedding(fixed) => LSTM => Dense => Output
  * read about word2vec and CloVe to learn more

### Lecture 55. Code Preparation (NLP)

* we know how to build BBs that accept numerical input
* we will see how to turn a seq of words into an acceptable format to convert into a TxD matrix
* before building word vectors we must convert words to integers (indexes to word embedding matrix)
* we need a mapping
```
dataset = long sequence of words
current_idx = 1
word2idx = {}
for word in dataset:
  if word not in word2idx:
    word2idx[word] = current_idx
    current_idx += 1
```
* we start indexing from 1 not 0. in TF we have constant seq length so all our data fits in NxTxD array
* T is the max seq length so sorter sentences need padding
* 0 is used for padding so 0 is not acceptable as index
* TF does words to index automatically
* we want list  of strings containing words (tokenization)
* Tensorflow Tokenizer
  * For a Google size dataset we mwy have 1 million tokens, most of which are extremely rare and useless
  * we can assign these to generic <RARE> or <UNKNOWN> token using fit_on_text
```
MAX_VOCAB_SIZE = 2
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
```

 * sequences stil need padding as they have different length
```
max_size = max(len(seq) for seq in sequnces)
for i in range(len(sequences)):
  sequences[i] = sequences[i] + [0]*(max_size - len(sequences[i]))
```

* TF does it automaticaly `data = pad_sequences(sequences, maxlen=MAXLEN)`
* Input: A list if N lists of integers (max lenght is T)
* Output: An NxT matrix (Or an NxMAXLEN matrix depending on which is smaller)
* If we truncate a sentence, will it truncate at beginning or end?
* we can control this by setting `truncating` arg to `pre` or `post`
```
data = pad_sequences(
  sequences,
  maxlen=MAXLEN,
  truncating="pre"
)
```
* we can control where padding goes also. this is useful as we usually want padding at beginning. RNNs have trouble to learn patterns in distance past 
```
data = pad_sequences(
  sequences,
  maxlen=MAXLEN,
  padding="pre"
)
```
* in translations target language might end with larger sentence than input. post padding is better then
* N = # samples , T = sequence length, X[n,t] is the word appearing in  document n, timestep t
* the NxT matrix of word indices is passed to Embedding Layer to get a NxTxD tensor, it converts each word to a word vector
```
i = Input(shape=(T,))
x = Embedding(V,D)(i) # x is now NxTxD
...rest of RNN...
```

### Lecture 56. Text Preprocessing

* we build a notebook
```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Just a simple test
sentences = [
  "I like eggs and ham.",
  "I love chocolate and bunnies.",
  "I hate onions."
]
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# How to get the word to index mapping?
tokenizer.word_index
# use the defaults
data = pad_sequences(sequences)
print(data)
# customize padding
data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH,padding="post")
print(data)
```

### Lecture 57. Text Classification with LSTMs

* we will do spam detection in a notebook
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model
# unfortunately this URL doesn't work with pd.read_csv
!wget https://lazyprogrammer.me/course_files/spam.csv
# read with pandas using correct encoding
df = pd.read_csv('spam.csv',encoding='ISO-8859-1')
# drop unnecessary columns
df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
# rename columns to sthing better
df.columns = ['labels','data']
# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values
# split up the data
df_train,df_test,Ytrain,Ytest = train_test_split(df['data'], Y, test_size=0.33)
# convert sentences to sequences
MAX_VOCAB_SIZE=20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)
# get word -> integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique tokens.' % V)
# pad sequences so that we get a N x T matrix
data_train = pad_sequences(sequences_train)
print('Shape of data train tensor:',data_train.shape)
# get sequence length
T = data_train.shape[1]
data_test = pad_sequences(sequences_test,maxlen=T)
print('Shape of data test tensor:',data_test.shape)
# create the model

# We get to choose embeding dimensionality
D = 20

# Hidden stats dimensionality
M = 15

# Note: we actually want to the size of the embedding to (V+1) x D.
# because the first index starts from 1 and not 0.
# Thus, if the final index of the embedding matrix is V,
# then it actually must have size V+1.

i = Input(shape=(T,))
x = Embedding(V+1,D)(i)
x = LSTM(M,return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i,x)
# Compile and fit
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print('Training model....')
r=model.fit(
    data_train,
    Ytrain,
    epochs=10,
    validation_data=(data_test,Ytest)
)
```

* it converges fast with good accuracy

### Lecture 58. CNNs for Text

* CNNs work with sequences as well
* convolution is about multiplying and adding
* images have 2D and have correlations
* Sequences has 1 feature dimension. time
* Same type of correlation appears in images and sequences. nearby data are correlated
* 1D convolution is simpler than 2D convolution.slide filter along sequence. multiply and add. its called cross-correlation x(t)*w(t)=Σ[τ]x(t+τ)w(τ),τ=1..len(w)
* 1D Convolution with multiple Feature seq. 
  * Input is TxD (T=#of time steps, D = # of input feaures) 
  * Output is TxM (M=# of output features)
  * Then W (the filter) has the shape TxDxM
  * y(t,m)=Σ[τ]Σ[d=1->D]x(t+τ,d)w(τ,d,m)
* For images we have: 2 spatial dimensions, 1 input feature dimension, 1 output feaure dimension = 4 
* For sequences we have: q1 time dimension, 1 input feat di, 1 output feat dim=3
* Convolution is matrix mult with shared weights. a pattern matcher
* Convolution on Text:
  * we use embeddings to give us what we need
  * for 1D convolution we need TxD input
  * thats what we get when we use embedding on a seq of words with length T
  * the word vectros are of length D
* In Text CNNs the data shrinks in time dimension but grown in feature dimension(more featue maps)

### Lecture 59. Text Classification with CNNs

* we build a noteboo for text classification with CNNs
* only the model is different than before
```
from tensorflow.keras.layers import Conv1D,MaxPooling1D, Embedding
# create the model

# We get to choose embeding dimensionality
D = 20

# Note: we actually want to the size of the embedding to (V+1) x D.
# because the first index starts from 1 and not 0.
# Thus, if the final index of the embedding matrix is V,
# then it actually must have size V+1.

i = Input(shape=(T,))
x = Embedding(V+1,D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i,x)
```

* we train for 5 epochs now. it converes faster than RNN with better results

## Section 8: Recommender Systems

### Lecture 60. Recommender Systems with Deep Learning Theory

* recommender systems is the most applicable concept in ML
* used in any consumer-facing business. 
* we encounter them all the time (Youtube,Spotify,Netflix,Amazon,Fb,Google)
* pages in google is built with recommendations 
* even news do it to increase conversion rate...
* We will focus on Ratings Recommenders. it works with data that come in the form of triples (user,item,rating) e.g. (Alice,Avatar,5)
* Ratings Dataset must be incoplete... it cant be that a user will rate all movies e.g
* How it works???
  * Given a dataset of triples: (user,item,rating)
  * Fit a model to the data; F(u,i)->r
  * If the user u and item i appeared in the dataset, then the predicted rating shold be close to the true rating
  * the function should predict the rating of a user to  an item even if it didn;t appear in the training set
  * NNs (function approximators) do this.
* The recommendation comes from the ability of our model to predict ratings for unknown items.
* For a given user, get predictions for every item. 
* then sort them by predicted rating in descending order
* present as recommendations the items with highest predicted rating
* To build the model:
  * both users and items are categorical data (problem)
  * NNs work with matrices. we cannot multiply a category with a number
  * We resort to NLP for insipration
  * we use embeddings to map a category to a feature vector
* IN Recommender systems we get 2 vectors: 1 for user and 1 for item.
* we concat them to 1 and pass them to NN
* In Recommender systems we can use simple NNs as it is a regression task to predict ratings. not a classification one.
* psudomodel
```
u = input(shape=(1,))
m = input(shape=(1,)) # think in terms of NLP, seq len = T

# convert u and m to feature vectors
u_emb = Embedding(num_users,embedding_dim)(u)
m_emb = Embedding(num_movies,embedding_dim)(m)

# make it a single feature vector
x = Xoncat()(u_emb,m_emb)

# ANN
x = Dense(512,aciation='relu')(x)
x = Dense(1)(x)
```
* in this model TF Functional API is required
* 2 inputs appear in parallel
* seq layer dont support this
* NLP is the insiration on Recommender systems. Also Recommender systems also inspire NLP. 
* Matrix Factorization is a wll known algo in Recommenders. when word embeddings became popular, word2vec and Glove where the main algos to find word embeddings, bo th find word relations as vectors lie "king - man" = "gueen - woman"
* Glove is matrix factorization

### Lecture 61. Recommender Systems with Deep Learning Code

* We build a Recommender in Notebook
```
# More imports
from tensorflow.keras.layers import Input,Dense,Embedding,Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data is from: https://grouplens.org/datasets/movielens
# in case the link changes in the future
!wget -nc http://files.grouplens.org/datasets/movielens/ml-20m.zip
!unzip -n ml-20m.zip
df = pd.read_csv('ml-20m/ratings.csv')
df.head()
# we cant trust the userId and movieId to be numbered 0...N-1
# lets just set our own ids
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes
# df['new_user_id'] = df.apply(map_user_id,axis=1)
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes
# Get user IDs, movie IDs, and ratings as separate arrays
user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values
# Get number of users and number of movies
N = len(set(user_ids))
M = len(set(movie_ids))
# Get embedding dimension
K = 10
# Make a neural network
# User Input
u = Input(shape=(1,))
# Movie input
m = Input(shape=(1,))
# User embedding
u_emb = Embedding(N,K)(u) # output is (num_samples,1,K)
# Movie embedding
m_emb = Embedding(M,K)(m) # output is (num_samples,1,K)
# Flatten both embeddings
u_emb = Flatten()(u_emb) # now its (num_samples, K)
m_emb = Flatten()(m_emb) # now its (num_samples, K)
# Concatenate user-movie embeddings into a feature vector
x = Concatenate()([u_emb,m_emb]) # now its (num_samples, 2K)
# Now that we have a feature vector, its just a regular ANN
x = Dense(1024, activation='relu')(x)
# x = Dense(400, activation='relu')(x)
# x = Dense(400, activation='relu')(x)
x = Dense(1)(x)
# Build the model and compile
model = Model(inputs=[u,m],outputs=x)
model.compile(
    loss='mse',
    optimizer=SGD(lr=0.08,momentum=0.9),
)
# split the data
user_ids,movie_ids,ratings = shuffle(user_ids,movie_ids,ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

# center the ratings
avg_rating = train_ratings.mean()
train_ratings = train_ratings - avg_rating
test_ratings = test_ratings - avg_rating
r = model.fit(
    x=[train_user,train_movie],
    y=train_ratings,
    epochs=25,
    batch_size=1024,
    verbose=2, #goes faster when you dont print the progrss bar
    validation_data=([test_user,test_movie], test_ratings),
)
```

 * loss is not very good but is ok compared to other resources

## Section 9: Transfer Learning for Computer Vision

### Lecture 62. Transfer Learning Theory

* A very important topic in modern deep learning
* with Transfer learning we get:
  * higher start
  * higher slope
  * higher asymptote
* so better results in convergence overall
* Features are hierarchical and related, geting progressively more complex
* e.g CNNs start with simple lines and strokes common for many feats
* Features we find from one task may be used for another task. this is the concept of Transfer learning. mainly used in CV
* ImageNet is a large scale image dataset. 1m images 1k categories. because dataset is diverse, weights from this dataset can be used in many vision tasks
* even for new never seen images (microscope)
* Its not feasible for us to train on Imagenet, too costly
* Major CNNs tha won ImagNet contest come pretrained
* Pretrained models are already included in Tensorflow
* A 2-part CNN: feature trandformer part "body" and the ANN classifier "head"
* With transfer learning we keep the "body" and replace the "head"
* Head can be logistic regression or ANN. carsvstrucks can use logistic regression with sigmoid
* To do that we retrain a pretrained model, freezng the "body" layers and train only the "head"
* Advantage of transfer learning is tha we dont need a lot of data to build a state of the art model. we just need relevant data

### Lecture 63. Some Pre-trained Models (VGG, ResNet, Inception, MobileNet)

* VGG
  * named after a research group that created it. visual geometry group
  * like normal CNN but bigger
  * VGG16, VGG9 options (9 or 16 layers)
* ResNET
  * a CNN with branches(one branch is the identity function, the other learns the reidual)
  * Variations: ResNet50,ResNet101,ResNet152,ResNet_v2,ResNext
* Inception
  * Multiple convolutions in parallel branches
  * Instead of trying to choose different filter size (1x1,3x3,5x5 etc) try them all
* MobileNet
  * lightweight: tradeoff between speed and accuracy
  * used in less powerful machines (mobile,embedded)
* Preprocessing wehen using pretrained CNNs
  * our data must be formated like original train set of pretrained moel
  * we usualy work with RGB vals of [0.1] or [-1,1]
  * VGG uses BGR with pixel values centered nit scaled
  * just import `preprocess_input` from the sma module as the model
```
from keras.applications.resnet50 import ResNet50,preprocess_input
from kears.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
```

### Lecture 64. Large Datasets and Data Generators

* For learning DL, MNIST,CIFAR-10,SVHN are ok and also come as numpy arrays or csv
* In real world images come as imag files (JPEG,PNG ...)
* Images are also larger.
* VGG and ResNet are trained on ImageNet images resized to 224x224. so a hefty 150GB. this does not fit on RAM
* we use batching
* we assume 2 arrays. 1 for filenames, 1 for actual image as tables
* we assume batch size of 32
```
for i in range(n_batches):
  x = load_images(filenames[i:i+32])
  y = labels[i:i+32]
  model.train_on_batch(x,y)
```
* `gen = ImageDataGenerator()` automatically generates data in batches, does data augmentation, accepts preprocessing methods like from `preprocess_input`
* `generator = gen.flow_from_directory()` where we specify the target image size
* `model.fit_generator(generator)` is used instead of .fit
* use of genrators dictates a specific folder structure

### Lecture 65. 2 Approaches to Transfer Learning

* We go through 2 approaches for Transfer learning
* if we have a 100 layer bosy and 1 layer head, even if we freeze body weights, it takes time to compute the output prediction
* 2 part Computation
  * Part1: z=f(x) # pretrained CNN - slow
  * Part2: y_hat = softmax(Wx+b) # logistic regression - fast
```
for epoch in epochs:
  shuffle(batches)
  for x,y in batches:
    z = vgg_body(x)
    y_hat = softmax(z.dot(x) + b)
    gw = grad(error(y,y_hat),w)
    gb = grad(error(y,y_hat),b)
    # update w and b using gw and gb
``` 
* we take `z = vgg_body(x)` out of the loop
* before training we convert all input data into tabular matrix of feature vectors (Z)
* then we just have to run log reg on Z
* the problem is data augmentation that we get from generator works in iterations
* 1st approach: use data augmentation with `ImageDataGenerator` where entire CNN computation must be in the loop (slow)
  * Pros: possbly better for generalization
  * Cons: slow (input must pass from CNN)
* 2nd approach: precompute Z without data augmentation, only need to train log reg on (Z,Y) (fast)
  * Pros: data must pass only from 1 layer
  * Cons: possibly worse generalization

### Lecture 66. Transfer Learning Code (pt 1)

* we see 1st approach with a notebook example
```
# More imports
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
# Data from https://mmspg.epfl.ch/downloads/food-image-datasets/
!wget -nc https://lazyprogrammer.me/course_files/Food-5K.zip
!unzip  -qq -o Food-5K.zip
!ls Food-5K/training
# look at an image for fun
plt.imshow(image.load_img('Food-5K/training/0_808.jpg'))
plt.show()
# Food images start with 1, non-foo images start with 0
plt.imshow(image.load_img('Food-5K/training/1_616.jpg'))
plt.show()
```

* we reorganize and prepare data for CNN
```
!mkdir data
# make diectories to store the data Keras-style
!mkdir data/train
!mkdir data/test
!mkdir data/train/nonfood
!mkdir data/train/food
!mkdir data/test/nonfood
!mkdir data/test/food
# Move the images
# Note: we will consider 'training' to be train set
# 'validation' folder will be the test set
# ignore the 'evaluation' set
!mv Food-5K/training/0*.jpg data/train/nonfood/
!mv Food-5K/training/1*.jpg data/train/food/
!mv Food-5K/validation/0*.jpg data/test/nonfood/
!mv Food-5K/validation/1*.jpg data/test/food/
train_path = 'data/train'
valid_path = 'data/test'
# These images are pretty big andof different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200,200]
# useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
# useful for getting number of classes
folders = glob(train_path + '/*')
folders
# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()
```

* we work with the pretrained model
```
ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights = 'imagenet',
    include_top = False)
# freeze pretrained model weights
ptm.trainable = False
# map the data into feature vectors
# Keras image data generator returns classes one-hot encoded
K = len(folders) # number of classes
x = Flatten()(ptm.output)
x = Dense(K,activation='softmax')(x) #softmax is more generic it can work with multiclass classification
```

* we create and train the model using a generator
```
# create a model object
model = Model(inputs=ptm.input,outputs=x)
# view the structure of the model
model.summary()
# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range =0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)
batch_size = 128

# create generators
train_generator = gen.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# fit the model
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    steps_per_epoch=int(np.ceil(len(image_files)/batch_size)),
    validation_steps=int(np.ceil(len(valid_image_files)/batch_size)),
)
```

* results are good

### Lecture 67. Transfer Learning Code (pt 2)

* we try 2nd approach without generators on a notebook
* preparing of data is the same like before
* the code difers when we start with the pretrained model
* in this approach we dont hot encode targets
```
ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)
# map the data into feaure vectors
x = Flatten()(ptm.output)
# create a model object
model = Model(inputs=ptm.input,outputs=x)
# view the structure of the model
model.summary()
# create an instance of ImageDataGenerator
gen = ImageDataGenerator(preprocessing_function=preprocess_input)
batch_size = 128

# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
  class_mode='binary',
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
  class_mode='binary',
)
```

* we create our dataset
* we make empty datasets and then actual data passing them from the pretrained odel in batches
* generator is infinite loop. we need to brake manualy
```
Ntrain = len(image_files)
Nvalid = len(valid_image_files)

# Figure out the output size using model predict on random data
feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))
D = feat.shape[1]# populate X_train and Y_train
i = 0
for x,y in train_generator:
  # get features
  features = model.predict(x)
  #size of the batch (may not always be batch_size)
  sz = len(y)
  # assign to X_train and Y_train
  X_train[i:i+sz] = features
  Y_train[i:i+sz] = y
  #increment i
  i += sz
  print(i)

  if i >= Ntrain:
    print('breaking now')
    break
print(i)

X_train = np.zeros((Ntrain,D))
Y_train = np.zeros(Ntrain)
X_valid = np.zeros((Nvalid,D))
Y_valid = np.zeros(Nvalid)

```

* we do the same for valid data
```
# populate X_valid and Y_valid
i = 0
for x,y in valid_generator:
  # get features
  features = model.predict(x)
  #size of the batch (may not always be batch_size)
  sz = len(y)
  # assign to X_train and Y_train
  X_valid[i:i+sz] = features
  Y_valid[i:i+sz] = y
  #increment i
  i += sz
  print(i)

  if i >= Nvalid:
    print('breaking now')
    break
print(i)
```

* we check feature value range after the body of the network and standardize for head phase of NN
```
X_train.max(), X_train.min()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_valid2 = scaler.transform(X_valid)
```

* we try SKlearn built in log regression to test results
```
# Try the built-in logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train2,Y_train)
print(logr.score(X_train2,Y_train))
print(logr.score(X_valid2,Y_valid))
```

* Then we do LogReg in TF
```
# Do logistic regression in Tensorflow

i = Input(shape=(D,))
x = Dense(1,activation='sigmoid')(i)
linearmodel = Model(i,x)
linearmodel.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    merics=['accuracy']
)
# Can try both normalized and unnormalized data
r = linearmodel.fit(
    X_train, Y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_valid,Y_valid),
)
```

* this approach is faster and gives even better results than with data augmentation with fast convergence

## Section 10: GANs (Generative Adversarial Networks)

### Lecture 68. GAN Theory

* Ian Goodfellow is the inventor of GANs
* GANs combine existing parts
* NNs are nothing more than logistic regressions chained together
* CNNs and RNNs are NNs with shared weights
* GANs are really good at generating data (especially images) [Demo](https://www.thispersondoesnotexist.com/)
* GANs:
  * A system of 2 NNs: generator and discriminator
* Generator:
  * generates fake data out of noise. 
* Discriminator:
  * decides if its generator output is real or fake. 
  * trained on real data
* We need an objective. loss function to minimize
* In GANs we have 2 loss functions. one for generator and one for discriminator
* The discriminator must classify fakes from real images: binary classification (binary crossentropy)
* The Genrator Loss:
  * we treat GAN as a large NN
  * to train the Generator we freeze the discriminator layers so we only train Generator layers
  * we continue to use binary crossentropy (real=1,fake=0)
  * we pass fake images but use 1 for label. discrim is frozen only gen is trained. we encourage thus to classify these images as real
  * `JG= -1/NΣ[n=1->N]log(yhatn)` yhatn=fake image (output probaility of real given a fake image), target is always 1
* Generator input is noise. e.g a random vector of size 100
* These vectors come from the latent space
* Latent space is an imaginary space where generator believes images come from. it maps all possible images
* Generator learns to map areas of latent space in actual images
* Generator works in reverse of a Feature Transformer or Embedding Vector -> Image
* Pseudocode:
* 1: load in data `x,y=get_mnist()``
* 2: create networks.
```
d=Model(image,prediction)
d.compile(loss='binary_crossentropy',...) #1 fror real, 0 for fake
g=Model(noise,image)
```

* 3: combine d and g
```
fake_prediction=d(g(noise))
combined_model=Model(noise,fake_prediction)
combined_model.compile(loss='binary_crossentropy',...) # 1 is always target to fool the d
```

* 4: gradient descent loop
```
for epoch in epochs:
  # train dicriminator
  real_images = sample from x
  fake_image = g.predict(noise)
  d.train_on_batch(real_images,ones)
  d.train_on_batch(fake_images,zeros)
  
  # train generator
  combined_model.train_on_batch(noise,ones)
```
* Loss goes down and accuracy goes up? not in GANs. Discr and Gen are trying to outsmart each other

### Lecture 69. GAN Code

* we showcase GANs in TF with a notebook
```
# More Imports
from tensorflow.keras.layers import Input,Dense,LeakyReLU,Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
# Load in the data
mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()

# map inputs to (-1,1) to improve training
x_train,x_test = x_train/ 255.0*2 - 1, x_test/ 255.0*2 - 1
print('x_train.shape:', x_train.shape)
# Flatten the data
N, H, W = x_train.shape
D = H*W
x_train = x_train.reshape(-1,D)
x_test = x_test.reshape(-1,D)
# Dimensionality of the latent space
latent_dim = 100
# Get the generator model
def build_generator(latent_dim):
  i = Input(shape=(latent_dim,))
  x = Dense(256,activation=LeakyReLU(alpha=0.2))(i)
  x = BatchNormalization(momentum=0.8)(x)
  x = Dense(512,activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = Dense(1024,activation=LeakyReLU(alpha=0.2))(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = Dense(D,activation='tanh')(x)

  model = Model(i,x)
  return model

# Get the discriminator model
def build_discriminator(img_size):
  i = Input(shape=(img_size,))
  x = Dense(512,activation=LeakyReLU(alpha=0.2))(i)
  x = Dense(256,activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(1,activation='sigmoid')(x)
  
  model = Model(i,x)
  return model
  
# Compile both models in preparation for training

# Build and compilethe discriminator
discriminator = build_discriminator(D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002,0.5),
    metrics=['accuracy'])

# Build and compile the combined model
generator = build_generator(latent_dim)

# Create an input to represent noise sampe from latent space
z = Input(shape=(latent_dim,))

# Pass through generator to get an image
img = generator(z)

# Make sure only the generator is trained
discriminator.trainable = False

# The true output is fake, but we label them real!!
fake_pred = discriminator(img)

# Create the combined model object
combined_model = Model(z,fake_pred)

# Compile the combined model
combined_model.compile(loss='binary_crossentropy',optimizer=Adam(0.0002,0.5))

# Train the GAN

# Config
batch_size = 12
epochs = 30000
sample_period = 200 # every 'sample_period' steps generate and save some data

# Create batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

# Store the losses
d_losses = []
g_losses = []

# Create a folder to store generated images
if not os.path.exists('gan_images'):
  os.makedirs('gan_images')

# A function to generate a grid of random samples
# and save them to a file
def sample_images(epoch):
  rows,cols = 5,5
  noise = np.random.randn(rows*cols,latent_dim)
  imgs = generator.predict(noise)

  # Rescale images 0-1
  imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows,cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H,W),cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig('gan_images/%d.png' % epoch)
  plt.close()

# Main Training Loop
for epoch in range(epochs):
  ###########################
  ### Train discriminator ###
  ###########################

  # Select a random batch of images
  idx = np.random.randint(0, x_train.shape[0], batch_size)
  real_imgs = x_train[idx]

  # Generate fake images
  noise = np.random.randn(batch_size,latent_dim)
  fake_imgs = generator.predict(noise)

  # Train the discriminator
  # both loss and accuracy are returned
  d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
  d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
  d_loss = 0.5 * (d_loss_real + d_loss_fake)
  d_acc = 0.5 * (d_acc_real + d_acc_fake)

  #######################
  ### Train generator ###
  #######################

  noise = np.random.randn(batch_size, latent_dim)
  g_loss = combined_model.train_on_batch(noise,ones)

  # Save the losses
  d_losses.append(d_loss)
  g_losses.append(g_loss)

  if epoch % 100 == 0:
    print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}" )
  
  if epoch % sample_period == 0:
    sample_images(epoch)
    
plt.plot(g_losses, label='g_losses')
plt.plot(d_losses, label='d_losses')
plt.legend()
```

* we see that losses go hand in hand betwee gen and dic as they try to win eachother.
* the generated images thus become better and better

## Section 11: Deep Reinforcement Learning (Theory)

### Lecture 70. Deep Reinforcement Learning Section Introduction

* RL is very different from supervised (SL) or unsupervised learning (UL)
* We can think SL as a static function, we pass in data and get a prediction
* Even RNNs are a static method. we pass in a sequence and get a prediction, there is no actual concept of time
* In a simulation game (self driving car) we can take a screenshot at each timestamp and decide what to do
* SL is a function. RL is more like a control loop to achieve a goal
  * It has an Agent interacting with the Environment
  * Agent gets state and rewars from environemt. and triggers actions to the environment
* RL has the concept of time built in. it can plan for the future. a series of actions to reach its future goal
* In SL all input data must have a label. its humans who put the labels
* If machines put correct labels we dont need ML
* In an RL example Supervised driving its dificult for us to give the car specific action directions, and we cannot even do it for any frame
* RL uses goals instead of labels or targets. e.g park the car or exit a maze

### Lecture 71. Elements of a Reinforcement Learning Problem

* RL Terminology:
* The main objects are agent and environment
* In a Tic Tac Toe game
  * Env is the computer game implementing the game (even AI implementing Player 2). The Env can offer an API for interaction
  * Agent is another program that interfaces with the Game
* Episode, when interacting with the env. a series if actions take place and a goal is reached or not. this is an episode. like a game session or round
* An environment is episodic if it ends and can start again
* no relationship exists between episodes
* there are non episodic environments e.g stock market
* non episodic environments are called "infinite horizon"
* States, actions, rewards describe how agent and enviornmnet interact
* State: eg agent location, status of agent in e
* Action: eg agent move
* Reward: eg a number received at each step of a game. we van assign rewatds ourself to improve training
* we can set max steps
* we can set intermediate rewards
* we can subtract 1 for every step to give incentive to solve it faster
* we must receive a reward after each step
* agent will try to maximize reward
* State can get incredibly complex and constly but it should not.
* it should have the minimum to go back to a perfectly defined state in the past
* using screenshots as state is costly but resepmbles human perception
* image has no concept of movement though
* state in that case can be derived by looking at current and past observations. like subtracting images
* In DQN 4 sequential frames are used to represent state
* To describe spaces we need to understand sets.
* State Spaces: set of all possible states
* Action Spaces: set of all possible actions
* The Canonical Example of RL is the Gridworld
* State space is complicated while action space is usually simple
* if we use screen as state the state space is the screen resolutuon.
* images like time series are considered continuous-valued. the appear discreet because they are quantized
* action space can also e infinite (continuous actions)

### Lecture 72. States, Actions, Rewards, Policies

* Rewards are just numbers no need for encoding
* States and Actions are more complex and need encoding
* States can be discrete or continuous:
  * Discrete: tic-tac-toe game state
  * Continuous: Robot with sensors
* There is an analogy to SL (categorical vs linear)
* If targets are discrete we encode them to integers or even one hot encode them.
* If targets are continuous: we store them in vectors
* Images are stored in tensors
* In RL state usually is stored as a tensor with 1 or more dims
* Policy: what the agent uses to determine what action to perform (given a state)
* Policy yields an action given only the curent state. It does not use combos of past states or rewards,
* State however could be made up of past observations or rewards although its unconventional
* We can think of a Policy as a dictionary mapping (key value pair) key is the state, value is the action
```
def get_action(s):
  a = policy[s]
  return a
```
* We start by encoding states and actions then the policy
* for Gridworld
```
actions = [up, down, left, right]
policy = {
  (0,0): right,
  (0,2): right,
  (1,0): up,
  (1,2): up,
  (2,0): up,
  (2,1): right,
  (2,2): up,
  (2,3): left,
}
```
* states are represented as tuples, using integers to index an array is more efficient. indexing arrays than indexing dictionaries
* Encoding policy as dictionary has limitations (e.g infinite state space)
* is restrictive. it does not allow agent to explore
* policies must be stochastic / random
* a general way is to express them as probabilities. it allows agent some randomness.
* one method is called 'epsilon-greedy' vommone valie of ε=0.1
```
def get_action(s):
  if random() < epsilon:
    a = sample from action space
  else:
    a = fixed_policy[s]
  return a
```
* Considering Policies as probailistic supports the concept of Continuous (Infinite) State Spaces
* if state is a vector s (a D dimensional vector)
* and policy paramteters is a vector W (shape W = Dx|A|)
* W is the dimsnionality of state space by the size of action space
* we consider action spave as categorical* to output probabilities for a set of categories? it looks like a classification problem so we use softmax 
* for a given state s, we can calculate the probaility we should perform each action π(a|s) - softmax(WTs)
* to choose an action we sample from π(a|s) or use argmax
* this is a linear model approach, we can use a Neural Network instead
* Is it possible for an agent to make an intelligent decision using only current state? does he need a target like in SL?? In RL training gives Agent experience and the ability to plan ahead. it will try to maximize future rewards

### Lecture 73. Markov Decision Processes (MDPs)

* We need a framework in RL to solve problemns in a consistent way
* We can use it to define the problem accurately and find a solution
* The main assumption in RL is the Markov assumption. It is uused in the context of sequence modeling
* It says that next state is dependent on present state and not previous ones
* There is work on RL hat does not use Markov assumption
* Markov Assumption: State at time t depends only on the state at time t-1
  * p(st|st-1,st-2...,s0) = p(st|st-1)
  * State can be simple or complex. it is on our control to define it
* Markov Decision Process (MDP): it describes a RL system
* Markov Assumptions: we saw it in the state concept
* MDP: we describe the environment with the state-transition brobability
* The probability at ariving at a state at time t+1 and getting the reward at time t+1 giving the state and action at time t p(St+1,Rt+1|St,At) or p(s',r|s,a)
* p(s',r|s,a) is the most general form of State-Transition Prob. sometimes reward is deterministic so we dont need to model it with probability
  * State-trans-prob: p(s'|s,a)
  * Reward function: R(s,a,s') or simply R(s),R(s')
* In complex games the state space is infeasible to enumerate. how we will calcualte the State-Transition Probability?
* In Q-Learning Algorithm, State-Trans_prob is not used at all
* So why bother with MDP??? it is a step towards building practical RL solutions
* simple Games involving physics (inverted pendulum) are modeled using physics laws that are deterministic not probabilistic. why we need State-Trans=Prob??
* State may not capture the wholw info on a game. e.g if there is another player. you cant deterministicaly predict his move
* also in physics there is chaos therory
* State-Transition-Probability works as Environment Dynamics
* MDP offers a system where Agent and Environment are represented as probabilities that interact with each other. this allows us to describe a RL system mathematicaly and solve the equations.
  * Agent: π(a|s)
  * Environment: p(σ',r|s,a)

### Lecture 74. The Return

* Agents goal is to maximize Rewards
* Rewards maay be structured differently in different environments
  * might be awarded per step or per episode
* So what it means to maximize the reward? Agent wants to maximize the SUM of future rewards. this makes the agent to plan ahead for the future an is not deadlocked on the next step
* In rela world working towards a big reward in the future might involve negative rewards in the process. if we go for the small rewards we lose the big reward
* Sum of future rewards is called The Return (G). Some call it the 'utility' 
* G(t) = Σ[τ=1->Τ-t]R(t+τ) = R(t+1)+R(t+2)+...+R(T)
* If we have an Infinite Horizon MDP (game with no end) will our reward be infinite? no if we use discounting (discounting factor γ). Future rewards are discounted by a factor. γ is usually close to 1. it serves the purpose that the further we go in the future the harder is to predict rewards
* G(t) = Σ[τ=1->Τ-t]γt-1R(t+τ) = R(t+1)+γR(t+2)+...+γΤ-t-1R(T)
* Return can be defined recursively: G(t) = R(t+1) + γG(t+1)
* R is Reward, G is return

### Lecture 75. Value Functions and the Bellman Equation

* We will finally express a generic RL problem with an equation from which we can derive a solution
* We start with Expected Value. A.K.A Mean Value AKA Average
* Expected value is not what we expect to get its just an artificial value
* Expected Value (weighted sum):
  * Discrete Random Vals: E(X) = Σ[κ=-inf->inf]p(x=k)k
  * Continuous Random Val: E(X) = Integral[-inf->inf]xp(x)dx
* Why expected values are important?
  * rewards and thus return is a random variable
  * environment dynamics (state-trans-prob) and policy are probabilities as well
* it makes sense not to think about plain return but Expected Value of Return. then we will try to maximize the expected sum of future rewards
* Value Function = Expected Return = Expected sum of future Rewards
* For given state s at time t: V(s) = E(Gt|St=s) = E(Rt+1 + γRτ+2 + ... | St=s)
* Return is expressed recursively and so is Value Function: V(s)=E(Rt+1 + γV(s') |St=s)
* We now have all the pieces to build the Bellman Equation that expresses an MDP system. it is the Value Function expressed in terms of the probability equations for STP and Policy `V(s)=Σ[a]Σ[s']Σ[r]π(a|s)p(s',r|s,a){r+γV(s')}`
* Bellman Equation is the cenerpiece of al upcoming RL solutions:
  * it is built using probability math equations
  * π(a|s) and p(s',r|s,a) express different physical processes, the Policy = Agent and State-Trans-Prob = Environment respectively
* We finaly have a problem to solve...
* There are multiple possible policies, good anf bad. 
* To tell a good from a bad policy we check their Value Functions Vπ(s) = value function for policy π
* Finding a value function for a given policy is called the Prediction Problem
* if we know  π(a|s) and p(s',r|s,a) we can express Bellman Equation as a system of linear equations. for 3 states it can be expressed as
```
c11V(s1)+c12V(s2)+c13V(s3)=b1
c21V(s1)+c22V(s2)+c23V(s3)=b2
c31V(s1)+c32V(s2)+c33V(s3)=b3
```
* we can use `np.linalg.solve()` and find the V(s1),V(s2),V(s3).
* So if we know Policy and State-Trans_Prob we can solve for Value Function using only Linear Algebra

### Lecture 76. What does it mean to “learn”?

*  In RL there are 2 main tasks
  * Prediction problem: Given a policy π, find the corresponding Value function V(s)
  * Control problem: Find the optimal policy π* that yields the maximum V(s)
* Value Function V(s) is also called State-Value function
* Q(s,a) is the Action-Value functio (known as Q-table) or Bellman Equation for Q
  * we dont need to sum over action as it is given
  * `Q(s,a) = E(r+γV(s')|St=s,At=a)`
  * `Q(s,a) = Σ[s']Σ[r]p(σ',r|s,a){r+γV(s')}`
* How much space is needed to store the Value Functions V(s) and Q(s,a)
  * We assume a finite set of discrete states |S| and actions |A|
  * V(s) can be stored in an array of size |S| (Linear Complexity O(n))
  * Q(s,a) can be stored in a 2-D array of size |S|x|A| (Quadratic Complexity O(n2))
* The optimalpolicy is the one that maximizes the value for all states
  * π1 >= π2 if Vπ1(s)  > Vπ2(s)  for all s in |S| 
* Best Value Function: `V*(s) = maxπVπ(s)` for all s in |S|
* Best Policy derived from Value Function: `π* = argmaxπVπ(s)` for all s in |S
* Βest Action-Value Function: `Q*(s,a)=maxπQπ(s,a)` for all s in |S|
* Best Value Function derived from Action Value Function: `V*(s) = maxaQ*(s,a)` for all s in |S|
* If we find the optimal action value it is very easy to choose the best action given the state: `a*=argmaxa:*(s,a)`. it is fust a dictionary lookup
* * A simple way to find the optimal policy is a Naive Search
```
policies = enumerate_all_possible_policies()
best_policy = None
best_value = {s1: -inf, s2: inf, ..., sN: -inf}
for policy in policies:
  current_value = evaluate(policy)
  if current_value > best_value
    best_value = current_value
    best_policy = policy
# now best_policy stores best policy
```

* `evaluate(policy` we saw how to do it when we know Agent π(a|s) and Environment p(s',r|s,a)
* enumerating all possible policies is simple to understand but impractical

### Lecture 77. Solving the Bellman Equation with Reinforcement Learning (pt 1)

* W saw there are 2 types of problems
  * Prediction problem: find V(s). Ιf we know all probabilities (π(a|s) and p(s',r|s,a)) it is a simple linear algebra problem
  * Control problem: find π*, loop through all possible policies, return the one that yields the best V(s)
* Except from trivial games its impossible to know all the environment dynamics (posibilities). All we can do is repeat the episode 1000s of times
* But also state space can be extremely large
* So usually we cannot rely on enumerating all states and figuring out all strate-trans probabilities
* Control Problem: Can we enumerate all possible policies. their number is |S|^|A|. it grows exponentially
* The solution to all this is to work with the Expected Value (Mean). To calculate it we must know the distribution
* We can estimate the mean with the sample mean. `E(X)~~(1/N)Σ[i=1->Ν]xi` where x are the sampes, when N->inf, sample mean equals actual mean
* Sample mean is extensively used in experimental sciences
* The value function is the Expected Return (we saw that). We can sample many returns to estimate the value of each state
  * G(t) is generic random variable. return at time t
  * g(i,s) means a sample of return. the ith time we reacheched state s (ith sample return from state s)
  * `V(s)=E(Gt|St=s)~~(1/N)Σ[i=1->N]g(i,s)`
* How samples are obtained?? to sample from a std distribution `np.random.randn()`. But return is different in every episode, even with same policy and environment as botha re probabilistic
* To Calculate Rewards from Samples We use the Monte Carlo approach (Monte Carlo sampling)
  * given a policy we try to find a value function
  * the idea: playing an episode yields a series of states and correcponding rewards
  * key point: go backwords and use the recursive definition
  * gT=0 as there are no future states so V(sT)=0
  * gT-1=rT
  * gT-2=rT-1+γrT=rT-1+γgT-1
  * gT-3=rT-2+γgT-2
  * gt=rt+1 + γgt+1
```
states, rewards = play_episode_using_policy

returns = []
g = 0
returns.append(g)
for r in reversed(rewards):
  g = r + gamma*g
  returns.append(g)
```
* now we run it 100s of times to collect samples
```
policy = ... # given as input
sample_returns = {} # state -> list of returns
for i in range(num_episodes):
  states, rewards = play_one_episode(policy)
  returns = ...calculate as previously discussed...
  for s,r in zip(states, returns):
    sample_returns[s].append(r)
```

* for each state take the average return of the samples
```
# calculate the average return
V = {}
for s, g_list in sample_returns.items():
  V[s] = np.mean(g_list)
```
* with theis approach we cannot ensure we encountered every possible state during episodes

### Lecture 78. Solving the Bellman Equation with Reinforcement Learning (pt 2)

* For prediction we want V(s)
* For control we want Q(s,a), it allows us to select the best action to perform `a*=armaxaQ*(s,a)`
* Policy iteration and Policy Improvement
  * Given a policy, we can use Monte Carlo to evaluate the Value Function V(s) or Q(s,a)
  * Given the aaction value Q(s,a) we can choose the best action
  * We see the dependency and the loop formed => find Q(s,a) given π "evaluation", find π as argmax Q(s,a) "improvement"
* Ints proven with math that this process leads to monotonic improvemnt in thepolicy
* By repeating this process untill convergence we get the optimal prolicy
```
Q = random. policy = randomfor i in range(num_eposodes):
  Q = evaluate(policy) # policy evaluation step
  for s in Q.states(): # policy improvement step
    policy[s] = argmax{ Q(s, :) }
```
* Earlier we looked at evaluating V(s) using Monte Carlo
* To evaluate Q(s,a) we need to keep track apart from states and rewards on actions too
* We end up with triples: {(s1,a1,r1),(s2,a2,r2)...}
```
policy = ... # given as input
sample_returns = {} # (s,a) -> list of returns
for i in range(num_episodes):
  states, actions, rewards = play_one_episode(policy)
  returns = ...calculate as previously discussed...
  for s,a,r in zip(states,actions,returns):
    sample_returns[s,a].append(r)
```
* Calculate Q as mean
```
Q = {}
for s,a,q_list in sample_returns.items():
  Q[s,a] = np.mean(q_list)
```
* Our solution works but is not ideal
  * V(s) must store |S| vals, Q(s,a) |S|x|A| vals
  * Monte Carlo sampling gives better results the more samples we collect
  * with Q we have more vals to estimate thus we need more samples
  * we use nested loops in our calculations (slow)
* Value iteration works better:
  * Its not elegant but it works
  * In evaluation step, instead of playing muliple episodes to obtain a Monte Carlo estimate of the value, just play one episode
  * we ll only get a single series of states, actions, returns
  * use this to update a single running copy of Q(s,a) and policy
```
Q = random, policy=random
for i in range(num_episodes):
  # replace policy evaluation with one episode only
  states, actions, rewards = play_one_episode(policy)
  returns = ...calculate as previously discussed...
  # update Q(s,a) and policy
  for s,a,q in zip(states,actions,returns):
    Q(s,a) = update Q(s,a) with the latest return q
  for s in Q.states(): # policy improvement step
    policy[s] = argmax{ Q(s,:)}
```
* ` Q(s,a)` should be the mean of asamples. is it efficient? NO
* Taking the sum of N samples is O(N)
* instead we can calculate the Nth sample mean from the N-1th sample mean
```
XavgN = (1/N)Σ[i=1->N]xi=...= ((N-1)/N)ΧavgΝ-1 + (1/N)xN = XavgN-1 + 1/N(xN - XavgN-1)
```
* This looks like gradient descent and in fact it is `XavgN = ((N-1)/N)ΧavgΝ-1 + (1/N)xN`
  * `XavgN-1` is the prediction
  * `xN` is the target (nth collected sample)
  * `1/N` is the learning rate (decays over time)
* We rewrite the equation in terms of Q(s,a) and g (sample) `Q(s,a)=Q(s,a)+1/N(g-Q(s,a))` which is an assignemnt
* We only have 1 copy of Q. all the samples we collect on each iteration come from different policies (policies are updated at each step)
* The samples we collect do not come from the same distribution... Problem. Our intuition says that old samples come from older policies thus matter less. than new samples
* To workaroud our finding instead of using 1/N as learning rate which yields a standard (equaly weighted average) we use a constant learning rate that yields an exponentialy decaying average
* `Q(s,a)=Q(s,a) + η(g -Q(s,a))`
* Pseudocode
```
Q = random, policy=random
for i in range(num_episodes):
  # replace policy evaluation with one episode only
  states, actions, rewards = play_one_episode(policy)
  returns = ...calculate as previously discussed...
  # update Q(s,a) and policy
  for s,a,q in zip(states,actions,returns):
    Q(s,a) = Q(s,a) + learning_rate * (g - Q(s,a))
  for s in Q.states(): # policy improvement step
    policy[s] = argmax{ Q(s,:)}
```

### Lecture 79. Epsilon-Greedy

* Policy as Distribution
* Advantages:
  * Entire MDP is just 2 probabilities: π(a|s) aνδ p(s',r|s,a)
  * in order to find the best action we need to know the result of performing these actions (we need to collect the samples)
* Problem: currently we take argmax to determine action. Q does not change much
  * say we have 3 actions: Q(s,a1),Q(s,a2),Q(s,a3). we init them to 0,0,1 and all rewards are +ve
  * during game Q(s,a3) is updated to 2 after few episodes. we will never know the true values of Q(s,a1) and Q(s,a2) as we can never use them
* It all boils down to the explore-exploit dilemma. we must balance exploration and exploitation
  * explore: collect more data to determine which policy is truly the best
  * exploit use the best policy with minimum cost to get maximum turnover
* In RL this is called Epsilon-Greedy:
  * we have a small probability (a hyperparam called ε) of choosing a random action
  * otherwise we will perform the greedy action (argmax over Q(s,:))
```
# Instead of a=argmax(Q[s,:])
# use 
if random() < epsilon:
  a = action_space.sample()
else:
  a = argmax(Q[s,:])
```

### Lecture 80. Q-Learning

* To get to Q-Learning Algorithm we had to:
  *  Define relevant terms: agent,environment,state, action,reward
  *  Define the math structure: MDP(Markov Decision Process): Agent =>[Take Action based on current state]=> Environment, Agent<=[get reward, new state]<=Env
  *  Solve Prediction Problem: find the value function given a policy
  *  Solve Control Problem: find the optimal policy in a given env
  *  If we know the probabilities the problem is easy, if not we use Monte Carlo and samples
* Monte Carlo Limitation
  * To calculate results we have to wait till episode is over. return is the sum of rewards until the end of episode
  * in games with infinite horizon or very long episonds we cannot use Monte Carlo
  * even if episode ends MC is not ideal as agent performs suboptimal for long time untill improvement
* Teporal Difference Methods are a solution to that
  * we make use of the recursive structure of the return  `G(t) = R(t+1) + γΓ(t+1)`
  * Monte Carlo MC is an approximation to the expected value problem used in Bellman Equation
  * Temporal Difference (TD) is an approximation to MC
* Monte Carlo Update Trick
  * we converted the usual expression for sample mean to a gradient descent expression (using pure algebra) Q(s,a) = Q(s,a) + η(g-Q(s,a))
  * instead of keeping old values around and adding them every time, we can calc the new estimate from the old estimate
* The Gradient Descent Perspective
  * We test the theory by defining J as the squared error between a sampe (g) and prediction V(s) `J=(g-V(s))^2`
  * we calculate the gradient (we can ignore the 2 as it is absorved in learning rate)
  *  V(s) <- V(s) - ηGRADV(s)J, GRADV(s)J = -2(g - V(s)), V(s) <- V(s) + η(g-V(s))
* There is no difference between Gradient Ascent and Descent expressions
  *  V(s) <- V(s) + η(g-V(s)) (ascent when derived from sample mean)
  *  V(s) <- V(s) - η(V(s)-g (descent when derived from GD or loss)
* Combine the Ideas:
  * 1: updating the value function using the exponentialy decaying average is the same as gradient descent
  * 2: the return can be defined recursively
* We use gradient descent but now using the estimate return not the real one (g) as target
* we then collect the next reward (r) and estimate the rest V(s')
* instead of `g=r+γr'+γ^2r''+....` we use `g=r+γg' ~= r+γV(s')`
* what we gain? we now have to wait just one step before updating the model not a whole episode
* we call `r+γV(s')` a bootstrapped estimate of the return
* Pseudocode like openAIGym
```
#given: env,policy
V = random
for i in range(num_episodes):
  s = env.reset()
  done = false
  while not done:
    a = policy[s]
    s',r, done = env.step(a)
    
    # the big update
    V(s) = V(s) + learning_rate *  (r+gamma*V(s')-V(s))
    
    # important: update the current state
    s = s'
# by now V(s) has converged (can also check it ourselves)
```
* There is an oddity in Temporal Diffeerence Learning
* In `V(s) = V(s) + learning_rate *  (r+gamma*V(s')-V(s))`
  * `r+γV(s')` is the target
  * `V(s)` is the prediction
  * in SL we are given the target as part of the dataset
  * here we predict it. part of it is given `r` but `V(s')` is a model prediction like `V(s)`
  * So the update method is not true gradient descent but Semi-Gradient Descent
* We learned how to apply the TD method to the prediction problem
* Now we will focus on the Control Problem 'Q-Learning'
* We will finally solve the Control Problem using the Q-Learning Algorithm
  * we look at Q rather than V
  * we are mostly interested in the innermost part of the loop
  * we must do 2 things: 1. Choose an action, 2. Update Q 
```
def choose_action(Q,s):
  if random() < epsilon:
    return random action
  else:
    return argmax{Q(s,:)}

Inside the main loop:
# choose action and execute
a = choose_action(Q,s)
s',r,done = env.step(a)
# update Q
y = r + gamma * max_a'{ Q(s',a')} # target
Q(s,a) = Q(s,a) + learning_rate * (y - Q(s,a))
```

* when calculate a target it does not matter what action we choose next (a')
* we assume we will take the greedy action and get the max over Q given the state s'
* Advantage:
  * we dont have to wait until we determine the next action to update Q
  * it makes Q-Learning an off-policy algorithm (the update might not match the action taken)
* So we can freely explore while the algorithm will update the Q tBLW as if we had acted greedily
* Pseudocode
```
def choose_action(Q,s):
  if random() < epsilon:
    return random action
  else:
    return argmax{Q(s,:)}

env = ... #given: some environment object
Q = random
for i in range(num_episodes):
  s = env.reset()
  done = false
  while not done:
    a = choose_action(Q,s)
    s',r, done = env.step(a)
    
    # update Q
    y = r + gamma * max_a'{ Q(s',a')}
    Q(s,a) = Q(s,a) + learning_rate * (y - Q(s,a))
    
    # important: update the current state
    s = s'
# by now Q(s,a) has converged (can also check it ourselves)
```

### Lecture 81. Deep Q-Learning / DQN (pt 1)

* Deep Q-Learning = Q-Learning using Neural Networks
  * Agent gets Reward from Env
  * Agent Observs state from Env
  * Agent takes action to Env
  * In Agent from state as Input a DNN outputs the optimal policy based on which action is taken

* Q-Learning involves finding the optimal Q*(s,a) and the corresponding optimal policy π*
* so far: 
  * states/actions are categorical (encoded by integers starting at 0)
  * most of what we discussed was in the context of tabular methods (States as rows, actions as columns, Q(s,a) as cells)
* If our State or Action Space is continuous or infinite, storing a Q table is non viable
  * we can use the binning method to force a discrete finite set of states/actions. like cluster continuous vals itno categories (bins)
  * a more flexible approach is to use machine learning (function approximation)
  * these are called approximaton methods compared to  tabular methods
* In DQN (Deep-Q-Learning) the state space is possible infinite but the action space discrete
* suppose our state is a vector s and we have 2 actions a0 and a1
  * Q(s,a0) = w0Ts+b0
  * Q(s,a1) = w1Ts+b1
* we can combine weights and bias into vectors Q(s,:) = WTs+b
  * W is of shape DxK
  * b is a vector of length K
  * : means select all elements in this dimension
* In Inverted Pendulum
  * State has 4 components: s = (x,dx/dt,θ,dθ/dt)
  * action is -1 or +1  force to the cart
  * For Q(s,a) shape(W)=(4,2), shape(b)=(2,)
  * For V(s) shape(W)=(4,2) b is scalar
* when we discussed TD and MC learning we updated V(s) and Q(s,a) directly as they were just table values
* Now that tey are vectors we need to update w and b
* The treat finding V(s) as a Supervised Learning Problem
  * Target is `r+γV(s')` the estimation of the return, if its terminal state, target = r
  * Prediction is `V(s) = WTs+b`
* this is a regression problem. we need to get the square root eeror and do gradient descent to update params 
* although V(s) and V(s') depend on params w,b we differentiate with respect to V(s) only
```
J = (r + γV(s') - V(s))^2
δJ/δw = [V(s) - (r + γV(s'))]s
δJ/δb = V(s) - (r + γV(s'))
```
* The Pseudocode for the prediction problem becomes
```
env = ... #given: some environment object
policy = ... # given: some policy
w,b = random
for i in range(num_episodes):
  s = env.reset()
  done = false
  while not done:
    a = policy(s)
    s', r', done = env.step(a)
    # the big update
    V(s) = w.dot(s) + b
    V(s') = w.dot(s') + b
    w = w - learning_rate * (V(s) - (r + gamma*V(s')))s
    b = b - learning_rate * (V(s) - (r + gamma*V(s')))
    # important: update current state
```
* Approximating V(s) is not by itself useful
* What if we use V(s) to help us choose an action. 
* but how? we know that normally we would take the argmax of Q(s,a) over all a
* V(s) is not indexed by any action. only state
* A suboptimal approach
  * Pretend we have a special env, where we can try an action and go back to the previous state
  * In real envs we cannot just go back to state s
* A realistic approach is to use Q(s,a) instead
  * Target is: `y = r + γmaxa'Q(s',a')`, if its terminal its just r
  * Prediction is Q(s,a)
  * Params to update are still W and b
* Updating Q(s,a):
  * Only components of W and b corresponding to the action taken (a) are updated
  * Its equivalent to saying that the error for actions not taken is zero
```
J = (r + γQ(s',a') - Q(s,a))^2
δJ/δwa = [Q(s,a) - (r + γmaxa'Q(s',a'))]s
δJ/δba = Q(s,a) - (r + γmaxa'Q(s',a'))
```
* Q-Learning when using approximation methods, we use epsilon-greedy for exploration
```
def choose_action(s):
  if random() < epsilon:
    return random action
  else:
    Q(s,:) = s.dot(W) + b
    return argmax(Q(s,:))

env = ... #given: some environment object
w,b = random
for i in range(num_episodes):
  s = env.reset()
  done = false
  while not done:
    a = choose_action(s)
    s', r', done = env.step(a)
    # Update Q
    y = r + gamma * max_a'{Q(s',a')}
    W[a] = W[a] - learning_rate * (y - Q(s,a))s
    b[a] = b[a] - learning_rate * (y - Q(s,a))
    # important: update current state
    s = s'
```

### Lecture 82. Deep Q-Learning / DQN (pt 2)

* Is it not trivial to replace the Linear Regression with a NN?
* we use the princple that all ML interfaces are the same
* this is better because we dont know if Q is linearly dependent on s
* conceptually we only need to continue using Gradient Descent
* Θ represents all params of the model here
* Gradient will be complicated, so we can use automatic differentiation
```
J = (r+γmaxa'Q(s',a') - Q(s,a))^2
δJ/δθ = [Q(s,a) - (r+γmaxa'Q(s',a'))]δQ(s,a)/δθ
```
* Τraining NNs is inherently unstable, we can not just chose any hyperparam and expect good results.Sometimes cost explodes, or model wont fit
* Linear Regression is much more stable than NNs
* TD Learning is by itself unstable. we do grad descent but target is not a real target and so loss is no real loss and gradient no real gradient. all are approximations
* When we combine both we get sthing that barely works if it works at all
* there are a number of approaches to Deep Q-Learning (DQN).we ll see just 1
* Its the experience replay buffer / experience replay memory
* what we used in DNNs was stochastic gradient descent SGD (one sample at time)
* what we want to do is batch gradient descent that more stable working with multiple samples at a time
* In TF, people call it SGD even when they work with batches
* we can try it using different batch sizes (1,32,128 ,, N) on existing SL models using the full dataset as batch , our loss should decrease monotonically
* Sample Collection Hints
  * not good to have sequential samples correlated when doing GD
  * that why we shuffle data on each epoch
  * we ll see how to randomize the replay buffer to avoid seeing the tuples in the order they were encountered again and again
* Replay Buffer can work as a python list storing tuples (s,a,r,s',done) as we encounter them. these tuples are called transitions
* its an 1 line addition to our loop, everytime we take a step in environment we add a transition tuple to our replay buffer
* At some point the transitions in the replay buffer are stale snd correspond to a policy very different from the currrent one 
* when adding new transitions we remove old ones when buffer is full
```
s',r,done = env.step(a)
replay_buffer.append((s,a,r,s',done))
if len(replay_buffer) > max_size:
  replay_buffer.pop(0)
```
* When we want to updatte
  * sampe a batch of transitions from repolay buffer
  * populate inputs and targets in our DNN
  * do one step of DG on the data
```
batch = random.sample(replay_buffer)
inputs = [], targets = []
for s,a,r,s',done in batch:
  inputs.append(s)
  y = r + gamma *  max)a'{Q(s',a')}
  targets.append(y)
model.train_on_batch(inputs,targets) # one step of GD
```
* Pseudocode for Deep Q-Learning
```
env = ... #given: some environment object
replay_buffer = []
model = Model() # a neural net with random initialization
for i in range(num_episodes):
  s = env.reset()
  done = false
  while not done:
    a = choose_action(s)
    s', r', done = env.step(a)
    update_replay_buffer(s,a,r,s',done)
    # Update Q
    train() # as described earlier
    # important: update current state
    s = s'
```

### Lecture 83. How to Learn Reinforcement Learning

* When we learn SL and UL many courses avoid implementing the models (they use them ready from SKlearn)
* We end up not even knowing what we don't know. we cant fix problems in our code or results
* Thats maybe Ok if we just ant to use APIs
* In RL there are no APIs (not yet)
* If we dont know to implement ML algorithms we are in deep shit.
* To learn RL for real we need a full course or multiple ones. learn and experiment with tabular RL before moving to approximation methods
* Learn about 3 basic approaches
  * Dynamic Programming
  * Monte Carlo
  * Temporal Difference
* then graduate to approximation methods with linear models
* then apply deep learning
* Deep RL is very hard to get right even for expert programmers (it can take 1 year for a single game)

## Section 12: Stock Trading Project with Deep Reinforcement Learning

### Lecture 84. Reinforcement Learning Stock Trader Introduction

* When people think about applying ML to the stock market:
  * They think about predicting the value of a stock
  * or even the direction. (go up or go down)
* this info doesnt make it happen
  * we must sit down and make the trade from our computer
  * it is not even sure we will act on themodel predictions
* (Un)Supervised Learning makes the prediction but does not take action
* RL makes predictions and takes actions in the environment we provide (the action it believes will maximize reward)
* Stock prices we can think as Time Series. it sounds as a prediction problem for RNNs rather than RL
* RL perspective
  * consider we use a stock trading API `api.buy('GOOG',10)`if each share is $50, we are -$500 in our account and we own 10 shares of GOOG
  * if we use `api.sell('AAPL',5)` and APPLE shares are $30 we are +$150 and we lost 3 shares of APPLE
* We can treat the api calls as actions and the stock prices at any given time as state
* Environment is the Stock Market. there is inherent randomness.we cannot predict tomorrows stock prices
* we have areal RL problem
  * actions = buy/sell/hold
  * state = stock prices/#shares owned/account balance
  * reward = function of portfolio value gained/lost
* we as agents follow the rule "buy low/sell high" maybe if we dont need the money and
* but in real world we make choices without knowing the future..

### Lecture 85. Data and Environment

* we will work with historical stock data (as a simulation)
* we build the environment in OpenAI Gym style
```
env = MYTradingSimulationEnv()
done = False
state = env.reset() # bring us back to init state
while not done:
  action = get_action(state) # could come from our agent

  # perform the trade, 'info' contains portfolio value
  next_state, reward, done, info = env.step(action)
  
  state = next_state
```
* State Variables:
  * consider a fixed window of past and current stock prices
  * do we have cash to buy?
  * given our portfolio, is it worth selling to get cash for trades?
  * we need to keep it simple
  * we get ideas from "Practical Deep RL Approach for Stock Trading" paper. they use DDPG (advanced algo)
  * According to paper our state will have 3 parts: 1) how many shares of each stock we own 2) current price of each stock 3) pure cash we have
  * For N stocks our state vector has size 2N+1
* Actions:
  * many options
  * for any stock: buy/sell/hold
  * for a simple env we consider 3 stocks AAPL,MSI,SBUX. we have 3^3=27 osssibilities e.g [sell,sell,sell]
  * but we can trade >1 stocks..
  * to simplify we will ignore transaction costs, when we sell we will sell all our shares, when we buy we will buy as many as possible given our cash
  * still its complicated (Knapsack problem), we want to avoid it as its NP hard
  * when we buy multiple stocks we will do it round robin. loop through every stop, buy 1 till money dries out,
  * also we will sell before buy to have max cash
  * all these conventions are to get down to 27 actions
  * one action in our env will involve performing all steps at once
* Reward:
  * simple. the change in value of our portfolio from one step (state s -> state s') + cash
```
s = vector of # shares owned
p = vector of share prices
c = cash
potfolio value = sTp+c
```

### Lecture 86. Replay Buffer

* How to implement Replay Buffer / Replay Mem efficienty?
* Start with naive approach. just a Python List storing 5element tuples (s,a,r,s',done) aka 'transitions'
* we will add them as we get them. buffer will have a max size. when we hit it throw oldest val
* during train we will grab minibatches from buffer for training (do grad descent)
```
minibatch = random.sample(buffer, batch_size)
targets = []
# calculate input-target pairs
# for each sampe in the minibatch (s,a,s',r,done)
for (s,a,s',r,done) in minibatch:
  if done:
    target = reward
  else: 
    target = reward + gamma * max[a']{Q(s',a')}
  targets.append(target)
```
* we prefer numpy batch operations than for loops for performance
* with appending new tuples and poping old ones the lists grows without stop and it leads to memory leaks
* to avoid it we need to implement our own replay buffer
* we will preallocate arrays
  * States (NxD)
  * Actions (N)
  * Rewards (N)
  * Next states (NxD)
  * Done flags (N)
* we will never allocate more arrays or remove existing
* we will use a pointer to tell us where to store the next value
* our buffers will be circular. when we reach the end go back to start
* when buffer is not yet full we must keep track of size

### Lecture 87. Program Design and Layout

* 2 modes of operation: train and test
* train data must come before test data
* top level code organization
```
env = Env()
agent = Agent()
portfolio_values = []
for _ in range(num_episodes):
  val = play_one_episode(agent,env)
  portfolio_values.append(val)
plot(portfolio_values)
```
* we dive in play one episode function
```
def play_one_episode(agent, env):
  s = env.reset()
  done = False
  while not done:
    a = agent.get_action(s)
    s',r,done,info = env.steps(a)
    if train_mode:
      agent.update_replay_memory(s,a,r,s',done)
      agent.replay() # sample batch and do gradient descent
    s = s'
  return info['portfolio_val']
```
* our data is not normalized
* different parts of state have different ranges
* # shares owned, stock prices, cash
```
...
state = env.reset()
state = scaler.transform(state)
...
next_state,reward,done,info = env.step(action)
next_state = scaler.transform(next_state)
...
```
* environment object
```
class Environment:
  def __init__(self,stock_prices,initial_investment):
    self.pointer = 0
    self.stock_prices = stock_prices
    self.initial_investment = initial_investent
  
  def reset(self):
    # reset pointer to 0 and return initial state
  
  def step(self,action):
    # perform the trade, move pointer
    # calculate reward, next state, portfolio value, done
```
* Agent
```
class Agent:
  def __init__(self):
    self.replay_buffer = ReplayBuffer()
    self.model = Model()
  
  def update_replay_buffer(self,s,a,r,s',done):
    # store in replay buffer
  
  def get_action(self,s):
    # calculate Q(s,a), take the argmax over a
  
  def replay(self):
  # sample from replay_buffer, make input-target pairs
  # model.train_on_batch(inputs,targets)
```
* in train mode we replay the episode again and again to maximize return (store states/actions/rewards => update Q(s,a))

### Lecture 88. Code pt 1
### Lecture 89. Code pt 2
### Lecture 90. Code pt 3
### Lecture 91. Code pt 4

* we can run our code locally or in colab. locally its should run faster?!!?
* Trader code ' rl_trader.py'
```
# Install and import TF2
# !pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)

# More imports
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

# get the data
# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/aapl_msi_sbux.csv

# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
  # returns a T x 3 list of stock prices
  # each row is a different stock
  # 0 = AAPL
  # 1 = MSI
  # 2 = SBUX
  df = pd.read_csv('aapl_msi_sbux.csv')
  return df.values


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros(size, dtype=np.uint8)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.uint8)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # Note: you could also populate the replay buffer here

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  """ A multi-layer perceptron """

  # input layer
  i = Input(shape=(input_dim,))
  x = i

  # hidden layers
  for _ in range(n_hidden_layers):
    x = Dense(hidden_dim, activation='relu')(x)

  # final layer
  x = Dense(n_action)(x)

  # make the model
  model = Model(i, x)

  model.compile(loss='mse', optimizer='adam')
  print((model.summary()))
  return model


class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """

  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list = list(
        map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()

  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()

  def step(self, action):
    assert action in self.action_space

    # get current value before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    # perform the trade
    self._trade(action)

    # get the new value after taking the action
    cur_val = self._get_val()

    # reward is the increase in porfolio value
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs

  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = []  # stores index of stocks we want to sell
    buy_index = []  # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1  # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False


class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size)

  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action

  def replay(self, batch_size=32):
    # first check if replay buffer contains enough data
    if self.memory.size < batch_size:
      return

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    # Calculate the tentative target: Q(s',a)
    target = rewards + self.gamma * \
        np.amax(self.model.predict(next_states), axis=1)

    # The value of terminal states is zero
    # so set the target to be the reward only
    target[done] = rewards[done]

    # With the Keras API, the target (usually) must have the same
    # shape as the predictions.
    # However, we only need to update the network for the actions
    # which were actually taken.
    # We can accomplish this by setting the target to be equal to
    # the prediction for all values.
    # Then, only change the targets for the actions taken.
    # Q(s,a)
    target_full = self.model.predict(states)
    target_full[np.arange(batch_size), actions] = target

    # Run one training step
    self.model.train_on_batch(states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)
    state = next_state

  return info['cur_val']


if __name__ == '__main__':

  # config
  models_folder = 'rl_trader_models'
  rewards_folder = 'rl_trader_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  data = get_data()
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.h5')

  # play the game num_episodes times
  for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    portfolio_value.append(val)  # append episode end portfolio value

  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/dqn.h5')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
```
* plotter code 'plot_rl_rewards.py'
```
######### PLOT REWARDS  ###########
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

plt.hist(a, bins=20)
plt.title(args.mode)
plt.show()
```
* run trader and plot results
```
python rl_trader.py -m train && python plot_rl_rewards.py -m train
python rl_trader.py -m test && python plot_rl_rewards.py -m test
```

### Lecture 92. Reinforcement Learning Stock Trader Discussion

* we see that our trader always makes a profit
* to benchmark it we will compare it to an agent that takes completely random actions
* this is done by running a script setting epsilon=1
* the distribution is > initial investment so  but there is a significant chance to lose money
* our trained agent is not losing money but is still sensitive to hyperparams
* how come that even with random actions we make profit.?
* it has to do with the dataset. if we plot it we see there is a trend
* all3 stocks keep increasing
* we should test our agent on stocks that go up and down... even synthetic datasets
* it would be great to incorporate metadata as well (news and twotter sentiment)
* or even incorporate past stock price values into the state
* no concept of the movement of the stock price
* we could use returns instead on stock prices
* NNs for regression are not great for extrapolation
* we can see it with synthetic data outside our training range

## Section 13: Advanced Tensorflow Usage

### Lecture 93. What is a Web Service? (Tensorflow Serving pt 1)

* we will see TF serving
* it is a utility that allows us very easily to create a web service for our NN
* this is about using the model after training it
* say we offer a service like spam detection to our clients (email providers)
* we dont intend to build a whole email server from scratch 
* teams are split to Frontend Dev (HTML, CSS, JS), Backend Dev (Python,GO, Java)
* Frontend - Backend talk through an API, usually REST or other WebService API
* For Serving Tensorflow we use TF Serving

### Lecture 94. Tensorflow Serving pt 2

* we build a notebook to showcase TF Servings
* we use python request lib to hit an api and get a reply
```
# By the way, what is a server/service/API?

# We hit a service that reurns our IP in a JSON
import requests
r = requests.get('https://api.ipify.org?format=json')
j = r.json()
print(j)

# Our Tensorflow model server is the same, except that what it does is much more
# complex - it returns the predictions from a ML model:
```

* we do some tf imports, prepare data and build our Model to train it
```
# More imports
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

# the data is only 2D!
# convolution expects height x width x color
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print(x_train.shape)

# number of classes
K = len(set(y_train))
print("number of classes", K)

# Build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i,x)
model.summary()

# Compile and fit
# Note: make sure you are using the GPU for this
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=15)
```

* we save the model for reuse
```
# Save the model to a temporary directory
import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path= {}\n'.format(export_path))
if os.path.isdir(export_path):
  print('\nAlready saved a model: cleaning up\n')
  !rm -r {export_path}

tf.saved_model.save(model, export_path)

print('\nSaved model:')
!ls -l {export_path}
```
* we see that assets, variables and the model.pb (protocol buffer) file is saved
* we print out info on the model
```
!saved_model_cli show --dir {export_path} --all
```

* we see the Input and output data format. -1 is a windcard representing the batch size
* we do some prework before installing the tensorflow service program (get the packages required)
* we install the service
```
!apt-get install tensorflow-model-server
```

* we set model dir as env variable
```
os.environ["MODEL_DIR"] = MODEL_DIR
```

* we start the server serving the model (nohup means no hangup of service on logout). we also keep a log
```
%%bash --bg
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="${MODEL_DIR}" >server.log 2>&1
```

* we check the log `!tail server.log`
* we will build some requests to hit the TF server
* we start with helper methods
```
# Label mapping
labels = '''Tshirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split("\n")

def show(idx, title):
  plt.figure()
  plt.imshow(x_test[idx].reshape(28,28),cmap='gray')
  plt.axis('off')
  plt.title('\n\n{}'.format(title),fontdict={'size': 16})

i = np.random.randint(0,len(x_test))
show(i, labels[y_test[i]])
```

* prepare the request
```
# Format some data to pass to the server
# {
#   "signature_name": "service_defaut",
#   "instances": [an N x H x W x C list ],   
# }
import json
data = json.dumps({ "signature_name": "serving_default", "instances": x_test[0:3].tolist()})
print(data)
```
* the service name we saw in the printout. data array is converted to list as JSON does not accept np arrays.
* we hit the backend
```
headers = {"content-type": "application/json"}
r = requests.post("http://localhost:8501/v1/models/fashion_model:predict", data=data, headers=headers)
j = r.json()
print(j.keys())
print(j)
```
* the reply is JSON with one key 'predictions' and a floating point array
* we check the shape to confirm its then one-hot encoded class probbility
```
# it looks like a 2D array, lets check its shape
pred = np.array(j['predictions'])
print(pred.shape)

# this is the NxK output array from themodel
# pred[n,k] is the probability that we beleieve the nth sample belongs to the kth class
```
* we get the labels of predictions
```
# get the predicted classes
pred = pred.argmax(axis=1)
# Map them back to strings
pred = [labels[i] for i in pred]
print(pred)
```

* check actual values and plot input
```

actual = [labels[i] for i in y_test[:3]]
print(actual)
for i in range(0,3):
  show(i,f'True: {actual[i]}, Predicted: {pred[i]}')
```

* we ll see how to call different versions of the same model through the api (good for beta testing)
```
# Allows you to select a model by version

headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model/versions/1:predict', data=data, headers=headers)
j = r.json()
pred = np.array(j['predictions'])
pred = pred.argmax(axis=1)
pred = [labels[i] for i in pred]
for i in range(0,3):
  show(i, f"True: {actual[i]}, Predicted: {pred[i]}")
```

* we create a v2 of the model train and fit it and save it as v 2
```
# Let's make a new model version
# Build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Flatten()(x)
x = Dense(K, activation='softmax')(x)

model2 = Model(i, x)
model2.summary()
# Compile and fit
# Note: make sure you are using the GPU for this!
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model2.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
# Save version 2 of the model

version = 2
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
if os.path.isdir(export_path):
  print('\nAlready saved a model, cleaning up\n')
  !rm -r {export_path}

tf.saved_model.save(model2, export_path)

print('\nSaved model:')
!ls -l {export_path}
```

* no need to restart the server
* we call v2 of the model with the API and see the output. 
```
# Will Tensorflow serving know about the new model without restarting?

headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model/versions/2:predict', data=data, headers=headers)
j = r.json()
pred = np.array(j['predictions'])
pred = pred.argmax(axis=1)
pred = [labels[i] for i in pred]
for i in range(0,3):
  show(i, f"True: {actual[i]}, Predicted: {pred[i]}")
```

* nice results for a smaller network
* we try to hit a version that does not exist and see the error
```
# What if we use a version number that does not exist?
headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model/versions/3:predict', data=data, headers=headers)
j = r.json()
print(j)
```
* if we dont specify version in our model call TF serves the most recent model with same name
* to deploy it. go to devops.... (Docker,K8s,Nginx)

### Lecture 95. Tensorflow Lite (TFLite)

* 