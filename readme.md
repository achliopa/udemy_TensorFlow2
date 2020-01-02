# Udemy Course: Tensorflow 2.0: Deep Learning and Artificial Intelligence

* [Course Link](https://www.udemy.com/course/deep-learning-tensorflow-2/)
* [Course Repo](https://github.com/lazyprogrammer/machine_learning_examples)

## Section 1: Welcome

### Lecture 2: Outline

* TF2 has KEras API as standard
* TF2 is a library but also an API spec
* Easy to build NNs of various types
* Google Colab is Jupyter Notebook hosten on Google

### Lecture 3: Where to get the code

* Shared read-only notebooks in each lecture extras. copy them to GoogleDrive and run them on colab or download and run them locally in jupyter

## Section 2: Google Colab

### Lecture 4. Intro to Google Colab, how to use a GPU or TPU for free

* Colab hosted by Google
* many libs preincluded
* using GPUs and TPUs
* IN our GoogleDrive we go to Coonect more apps and connect Colab 
* in Settings => ManageApps => enable select by default to Colaboratory
* then in our drive we can select New => More => Colaboratory to add a Colab notebook to our drive and launch Colab
* we can use it like normal notebook
* we rename it
* we go to Runtime => Change Runtime type => There we select Python version and select HW accelarator. GPU or TPU is not always available as it is shared
* in Colab Notebook we can create Text cells and Code shells
* text is using Markdown. left is raw and on right is formatted
* code is Python
* we test a simple plot script (run with SHIFT+ENTER)
```
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10*np.pi, 1000)
y = np.sin(x)
plt.plot(x,y)
```
* pretty much any python lib in in colab
* stale notebook gets kicked out (share resources) so we need to rerun our notebook after using again our workspace

### Lecture 5. Tensorflow 2.0 in Google Colab

* if we import tenssorflow to colab and printout its version we will see not v2
```
import tensorflow as tf
print(tf.__version__)
```
* this is because TF2 is still in beta.
* if we pip install tensorflow locally we wont get 2.0
* colab allows us to install additional libs
* we install tf2 with running `!pip install -q tensorflow==2.0.0` in a cell
* in colab we can install libs with `!pip`
* to install the gpu version `!pip install -q tensorflow-gpu==2.0.0`
* to import the correct version first install v2 and then import
* runtime => restart runtime
* there is a way to permanently install tensorflow 2 to colab if we need to
* we can run unix commands from notebook in colab putting ! in front `!ls`
* we see that in colab all notebooks have a sample_data folder with some famous data sets to play  with `!ls sample_data` to view them

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
