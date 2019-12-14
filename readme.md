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
