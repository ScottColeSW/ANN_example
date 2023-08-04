# Requires the latest pip
#python -m pip install --upgrade pip

# Current stable release for CPU and GPU
#python -m pip install tensorflow

#Current stable Keras release
#python -m pip install keras

# Importing Necessary Libraries for Artificial Neural Network
# Let’s import all the necessary libraries here

#Importing necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow import keras

# ann = keras.models.Sequential()
# Importing Dataset
# In this step, we are going to import our dataset. Since our dataset is in csv format, we are going to use the read_csv() method of pandas in order to load the dataset.

#Loading Dataset
data = pd.read_csv("churn_modeling.csv")
X = data.iloc[:,3:-1].values
print(X)

# Generating Dependent Variable Vector(Y)
# In the same fashion where we have created our matrix of features(X) for the independent variable, we also have to create a dependent variable vector(Y) which will only contain our dependent variable values.

#Generating Dependent Variable Vectors
Y = data.iloc[:,-1].values
print(Y)

# Encoding Categorical Variable Gender
# Now we have defined our X and Y, from this point on we are going to start with one of the highly time-consuming phases in any machine learning problem-solving. This phase is known as feature engineering. To define it in a simple manner, feature engineering is a phase where we either generate new variables from existing ones or modify existing variables so as to use them in our machine learning model.
#Encoding Categorical Variable Gender
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))
print(X)

# Encoding Categorical Variable Country
# Now let’s deal with another categorical column named country. This column has a cardinality of 3 meaning that it has 3 distinct categories present i.e France, Germany, Spain.

# Here we have 2 options:-

# 1. We can use Label Encoding here and directly convert those values into 0,1,2 like that

# 2. We can use One Hot Encoding here which will convert those strings into a binary vector stream. For example – Spain will be encoded as 001, France will be 010, etc.
#Encoding Categorical variable Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting Dataset into Training and Testing Dataset
# In this step, we are going to split our dataset into training and testing datasets. This is one of the bedrocks of the entire machine learning process. The training dataset is the one on which our model is going to train while the testing dataset is the one on which we are going to test the performance of our model.
# Here we have used the train_test_split function from the sklearn library. We have split our dataset in a configuration such that 80 percent of data will be there in the training phase and 20 percent of data will be in the testing phase.

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# Performing Feature Scaling
# The very last step in our feature engineering phase is feature scaling. It is a procedure where all the variables are converted into the same scale. Why you might ask?. Sometimes in our dataset, certain variables have very high values while certain variables have very low values. So there is a chance that during model creation, the variables having extremely high-value dominate variables having extremely low value. Because of this, there is a possibility that those variables with the low value might be neglected by our model, and hence feature scaling is necessary.
# For a multitude of reasons, we should always perform feature scaling after the train-test split.
# We can scale using 2 common techniques: Standardization and Normalization
# Normalization is used when we have a normal distribution, Standardization is used when we don't know better as a first step.
#Performing Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing Artificial Neural Network
# This is the very first step while creating ANN. Here we are going to create our ann object by using a 
# certain class of Keras named Sequential.

#Initialising ANN
ann = tf.keras.models.Sequential()
#x = tf.
#ann = tf.keras.Sequential()
#ann.add(tf.keras.layers.Dense(8, input_shape=(16,)))

# Creating Hidden Layers
# Once we initialize our ann, we are now going to create layers for the same. Here we are going to create a network that will have 2 hidden layers, 1 input layer, and 1 output layer. So, let’s create our very first hidden layer
# Here we have created our first hidden layer by using the Dense class which is part of the layers module. This class accepts 2 inputs:-
# 1. units:- number of neurons that will be present in the respective layer
# 2. activation:- specify which activation function to be used

 #Adding First Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

# 1. In a binary classification problem(like this one) where we will be having only two classes as output (1 and 0), we will be allocating only one neuron to output this result. For the multiclass classification problem, we have to use more than one neuron in the output layer. For example – if our output contains 4 categories then we need to create 4 different neurons[one for each category].
# 2. For the binary classification Problems, the activation function that should always be used is sigmoid. For a multiclass classification problem, the activation function that should be used is softmax.
# Here since we are dealing with binary classification hence we are allocating only one neuron in the output layer and the activation function which is used is softmax.

# Compiling Artificial Neural Network
# We have now created layers for our neural network. 
# In this step, we are going to compile our ANN.
# We use compile method of our ann object in order to compile our network. Compile method accepts the below inputs:-
# 1. optimizer:- specifies which optimizer to be used in order to perform stochastic gradient descent. I had experimented with various optimizers like RMSProp, adam and I have found that adam optimizer is a reliable one that can be used with any neural network.
# 2. loss:- specifies which loss function should be used. For binary classification, the value should be binary_crossentropy. For multiclass classification, it should be categorical_crossentropy.
# 3. metrics:- which performance metrics to be used in order to compute performance. Here we have used accuracy as a performance metric.

#Compiling ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# Fitting Artificial Neural Network
# This is the last step in our ann creation process. Here we are just going to train our ann on the training dataset.
# Here we use the fit method in order to train our ann. 
# The fit method is accepting 4 inputs in this case:-
# 1.X_train:- Matrix of features for the training dataset
# 2.Y_train:- Dependent variable vectors for the training dataset
# 3.batch_size: how many observations should be there in the batch. Usually, the value for this parameter is 32 but we can experiment with any other value as well.
# 4. epochs: How many times neural networks will be trained. Here the optimal value that I have found from my experience is 100.

#Fitting ANN
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

#Predicting result for Single Observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])) > 0.5)

#Saving created neural network
# keras is a specific file format used by neural networks
ann.save("SAC_ANN.keras")