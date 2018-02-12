# My first try to predict bitcoin price using deep learning.

As cyptocurrencies are attracting attentions and I'm learning Deep Learning recent days, I tried to use<br>
Deep Learning to predict Bitcoin price.<br>

I chose Chainer as the library because it is easier to install with only one command and no other<br>
dependent libraries are needed.<br>
In this attempt, I used LSTM to create an Recurrent Neural Network.

## Dataset Preparation(poloniex_data_reader.py)
All Bitcoin data are from Poloniex through API.<br>
[Polomiex Python API](https://github.com/s4w3d0ff/python-poloniex)<br>

I plotted "close"(target variable) in the JSON response retrived from the above RESTful API as follows.<br>
![Bitcoin price](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/data_plot.jpg)

Next, we need to generate features(explanatory variables) from "close". A common method called differencing is<br>
used here for that purpose.<br>

As "close" is a time series dataset, we denote the sequence data "close" as close(0), close(1), close(2)...close(m).<br>
For each close(t) at time t, features for close(t) can be represented as a group of sequence data.<br>
Defining length of the sequence as L.(In another word we have L features.)<br>
So feature sequences can be defined as close(t-L), ..., close(t-1)<br>

Obvious, we will use feature sequence close(t-L), ..., close(t-1) to predict close(t).

Finally, I splitted the whole dataset into 60% as training set, 20% as cross-validation set and 20% as test set.

## Normalization

As LSTM use tanh inside, normalzation is required. I normalized both explanatory variables and target variables.<br>
While in real productions, normalizing target varibles is not considered as a good practice. Instead we should<br>
define a proper output layer and activation functions to scale the output to the same range as target variables.<br>
But here as a biginner, I chose to normalize target varibles and inverse them after predicting.

## LSTM（Long Short-Term Memory）
LSTM is a deep learning model for processing time series data.<br>
I'll not explain LSTM here as there are a lot of resouces about it. Here I'd like to explain how to use it in Chainer.

### 1.RNN definition(networks.py)
I designed a 3-layer(aka link in Chianer) RNN, linear layers as input and out layers, an LSTM layer as the hidden<br>
layer. Actually the network definition depends on experiences. As a beginner I just defined a simple network by<br>
referencing Chainer documents.

### 2.Chainer Updater(updater.py)
As states saved in the LSTM network need to be reset after a batch processing, in general, we need to reset states<br>
within the batch loop. But when we use

### 3.put all together and start learning(main.py)
- Make the dataset ready.
- Initialize the RNN
- Prepare an optimizer(SGD) for the RNN
- Intialize iterators for mini batch processing
- Initalize the updater to run iterators(reset states in LSTM each batch here)
- Define a trainer to run updater epoch times(loop is not needed)
- Call trainer.run() to start learing

## Predictions(lstm.ipynb)
[Click here to see the notebook](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/lstm.ipynb)

First let's see the the predictions of the cv data set.<br>
![predictions on CV data set](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/predictions_on_cv.jpg)

As the predictions on cross-validation data set are quite similar to real price values, at least I can say that the<br>
learning process is going well.

Next let's see the the predictions of the test data set.<br>
![predictions on test data set](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/predictions_on_test.jpg)

The predictions on test data set look like a straight line that means our model is overfitted and the predictions<br>
are not going well.

In a neural network, as stated before in general an activation function on the output layer is needed to scale outpus<br>
to the same range as the target variable price. In this attempt, instead of using an activation function, I scaled<br>
target variable price to 0~1, and after predicting, I inversed the predictions to the original scale.<br>

So here comes the problems.<br>

I created a scaler for 80% of the whole data set, and finally used this scaler to inverse the rest 20% which is the<br>
test data set set. We can simply consider this process as follows:<br>
1. create a scaler for 80% of the whole data set. Simply consider the scaler as a float number max_val which is the<br>
   maximun value<br>
2. after predicting on test data set, use the above scaler to inverse the predictions. Simply consider this step as<br>
   predictions*max_val<br>

## Conclusions

Obviously as shown in the Bitcoin price graph, the max_value of the first 80% is totally different to that of the<br>
rest 20%, the above normalization process makes it difficult to inverse the predictions. So as a conclusion,<br>
normalizing target variables is not a good idea as the scale is not always the same on each data set.<br>
(Though this approach performs well on the CV data set because of overfitting)<br>

We can also see that it's really hard to apply machine learing on Demand forecasting.

But as a beginner, I still have a lot to do with parameters to refine my learning model.

## Parameters for Learning for this time
- normalization: scale to 0~1
- mini batch size: 20
- epoch: 30
- network definition: linear-layer➛LSTM-layer➛linear-layer
- units of hidden layer: 100
- activation function: not used
- dropout: not used
- optimization algorithm: SGD
- loss function: mean_squared_error
Adjusting the above parameters to make a better model
