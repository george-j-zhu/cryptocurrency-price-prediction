# My first try to predict bitcoin price using deep learning.

As cyptocurrencies are attracting attentions and I'm learning Deep Learning recent days, I tried to use<br>
Deep Learning to predict Bitcoin price.<br>

I chose Chainer as the library because it is easier to install with only one command and no other<br>
dependent libraries are needed.<br>
In this attempt, I used LSTM to create an Recurrent Neural Network.

## Dataset Preparation(poloniex_data_reader.py)
All Bitcoin data are from Poloniex.<br>
[Polomiex Python API](https://github.com/s4w3d0ff/python-poloniex)<br>

As only "close"(target variable) in the JSON response retrived from the above RESTful API will be used, we<br>
need to generate features(explanatory variables) from "close". A common method called differencing is used<br>
here for that purpose.<br>

As "close" is a time series dataset, we denote the sequence data "close" as close(0), close(1), close(2)....<br>
For each close(t) at time t, features for close(t) can be presented as a group of sequence data.<br>
Defining length of the sequence as L.(In another word we have L features.)<br>
So feature sequences can be defined as close(t-L), ..., close(t-1)<br>

Obvious, We will use feature sequence close(t-L), ..., close(t-1) to predict close(t).

## Normalization
As LSTM use tanh inside, normalzation is required. I normalized both explanatory variables and target variables.<br>
While in real productions, normalizing target varibles is not considered as a good practice. Instead we should<br>
define a proper output layer and activation functions to scale the output to the same range as target variables.<br>
But here I chose to normalize target varibles and inverse them after predicting.

## LSTM（Long Short-Term Memory）
I'll not explain LSTM here as there are a lot of resouces about it. Here I'd like to explain how to use it in Chainer.

### 1.RNN definition(networks.py)
I designed a 3-layer(aka link in Chianer) RNN, linear layers as input and out layers, an LSTM layer as the hidden<br>
layer. Actually the network definition depends on experiences. As a beginner I just defined a simple network by<br>
referencing Chainer documents.

### 2.Chainer Updater(updater.py)
As states saved in the LSTM network need to be reset after a batch processing, in general, we need to reset states within<br>
the batch loop. But when we use

### 3.Learning(main.py)
- Make the dataset ready.
- Initialize the RNN
- Prepare an optimizer for the RNN
- Initalize the updater
- Finally define a trainer
- Call trainer.run() to execute.

## Plot and Conclusion(lstm.ipynb)
As shown in lstm.ipynb, the cv predicitons is quite similar to the real prices, but if we zoom out to see details,<br>
we'll find out the trend of the predicitons slipped away that means the predictions are not going well.
