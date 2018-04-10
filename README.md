# My first try to predict bitcoin price using deep learning in Keras.

As cryptocurrencies are attracting attentions and I'm learning Deep Learning recent days, I'll try to use Deep Learning to predict Bitcoin price.
In this attempt, I'll use LSTM to create an Recurrent Neural Network.

## What is LSTM

LSTM was proposed 20 years ago. Recently in the field of NLP(Natural Language Processing), a language model built by LSTM has better performance even than human. LSTM (Long-Short Term Memory) is known for good at processing long sequence such as long sentences. Gates are deigned in LSTM to contorl and decide what are important to learn.

Want to know more about LSTM? refer to the following site.
[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Stateful or Stateless LSTM

In keras, LSTM is stateless as default. But in this attempt, I'll use stateful LSTM. A stateful LSTM will retain the states after a batch processing while a stateless LSTM will reset them.<br>
In natural language processing, say we have a long sentence while it is splitted into two different batchs. In order to learning the long sentence correctly, the hidden states should be retained so that the second batch can know the hidden information of the previous batch.  
States here means the hidden information saved by gates in LSTM.<br>
In this attempt, as I'll just generate features from timeseries data and don't shuffle them, features are strongly connected. So stateful LSTM should be used to retain states.

## Dataset Preparation
All data here are from Poloniex through API.<br>
[Polomiex Python API](https://github.com/s4w3d0ff/python-poloniex)<br>

I use returnChartData method to retrieve a JSON string containing bitcoin price data.<br>
'close' column in the JSON string is the price of a specified cryptocurrency.<br>

I use 'close' data during 2018-03-30 00:00 ~ 2018-04-01 00:00 to train a LSTM model and make predictions.<br>
![Bitcoin price](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/data_plot.png)

Next, as this is a prediction problem, I need to generate features for bitcoin price data so that I can make predictions later.<br>  
I use the same method described in the following blog.<br>
[https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/](https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/)

Simply there are 2 steps:<br>
- I use a simple technique called differencing to remove the increasing trend in the data
- I make 30 continuous timeseteps as features, and the 31th timestep as target for prediction. 

Obvious, now in a supervised learning model I can use feature sequence close(t-L), ..., close(t-1) to predict close(t).

See [dataManager.py](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/dataManager.py) for implementation.

## Data Normalization

As LSTM use tanh as activation function inside, I rescale all data to range from -1 to 1. 

## Neural Network Definition

As a deep learning beginner, I design a shallow neural network here.<br>
In general a deep neural network can get a better performance. But as I'm not sure how two LSTM layers refines the performance and adding layers increses computation time, my definition is as follows.<br>
I add one LSTM layer as the input layer and a fully connected layer(aka. Dense in Keras) as the output layer.

<pre>
SEQ_LENGTH = 30
BATCH_SIZE = 1
DATA_DIM = 1
model = Sequential()
model.add(LSTM(32, return_sequences=False, batch_input_shape=(BATCH_SIZE, SEQ_LENGTH, DATA_DIM), stateful=True))
model.add(Dense(1, activation='linear'))
</pre>

The network should look as follows:<br>
![Network](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/network.png)

As batch_input_shape of LSTM constuctor is a bit confusing, I'd like to add some explain here.<br>
LSTM require a 3-dimensional array as input. The first 2 dimemsions represents for m samples and n features(aka. sequence length). Each element in a sequence can also have its own dimension. As each element here contains only one value, the 3rd dimension is 1.
I set batch_size as 1 so LSTM will process m samples one-by-one.

Next I set the function as rmsprop to do back propagation to reduce MSE score.
<pre>
model.compile(loss="mean_squared_error", optimizer="rmsprop")
</pre>

## Build Neural Network Model

### Cross Validation(CV)

Cross Validation is a common method to build a better learning model(aka. preventing overfitting). Tranditional CV method like k-fold do cross validation k times to get a reliable validation result. While in deep learning, k times validation would be rather time consuming, so in general we do CV only one time or ignore CV at all. Instead of CV, deep learning have another method called Dropout to prevent overfitting.
Dropout is improved effective in an early research paper.<br>
["Dropout: A Simple Way to Prevent Neural Networks from Overfitting". Jmlr.org. Retrieved July 26, 2015.](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)

Here I use one-time CV and don't apply Dropout.

### reshape data

Before using training data to fit our model, in Keras, we need to reshape input data as 3-dimensional arrays.<br>
Dimension definition is the same as batch_input_shape explained above.

### batch size

I stated before, in our model we use sequence close(t-L), ..., close(t-1) to predict close(t). This also means that in order to predict close(t+1), the real value of close(t) is required.<br>
This is called one step forecast and batch size should be the same size of forecast step. 

### epoch
Epoch is another important parameter in deep learning. Epoch defines how many times to train the model repeatly.
Large epoch leads to better performance, but it's a trade-off between performance and time.

### bring all together
<pre>
from sklearn.model_selection import train_test_split

EPOCHS = 30

# split CV set from training set.
train_data_scaled, cv_data_scaled = train_test_split(
    dm.train_scaled, test_size=0.25, random_state=2, shuffle=False)
x_train_scaled, y_train_scaled = train_data_scaled[:, 0:-1], train_data_scaled[:, -1]
x_cv_scaled, y_cv_scaled = cv_data_scaled[:, 0:-1], cv_data_scaled[:, -1]

# reshape data as a tensor(3 dims)
x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], SEQ_LENGTH, DATA_DIM))
x_cv_scaled = x_cv_scaled.reshape((x_cv_scaled.shape[0], SEQ_LENGTH, DATA_DIM))

# fit data
model.fit(x_train_scaled, y_train_scaled,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_cv_scaled, y_cv_scaled),
          shuffle=False)
</pre>

## make predictions
Implementation of one step forecast is as follows:

<pre>
# perform one-step forecast
def one_step_prediction_using_test_data(model, previos_y):

    y_pred = np.zeros((len(dm.test_scaled), 1))

    for i in range(len(dm.test_scaled)):

        x = dm.test_scaled[i, :-1].reshape(BATCH_SIZE, SEQ_LENGTH, DATA_DIM)
        _y_pred = model.predict(x)

        x = x.reshape(BATCH_SIZE, SEQ_LENGTH)
        _y_pred = _y_pred.reshape(BATCH_SIZE, 1)

        # inverse scale
        y_pred_indiffereced = dm.inverse_data(x, _y_pred, previos_y)

        previos_y = dm.test_original_df.iloc[i, -1]

        model.reset_states()
        y_pred[i, 0] = y_pred_indiffereced

        # create a new sequence for the next prediction
        x = np.delete(x, 0)
        x = np.append(x, _y_pred[0, 0])

    return y_pred


predicted_test = one_step_prediction_using_test_data(model, dm.train_original_df.iloc[-1, -1])

# plot predicitions
plt.figure(figsize=(20,10))
# get the original test data(not differenced)
plt.plot(dm.time_test, dm.test_original_df.iloc[:, -1], label = "real")
plt.plot(dm.time_test, predicted_test, label = "predicted")
plt.legend(loc='best', fontsize=14)
plt.tick_params(labelsize=14)
plt.show()
</pre>

Let's see the the predictions on the test data set.<br>
![predictions on test data set](https://github.com/george-j-zhu/cryptocurrency-price-prediction/blob/master/resources/predictions.png)

Predictions look quite similar to real values, but obviously, predictions are just one step behind.<br>
So I can not say the predictions are successful.<br>
The problem came from how we generate features for the dataset.<br>
I think using data from different source(such as volume of bitcoin) as features may refine this problem.

## Conclusions

I learned how to use LSTM in Keras to make predictions.<br>
I found out that it's hard to predict bitcoin price without other features.<br> 
So tring to find related feature is still an important part in deep learning though deep learning is strong to extract features for us from input data.<br>
Also it's really hard to apply machine learning on Demand forecasting.<br>

But as a beginner, I still have a lot to do with parameters to refine my learning model.

## Parameters for Learning this time
- epoch: 30
- network definition: LSTM-layer➛Dense-layer
- neurons of output for LSTM layer: 32
- activation function: linear
- dropout: not used
- optimization algorithm: rmsprop
- loss function: mean_squared_error

## Reference
1. [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/](https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/)
3. ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting". Jmlr.org. Retrieved July 26, 2015.](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
