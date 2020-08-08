import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from rpy2 import forecast
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn import preprocessing
from collections import deque
from sklearn.metrics import mean_squared_error
from math import sqrt
# importing packages for the prediction of time-series data
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# import ETH price
df = pd.read_csv('/Users/Lafa/Desktop/ETH-USD.csv', header=0)
df = df.sort_values('Date')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Volume'] = df['Volume'].astype('int64')
df.rename(column = {"Close": "ETH_Price"})
df.head()

# visualize the plot
price_plot = df.plot(y='Close', figsize=(12,6), legend=True, grid=True, use_index=True)
plt.title("Closing price distribution of ETH", fontsize=15)
plt.show()


# select duration
initial_date = '2017-01-01'
finish_date = '2017-12-31'
ETH_price_time = df[initial_date:finish_date]

# Testing the Stationarity
from statsmodels.tsa.stattools import adfuller


def test_stationarity(x):
    # Determing rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()

    rolstd = x.rolling(window=12, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(x, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey Fuller test
    result = adfuller(x)
    print('ADF Stastistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            print("The graph is non stationery")
            break
        else:
            print("The graph is stationery")
            break;
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f ' % (key, value))


ts = df['Close']
test_stationarity(ts)


# Since the p-value is 0.154786, which is greater than 0.05 (non-stationary), we use transformations to make the series stationary
# Log Transforming the series
ts_log = np.log(ts)
plt.plot(ts_log, color="green")
plt.show()
test_stationarity(ts_log)

# Still non-stationary, remove trend and seasonality with differencing
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
# now it's stationary!

# Auto Regressive model
model = ARIMA(ts_log, order=(1,1,0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

# Moving Average Model
model = ARIMA(ts_log, order=(0,1,1))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.7f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log, order=(2,1,0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

# Split into train&test model
size = int(len(ts_log)*0.8)
# Divide into train and test
train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...')
print('\n')
# We go over each value in the test set and then apply ARIMA model and calculate the predicted value. We have the expected value in the test set therefore we calculate the error between predicted and expected value
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)

    output = model_fit.forecast()

    pred_value = output[0]

    original_value = test_arima[t]
    history.append(original_value)

    pred_value = np.exp(pred_value)

    original_value = np.exp(original_value)

    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')

    predictions.append(float(pred_value))
    originals.append(float(original_value))

# After iterating over whole test set the overall mean error is calculated.
print('\n Mean Error in Predicting Test Case Articles : %f ' % (sum(error_list) / float(len(error_list))), '%')
plt.figure(figsize=(8, 6))
test_day = [t
            for t in range(len(test_arima))]
labels = {'Orginal', 'Predicted'}
plt.plot(test_day, predictions, color='green')
plt.plot(test_day, originals, color='orange')
plt.title('Expected Vs Predicted Views Forecasting')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.legend(labels)
plt.show()

plt(forecast(auto.arima(ts(ts_log,frequency=365),D=1),h=365))

predicted_day=list(range(365))



def build_lstm_model(input_data, output_size, neurons=20,
                     activ_func='linear', dropout=0.25,
                     loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
              input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))    model.compile(loss=loss, optimizer=optimizer)
    return modelmodel = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=50, batch_size=4)


# lead-lag analysis
# bitcoin price
df2 = pd.read_csv('/Users/Lafa/Desktop/BTC-USD.csv', header=0)
df2 = df2.sort_values('Date')
df2['Date'] = pd.to_datetime(df2['Date'])
df2.set_index('Date', inplace=True)
df2['Volume'] = df2['Volume'].astype('int64')
df2.rename(column = {"Close": "BTC_Price"})
df2.head()

df['Close'].corr(df2['Close'])

# combine ETH and BTC prices together
dataset = pd.concat([df.Close, df2.Close],axis=1)
dataset = dataset.loc['20150807':'20200116']

# plot the prices
dataset.plot(figsize=(10,4))
plt.ylabel('Price')

granger_test_result = grangercausalitytests(dataset, maxlag=12, verbose=False)
optimal_lag = -1
F_test = -1.0
for key in granger_test_result.keys():
    _F_test_ = granger_test_result[key][0]
    if _F_test_ > F_test:
        F_test = _F_test_
        optimal_lag = key
return optimal_lag

