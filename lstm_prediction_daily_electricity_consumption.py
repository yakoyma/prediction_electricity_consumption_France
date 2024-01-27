"""
===============================================================================
Time Series Regression project: Prediction of daily consumption of electricity
in France using the Long Short-Term Memory (LSTM) network
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import sktime
import tensorflow as tf
import keras_tuner

from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series
from sktime.transformations.series.lag import Lag
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('KerasTuner: {}'.format(keras_tuner.__version__))


# Constant
SEED = 0

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', 300)

# Set the default Seaborn style
sns.set_style('whitegrid')



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
# Loading the dataset
raw_dataset = load_dataset('Consommation-nationale-quotidienne-RTE.csv')

# Display the dataset's dimensions
print('\n\nDimensions of the dataset: {}'.format(raw_dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(raw_dataset.info())

# Display the head and the tail of the dataset
print(pd.concat([raw_dataset.head(), raw_dataset.tail()]))

# Description of the dataset
print('\nDescription of the dataset:')
print(round(raw_dataset.describe(include='all'), 0))

# The completion rate of the dataset
print(f'\nCompletion rate:\n{raw_dataset.count() / len(raw_dataset)*100}')

# Missing rate of the dataset
print(f'\nMissing rate:\n{raw_dataset.isna().mean() * 100}')

# Cleanse the dataset
dataset = raw_dataset.copy()

# Check for duplicated data
print(f'\nNumber of duplicated data: {dataset[dataset.duplicated()].shape[0]}')

dataset = dataset.rename(columns={'Consommation (MW)': 'Consumption'})
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(
    by=['Date'], ascending=True).reset_index(drop=True).set_index('Date')
dataset.index = pd.PeriodIndex(dataset.index, freq='D')

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Display head and the tail of the dataset
print(pd.concat([dataset.head(), dataset.tail()]))

# Display the daily consumption of electricity in France
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    dataset,
    markers='.',
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Daily consumption of electricity in France from '
             f'{dataset.index.min()} to {dataset.index.max()}')
plt.show()



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
# Transformation of the dataset to create exogenous variables.
# Creation of 31 lags (features)
X = Lag([i for i in range(32)]).fit_transform(dataset).dropna()
X = X.rename(columns={'lag_0__Consumption': 'Target'})

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Display the target
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X.Target,
    markers=['.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Daily consumption of electricity in France from '
             f'{X.index.min()} to {X.index.max()}')
ax.legend(loc='best')
plt.show()

# Split the dataset into training and test sets
train_set = X.iloc[0:int(0.8 * X.shape[0]),]
test_set = X.iloc[train_set.shape[0]:X.shape[0],]

# Display head and the tail of the datasets
print(f'\n\nTraining set shape: {train_set.shape}')
print(pd.concat([train_set.head(), train_set.tail()]))
print(f'\nTest set shape: {test_set.shape}')
print(pd.concat([test_set.head(), test_set.tail()]))

# Display training and test targets
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Daily consumption of electricity in France from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Standardisation
# Scale values between the range of 0 and 1
scaler = MinMaxScaler()
y_train = scaler.fit_transform(np.array(train_set.Target).reshape(-1, 1))
y_test = scaler.transform(np.array(test_set.Target).reshape(-1, 1))

exog_scaler = MinMaxScaler()
X_train = exog_scaler.fit_transform(
    np.array(train_set.drop(['Target'], axis=1)))
X_test = exog_scaler.transform(
    np.array(test_set.drop(['Target'], axis=1)))
print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Reshape exogenous features three-dimensional for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f'\nX_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# Train and evaluate the LSTM model
# Build the model
# Instantiate the tuner
tuner = BayesianOptimization(
    hypermodel=build_lstm_model,
    objective='val_loss',
    max_trials=5,
    seed=SEED)

# Create the callback that stops training the model
# when the loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=0,
    patience=5)

# Train and find the best model
tuner.search(
    x=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping_callback])

# Get the best model with optimised hyperparameters
model = tuner.get_best_models(num_models=1)[0]
print('\n\nOptimal hyperparameters:')
print(tuner.get_best_hyperparameters()[0].values)
print('\nSummary of the best model with optimised hyperparameters:')
print(model.summary())

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
for metric, score in zip(['Loss (MSE)', 'MSE', 'MAE'], scores):
    print(f'{metric}: {score:.3f}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test.flatten(), index=test_set.index),
    pd.Series(data=y_pred.flatten(), index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = scaler.inverse_transform(y_pred).astype(int)
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sklearn_metrics(
    np.array(test_set.Target), pred_target.flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target.flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
