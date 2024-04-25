"""
===============================================================================
Time Series Regression project: Prediction of daily consumption of electricity
in France
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 Sktime
       3.1.1 StatsForecastAutoARIMA
       3.1.2 AutoREG
       3.1.3 Prophet
       3.1.4 CNNRegressor
   3.2 Aeon
       3.2.1 ResNetRegressor
       3.2.2 FCNRegressor
       3.2.3 InceptionTimeRegressor
       3.2.4 KNeighborsTimeSeriesRegressor
   3.3 Scikit-learn
       3.3.1 ExtraTreesRegressor
       3.3.2 HistGradientBoostingRegressor
   3.4 CatBoostRegressor
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
import ydata_profiling
import sklearn
import statsmodels
import pmdarima as pm
import sktime
import aeon
import catboost

from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
from pmdarima.arima.utils import ndiffs
from sktime.utils.plotting import plot_series
from sktime.transformations.series.lag import Lag
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.fbprophet import Prophet
from sktime.regression.deep_learning import CNNRegressor
from aeon.regression.deep_learning import InceptionTimeRegressor
from aeon.regression.deep_learning import ResNetRegressor
from aeon.regression.deep_learning import FCNRegressor
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from catboost import CatBoostRegressor
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))
print('Pmdarima: {}'.format(pm.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('Aeon: {}'.format(aeon.__version__))
print('CatBoost: {}'.format(catboost.__version__))


# Constants
SEED = 0
PERIOD = 7
SP = 31
FOLDS = 5
EPOCHS = 50

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

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

# Description of the dataset
print('\nDescription of the dataset:')
print(round(raw_dataset.describe(include='all'), 0))

# Display the head and the tail of the dataset
print(pd.concat([raw_dataset.head(), raw_dataset.tail()]))

# Time Series raw dataset report
profile = ProfileReport(
    df=raw_dataset, tsmode=True, title='Raw dataset report')
profile.to_file('raw_dataset_report.html')

# Cleanse the dataset
dataset = raw_dataset.copy()
dataset = dataset.rename(columns={'Consommation (MW)': 'Consumption'})
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(
    by=['Date'], ascending=True).reset_index(drop=True).set_index('Date')
dataset.index = pd.PeriodIndex(dataset.index, freq='D')

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(round(dataset.describe(include='all'), 0))

# Display head and the tail of the dataset
print(pd.concat([dataset.head(), dataset.tail()]))

# Time Series dataset report
profile = ProfileReport(df=dataset, tsmode=True, title='Dataset report')
profile.to_file('dataset_report.html')

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
# Tests to determine whether the dataset is stationary and/or invertible
print(f'\n\nStationarity test result: {ArmaProcess(dataset).isstationary}')
print(f'Invertibility test result: {ArmaProcess(dataset).isinvertible}')

# Transformation of the dataset to create exogenous variables.
# Creation of 31 lags (features)
X = Lag([i for i in range(SP + 1)]).fit_transform(dataset).dropna()
X = X.rename(columns={'lag_0__Consumption': 'Target'})

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(X.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(X.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(round(X.describe(include='all'), 0))

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Time Series X dataset report
profile = ProfileReport(df=X, tsmode=True, title='X dataset report')
profile.to_file('X_dataset_report.html')

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
print(f'\nTraining set shape: {train_set.shape}')
print(pd.concat([train_set.head(), train_set.tail()]))
print(f'\nTest set shape: {test_set.shape}')
print(pd.concat([test_set.head(), test_set.tail()]))

# Time Series train set report
profile = ProfileReport(df=train_set, tsmode=True, title='Train set report')
profile.to_file('train_set_report.html')

# Time Series test set report
profile = ProfileReport(df=test_set, tsmode=True, title='Test set report')
profile.to_file('test_set_report.html')

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

# Display autocorrelation
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(acf(train_set.Target))
ax.set_xlabel('Lags')
ax.set_title('Autocorrelation')
plt.show()

# Standardisation
# Scale values between the range of 0 and 1
transformer = MinMaxScaler()
y_train = transformer.fit_transform(
    np.array(train_set.Target).reshape(-1, 1)).flatten()
y_test = transformer.transform(
    np.array(test_set.Target).reshape(-1, 1)).flatten()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(np.array(train_set.drop(['Target'], axis=1)))
X_test = scaler.transform(np.array(test_set.drop(['Target'], axis=1)))
print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# 3.1 Sktime
# 3.1.1 StatsForecastAutoARIMA
# Determine whether to differentiate in order to make it stationary
# Estimate the number of times to differentiate using the KPSS test
print(f"\n\nEstimate the number of times to differentiate the data using "
      f"the KPSS test: {ndiffs(np.array(train_set.Target), test='kpss')}")

# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('StatsForecastAutoARIMA', StatsForecastAutoARIMA(
        sp=PERIOD,
        d=ndiffs(np.array(train_set.Target), test='kpss'),
        seasonal=True,
        information_criterion='bic',
        n_jobs=-1,
        trend='t',
        trace=True))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set.Target,
    X=train_set.drop(['Target'], axis=1),
    fh=ForecastingHorizon(test_set.index, is_relative=False),
    X_pred=test_set.drop(['Target'], axis=1))

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set.Target).shape)
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(forecast),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.2 AutoREG
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('AutoREG', AutoREG(lags=None, trend='t', seasonal=True, period=PERIOD))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set.Target,
    X=train_set.drop(['Target'], axis=1),
    fh=ForecastingHorizon(test_set.index, is_relative=False),
    X_pred=test_set.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set.Target).shape)
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(forecast),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.3 Prophet
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('Prophet', Prophet())])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set.Target,
    X=train_set.drop(['Target'], axis=1),
    fh=ForecastingHorizon(test_set.index, is_relative=False),
    X_pred=test_set.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set.Target).shape)
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(forecast),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.4 CNNRegressor
# Instantiate the model
regressor = CNNRegressor(
    n_epochs=EPOCHS,
    batch_size=16,
    random_state=SEED,
    loss='mean_squared_error',
    metrics=['mean_squared_error'])

# Instantiate the grid search model
hyperparams_grid = {'kernel_size': [5, 7, 9, 11], 'avg_pool_size': [1, 3]}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# 3.2 Aeon
# 3.2.1 ResNetRegressor
# Instantiate the model
regressor = ResNetRegressor(
    n_epochs=EPOCHS,
    batch_size=16,
    loss='mean_squared_error',
    metrics='mean_squared_error')

# Instantiate the grid search model
hyperparams_grid = {
    'n_residual_blocks': [1, 2, 3, 4, 5],
    'activation': ['relu', 'elu', 'tanh']}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.2.2 FCNRegressor
# Instantiate the model
regressor = FCNRegressor(
    n_epochs=EPOCHS,
    random_state=SEED,
    loss='mean_squared_error',
    metrics='mean_squared_error')

# Instantiate the grid search model
hyperparams_grid = {
    'n_layers': [1, 2, 3, 4, 5],
    'activation': ['relu', 'elu', 'tanh']}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.2.3 InceptionTimeRegressor
# Instantiate the model
regressor = InceptionTimeRegressor(
    batch_size=64,
    depth=6,
    n_epochs=EPOCHS,
    random_state=SEED,
    loss='mean_squared_error')

# Instantiate the grid search model
hyperparams_grid = {
    'n_regressors': [7, 8, 9, 10, 11, 12, 13, 14, 15],
    'kernel_size': [30, 40, 50],
    'activation': ['relu', 'elu', 'tanh']}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.2.4 KNeighborsTimeSeriesRegressor
# Instantiate the model
regressor = KNeighborsTimeSeriesRegressor()

# Instantiate the grid search model
hyperparams_grid = {
    'n_neighbors': [i for i in range(200)],
    'weights': ['uniform', 'distance'],
    'distance': ['dtw', 'euclidean']}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(
    np.array(train_set.drop(['Target'], axis=1)),
    np.array(train_set.Target).flatten())

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
pred_target = model.predict(np.array(test_set.drop(['Target'], axis=1)))

# Denormalise the predicted target
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# 3.3 Scikit-learn
# 3.3.1 ExtraTreesRegressor
# Instantiate the model
regressor = ExtraTreesRegressor(random_state=SEED, n_jobs=-1)

# Instantiate the grid search model
hyperparams_grid = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [None, 3, 6]}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.3.2 HistGradientBoostingRegressor
# Instantiate the model
regressor = HistGradientBoostingRegressor(
    loss='squared_error', random_state=SEED)

# Instantiate the grid search model
hyperparams_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000, 1500, 2000],
    'max_depth': [None, 3, 6],
    'l2_regularization': [i for i in range(21)]}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# 3.4 CatBoostRegressor
# Instantiate the model
regressor = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='RMSE',
    iterations=1000,
    learning_rate=0.01,
    random_seed=SEED,
    l2_leaf_reg=3,
    bootstrap_type=None,
    bagging_temperature=1,
    subsample=0.8,
    random_strength=1,
    depth=6,
    min_data_in_leaf=1,
    max_leaves=31,
    early_stopping_rounds=5,
    logging_level='Silent')

# Instantiate the grid search model
hyperparams_grid = {
    'iterations': [500, 1000, 1500, 2000],
    'learning_rate': [0.001, 0.01, 0.1],
    'l2_leaf_reg': [i for i in range(21)],
    'subsample': [0.5, 0.8, 1]}
model = GridSearchCV(
    estimator=regressor,
    param_grid=hyperparams_grid,
    cv=FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Display the optimal hyperparameters
print('\n\nBest hyperparameters: {}'.format(model.best_params_))

# Make predictions
y_pred = model.predict(X_test)
print(f'\ny_pred shape: {y_pred.shape}')

# Display the metrics for predictions
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1))
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display the metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    np.array(pred_target).flatten(),
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(
    np.array(test_set.Target), np.array(pred_target).flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=np.array(pred_target).flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()
