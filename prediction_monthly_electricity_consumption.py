"""
===============================================================================
Time Series Regression project: Prediction of monthly consumption of
electricity in France
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 Sktime
       3.1.1 StatsForecastAutoARIMA
       3.1.2 SARIMAX
       3.1.3 StatsForecastAutoTheta
       3.1.4 STLForecaster
       3.1.5 StatsForecastMSTL
       3.1.6 AutoETS
       3.1.7 StatsForecastAutoCES
       3.1.8 AutoREG
       3.1.9 Naive Forecaster
       3.1.10 Croston
       3.1.11 Prophet
       3.1.12 ProphetPiecewiseLinearTrendForecaster
   3.2 PyCaret
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ydata_profiling
import statsmodels
import pmdarima as pm
import sktime
import pycaret

from ydata_profiling import ProfileReport
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf
from pmdarima.arima.utils import ndiffs
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.lag import Lag
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import (PolynomialTrendForecaster, STLForecaster,
                                      ProphetPiecewiseLinearTrendForecaster)
from sktime.forecasting.statsforecast import (StatsForecastAutoARIMA,
                                              StatsForecastAutoTheta,
                                              StatsForecastMSTL,
                                              StatsForecastAutoCES)
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.croston import Croston
from sktime.forecasting.fbprophet import Prophet
from pycaret.time_series import *
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))
print('Pmdarima: {}'.format(pm.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('PyCaret: {}'.format(pycaret.__version__))


# Constants
SEED = 0
PERIOD = 12
SP = 12
FOLDS = 3

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
raw_dataset = load_dataset('Consommation-nationale-mensuelle-RTE.csv')

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
dataset = dataset[['Date', 'Consommation (GWh)']].rename(
    columns={'Consommation (GWh)': 'Consumption'})
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(
    by=['Date'], ascending=True).reset_index(drop=True).set_index('Date')
dataset.index = pd.PeriodIndex(dataset.index, freq='M')
dataset = dataset[['Consumption']]

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

# Display the monthly consumption of electricity in France
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    dataset,
    markers='.',
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Monthly consumption of electricity in France from '
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

# Split the dataset into training and test sets
train_set, test_set = temporal_train_test_split(y=dataset, test_size=0.2)
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
    train_set,
    test_set,
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Monthly consumption of electricity in France from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# Display autocorrelation
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(acf(train_set.Consumption))
ax.set_xlabel('Lags')
ax.set_title('Autocorrelation')
plt.show()

# Transformation of the dataset to create exogenous variables.
# Creation of 13 lags (features)
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
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Monthly consumption of electricity in France from '
             f'{X.index.min()} to {X.index.max()}')
ax.legend(loc='best')
plt.show()

# Split the dataset into training and test sets
X_train = X.iloc[0:int(0.8 * X.shape[0]),]
X_test = X.iloc[X_train.shape[0]:X.shape[0],]

# Display head and the tail of the datasets
print(f'\nTraining set shape: {X_train.shape}')
print(pd.concat([X_train.head(), X_train.tail()]))
print(f'\nTest set shape: {X_test.shape}')
print(pd.concat([X_test.head(), X_test.tail()]))

# Time Series X_train report
profile = ProfileReport(df=X_train, tsmode=True, title='X_train report')
profile.to_file('X_train_report.html')

# Time Series X_test report
profile = ProfileReport(df=X_test, tsmode=True, title='X_test report')
profile.to_file('X_test_report.html')

# Display training and test targets
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Monthly consumption of electricity in France from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='best')
plt.show()



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
      f"the KPSS test: {ndiffs(np.array(train_set), test='kpss')}")

# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('StatsForecastAutoARIMA', StatsForecastAutoARIMA(
        sp=PERIOD,
        d=ndiffs(np.array(train_set), test='kpss'),
        seasonal=True,
        information_criterion='bic',
        n_jobs=-1,
        trend='ct',
        trace=True))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# The transformed dataset X with exogenous features
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('StatsForecastAutoARIMA', StatsForecastAutoARIMA(
        sp=PERIOD,
        d=ndiffs(np.array(X_train.Target), test='kpss'),
        seasonal=True,
        information_criterion='bic',
        n_jobs=-1,
        trend='ct',
        trace=True))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=X_train.Target,
    X=X_train.drop(['Target'], axis=1),
    fh=ForecastingHorizon(X_test.index, is_relative=False),
    X_pred=X_test.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(X_test.Target).shape)
display_sktime_metrics(
    np.array(X_test.Target),
    np.array(forecast),
    np.array(X_train.Target),
    SP)
display_sklearn_metrics(np.array(X_test.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.2 SARIMAX
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('SARIMAX', SARIMAX(
        order=(1, ndiffs(np.array(train_set), test='kpss'), 0),
        seasonal_order=(0, 0, 1, PERIOD),
        trend='ct',
        enforce_stationarity=True,
        enforce_invertibility=True,
        random_state=SEED))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# The transformed dataset X with exogenous features
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('SARIMAX', SARIMAX(
        order=(2, ndiffs(np.array(X_train.Target), test='kpss'), 0),
        seasonal_order=(0, 0, 1, PERIOD),
        trend='ct',
        enforce_stationarity=True,
        enforce_invertibility=True,
        random_state=SEED))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=X_train.Target,
    X=X_train.drop(['Target'], axis=1),
    fh=ForecastingHorizon(X_test.index, is_relative=False),
    X_pred=X_test.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(X_test.Target).shape)
display_sktime_metrics(
    np.array(X_test.Target),
    np.array(forecast),
    np.array(X_train.Target),
    SP)
display_sklearn_metrics(np.array(X_test.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.3 StatsForecastAutoTheta
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('StatsForecastAutoTheta', StatsForecastAutoTheta(
        season_length=PERIOD, decomposition_type='additive'))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.4 STLForecaster
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('STLForecaster', STLForecaster(sp=SP, seasonal=17, robust=True))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.5 StatsForecastMSTL
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('StatsForecastMSTL', StatsForecastMSTL(season_length=PERIOD))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.6 AutoETS
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('AutoETS', AutoETS(
        trend='add',
        sp=SP,
        auto=True,
        information_criterion='aic',
        allow_multiplicative_trend=True,
        n_jobs=-1,
        random_state=SEED))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.7 StatsForecastAutoCES
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('StatsForecastAutoCES', StatsForecastAutoCES(
        season_length=PERIOD, model='Z'))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.8 AutoREG
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('AutoREG', AutoREG(lags=None, trend='ct', seasonal=True, period=PERIOD))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# The transformed dataset X with exogenous features
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('AutoREG', AutoREG(lags=None, trend='ct', seasonal=True, period=PERIOD))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=X_train.Target,
    X=X_train.drop(['Target'], axis=1),
    fh=ForecastingHorizon(X_test.index, is_relative=False),
    X_pred=X_test.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(X_test.Target).shape)
display_sktime_metrics(
    np.array(X_test.Target),
    np.array(forecast),
    np.array(X_train.Target),
    SP)
display_sklearn_metrics(np.array(X_test.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.9 Naive Forecaster
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('Naive Forecaster', NaiveForecaster(sp=SP))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.10 Croston
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('Croston', Croston(smoothing=0.5))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()


# 3.1.11 Prophet
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=3))),
    ('Prophet', Prophet())])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# The transformed dataset X with exogenous features
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('Prophet', Prophet())])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=X_train.Target,
    X=X_train.drop(['Target'], axis=1),
    fh=ForecastingHorizon(X_test.index, is_relative=False),
    X_pred=X_test.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(X_test.Target).shape)
display_sktime_metrics(
    np.array(X_test.Target),
    np.array(forecast),
    np.array(X_train.Target),
    SP)
display_sklearn_metrics(np.array(X_test.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# 3.1.12 ProphetPiecewiseLinearTrendForecaster
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('ProphetPiecewiseLinearTrendForecaster',
     ProphetPiecewiseLinearTrendForecaster())])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train_set, fh=ForecastingHorizon(test_set.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test_set).shape)
display_sktime_metrics(
    np.array(test_set),
    np.array(forecast),
    np.array(train_set),
    SP)
display_sklearn_metrics(np.array(test_set), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set,
    test_set,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train_set.index.min()} to {test_set.index.max()}')
plt.show()

# The transformed dataset X with exogenous features
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=SP)),
    ('ProphetPiecewiseLinearTrendForecaster',
     ProphetPiecewiseLinearTrendForecaster())])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=X_train.Target,
    X=X_train.drop(['Target'], axis=1),
    fh=ForecastingHorizon(X_test.index, is_relative=False),
    X_pred=X_test.drop(['Target'], axis=1))

# Display metrics
print('\n\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(X_test.Target).shape)
display_sktime_metrics(
    np.array(X_test.Target),
    np.array(forecast),
    np.array(X_train.Target),
    SP)
display_sklearn_metrics(np.array(X_test.Target), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X_train.Target,
    X_test.Target,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (GWh)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{X_train.index.min()} to {X_test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# 3.2 PyCaret
"""
PyCaret determines other effective models to solve the business problem.
This first selection is based on the basic hyperparameters of the models.
To select the final model, the best models must be compared on the basis of
their optimised hyperparameters.
"""
# Initialisation of the setup
s = setup(
    data=dataset,
    target='Consumption',
    scale_target='minmax',
    fold=FOLDS,
    fh=int(0.2 * dataset.shape[0]),
    n_jobs=-1,
    session_id=SEED)

# Check statistical tests on the dataset
print(check_stats())

# Selection of the best model by cross-validation using basic hyperparameters
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    sort='MASE')
print(f'\n\nClassification of models:\n{best}')

# Display diagnostics
plot_model(plot='diagnostics')

# Display forecast
plot_model(best, plot='forecast')


# The transformed dataset X with exogenous features
# Initialisation of the setup
s = setup(
    data=X,
    target='Target',
    scale_target='minmax',
    scale_exogenous='minmax',
    fold=FOLDS,
    fh=int(0.2 * X.shape[0]),
    n_jobs=-1,
    session_id=SEED)

# Check statistical tests on the dataset
print(check_stats())

# Selection of the best model by cross-validation using basic hyperparameters
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    sort='MASE')
print(f'\n\nClassification of models:\n{best}')

# Display diagnostics
plot_model(plot='diagnostics')

# Display forecast
plot_model(best, plot='forecast')
