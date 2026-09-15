"""
===============================================================================
Time Series Project: Application for Forecasting Daily Electricity Consumption 
in Metropolitan France Excluding Corsica using NeuralForecast Library
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Functions
   3.2 AutoNHITS Estimator
   3.3 AutoLSTM Estimator
"""
# Standard libraries
import random
import platform

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels
import darts
import neuralforecast


from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf
from darts import TimeSeries
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.core import NeuralForecast
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))
print('Darts: {}'.format(darts.__version__))
print('NeuralForecast: {}'.format(neuralforecast.__version__))



# Constants
SEED = 1
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
PERIOD = 365
LAGS = 365
FORECAST_HORIZON = 7

# Set the random seed for reproducibility
random.seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)

# Set the default Seaborn style
sns.set_style('whitegrid')



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print('\n\n\n1. Load the dataset')

# Load the dataset
INPUT_CSV = 'datasets/consommation-nationale-quotidienne.csv'
raw_dataset = load_dataset(file_path=INPUT_CSV, encoding='utf-8')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print('\n\n\n2. Feature Engineering')


# Split the dataset into train and test sets
train_dataset = raw_dataset.iloc[:-FORECAST_HORIZON]
test_dataset = raw_dataset.iloc[-FORECAST_HORIZON:]

# Display datasets information and description
dataset_info_description(dataset=train_dataset, max_rows=150)
dataset_info_description(dataset=test_dataset, max_rows=5)


# Save the train dataset in CSV format
OUTPUT_CSV = 'datasets/daily consumption/neuralforecast/train_dataset.csv'
train_dataset.to_csv(OUTPUT_CSV, index=False)


# Create Darts train and test series from the Pandas DataFrames
train = TimeSeries.from_dataframe(
    df=train_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)
test = TimeSeries.from_dataframe(
    df=test_dataset,
    time_col='Date',
    value_cols=['Consumption (MW)']
)


# Rename the features to comply with the NeuralForecast library’s requirements
train_dataset = train_dataset.rename(columns={
    'Date': 'ds', 'Consumption (MW)': 'y'})
train_dataset['ds'] = pd.to_datetime(train_dataset['ds'])
train_dataset['unique_id'] = 'consommation'
train_dataset = train_dataset[['unique_id', 'ds', 'y']]
test_dataset = test_dataset.rename(columns={
    'Date': 'ds', 'Consumption (MW)': 'y'})
test_dataset['ds'] = pd.to_datetime(test_dataset['ds'])
test_dataset['unique_id'] = 'consommation'
test_dataset = test_dataset[['unique_id', 'ds', 'y']]


# Tests to determine whether the dataset is stationary and/or invertible
print(f'\n\nStationarity test result: '
      f'{ArmaProcess(train.univariate_values()).isstationary}')
print(f'Invertibility test result: '
      f'{ArmaProcess(train.univariate_values()).isinvertible}')


# Plot autocorrelation and partial autocorrelation
plt.figure(figsize=(12, 6))
plt.plot(
    acf(x=train.values(), nlags=LAGS),
    label='Autocorrelation',
    color='tab:blue'
)
plt.plot(
    pacf(x=train.values(), nlags=LAGS),
    label='Partial Autocorrelation',
    color='tab:red'
 )
plt.title('Autocorrelation and Partial Autocorrelation')
plt.xlabel('Lags')
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print('\n\n\n3. Machine Learning')


# 3.1 Functions
print(f'\n\n3.1 Functions')

def get_model_forecasts(estimator, estimator_name: str):    
    """This function trains a NeuralForecast estimator, generates forecasts, 
    evaluates performance, and plots the results.

    Args:
        estimator (neuralforecast.auto): the estimator to train and use for 
                                         forecasts
        estimator_name (str): the name of the estimator
    """
    
    print(f'\n\n{estimator_name}:')
    
    # Instantiate the model
    model = NeuralForecast(models=[estimator], freq='D')

    # Train the model
    model.fit(df=train_dataset)
    
    # Model path
    model_path = f'models/daily consumption/neuralforecast/{estimator_name}'
    
    # Model persistence
    model.save(path=model_path, overwrite=True)
    
    # Load the pre-trained model
    model = NeuralForecast.load(path=model_path)
    
    # Generate forecasts
    forecasts_result = model.predict(h=len(test_dataset))
    print(forecasts_result)
    forecasts_dataset = forecasts_result.reset_index()[
        ['ds', f'{estimator_name}']]
    forecasts_dataset = forecasts_dataset.rename(
        columns={'ds': 'Date', f'{estimator_name}': 'Consumption (MW)'})

    # Create the Darts forecasts series from the Pandas DataFrame
    forecasts = TimeSeries.from_dataframe(
        df=forecasts_dataset,
        time_col='Date',
        value_cols=['Consumption (MW)']
    )

    # Evaluation
    print('\nEvaluation')
    evaluate_time_series(
        test=test, forecasts=forecasts, train=train, period=PERIOD)
    evaluate_regression(
        y_test=test.values(), y_pred=forecasts.values(), seed=SEED)
    
    # Plot actual values
    plt.figure(figsize=(12, 6))
    train.plot(label='Train', color='tab:blue')
    test.plot(label='Test', color='tab:red')
    forecasts.plot(label='Forecasts', color='tab:green')
    plt.title(label=f'Actual Values vs Forecasts')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


def autonhits_estimator_optimisation(trial):
    """This function generates a hyperparameter configuration for 
    the AutoNHITS estimator using Optuna.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        dict: A dictionary containing AutoNHITS hyperparameters for
              optimisation
    """
    
    return {
        'input_size': trial.suggest_categorical(
            'input_size', (365, 365 * 2, 365 * 3)),
        'n_blocks': trial.suggest_categorical(
            'n_blocks', (3 * [2], 3 * [3], 3 * [4], 3 * [5], 3 * [6])),
        'mlp_units': trial.suggest_categorical(
            'mlp_units', (3 * [[256, 256]], 3 * [[512, 512]])),
        'n_pool_kernel_size': trial.suggest_categorical(
            'n_pool_kernel_size', ([30, 7, 1], [365, 30, 7, 1])),
        'n_freq_downsample': trial.suggest_categorical(
            'n_freq_downsample', ([30, 7, 1], [365, 30, 7, 1])),
        'max_steps': trial.suggest_int('max_steps', low=100, high=1000),        
        'learning_rate': trial.suggest_float(
            'learning_rate', low=1e-4, high=1e-2, log=False),
        'batch_size': 32,
        'windows_batch_size': trial.suggest_int(
            'max_steps', low=256, high=1024),
        'scaler_type': trial.suggest_categorical(
            'scaler_type', ('standard', 'minmax', 'robust'))
    }


def autolstm_estimator_optimisation(trial):
    """This function generates a hyperparameter configuration for 
    the AutoLSTM estimator using Optuna.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        dict: A dictionary containing AutoLSTM hyperparameters for
              optimisation
    """
    
    return {
        'input_size': trial.suggest_categorical(
            'input_size', (365, 365 * 2, 365 * 3)),
        'encoder_n_layers': trial.suggest_int(
            'encoder_n_layers', low=2, high=6),
        'encoder_hidden_size': trial.suggest_int(
            'encoder_hidden_size', low=128, high=1024),
        'decoder_hidden_size': trial.suggest_int(
            'decoder_hidden_size', low=128, high=1024),
        'batch_size': 32,
        'max_steps': trial.suggest_int('max_steps', low=10, high=300),
        'learning_rate': trial.suggest_float(
            'learning_rate', low=1e-4, high=1e-2, log=False),
        'scaler_type': trial.suggest_categorical(
            'scaler_type', ('standard', 'minmax', 'robust'))
    }


# 3.2 AutoNHITS Estimator
print(f'\n\n3.2 AutoNHITS Estimator')

# Instantiate the estimator
estimator = AutoNHITS(
    h=len(test_dataset),
    config=autonhits_estimator_optimisation,
    backend='optuna'
 )
get_model_forecasts(estimator=estimator, estimator_name='AutoNHITS')


# 3.3 AutoLSTM Estimator
print(f'\n\n3.3 AutoLSTM Estimator')

# Instantiate the estimator
estimator = AutoLSTM(
    h=len(test_dataset),
    config=autolstm_estimator_optimisation,
    backend='optuna'
)
get_model_forecasts(estimator=estimator, estimator_name='AutoLSTM')
