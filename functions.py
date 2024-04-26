"""
===============================================================================
This file contains all the functions for this Time Series Regression project.
===============================================================================
"""
# Standard library
import csv

# Other libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sktime.performance_metrics import forecasting
from sklearn import metrics



def load_dataset(file_path):
    """This function loads a csv file and finds the type of separators by
     sniffing the file.

    Parameter
    ---------
    file_path: str
               The csv file path.

    Return
    ------
    dataset: pd.DataFrame
             The loaded dataset.
    """
    with open(file_path, 'r') as csvfile:
        separator = csv.Sniffer().sniff(csvfile.readline()).delimiter
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=separator,
        encoding_errors='ignore',
        on_bad_lines='skip')
    return dataset


def build_lstm_model(hp):
    """
    This function builds a TensorFlow LSTM (Long Short-Term Memory network)
    model with hyperparameters to be optimised in order to find the best model.

    Parameter
    ---------
    hp: keras_tuner.HyperParameters
        Hyperparameters to optimise.

    Return
    ------
    model: tf.keras.Model
           The optimised model.
    """
    # Instantiate the model
    model = Sequential()
    model.add(LSTM(
        hp.Int('units', 10, 1024),
        hp.Choice('activation', values=['elu', 'relu', 'tanh']),
        return_sequences=True))
    model.add(LSTM(
        hp.Int('units', 10, 1024),
        hp.Choice('activation', values=['elu', 'relu', 'tanh']),
        return_sequences=False))
    model.add(Dense(1))

    # Compile the model
    model.compile(
        loss=hp.Choice(
            'loss', values=['mean_squared_error', 'mean_absolute_error']),
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


def display_tensorflow_history(model_history, loss, metric):
    """
    This function plots training loss and training scores, training and
    validation scores, and training and validation losses based on
    the history of a TensorFlow model.

    Parameters
    ----------
    - model_history: tf.keras.callbacks.History
                     The history of the model.
    - loss: str
            The name of the optimisation metric.
    - metric: str
              The name of the evaluation metric.
    """
    score = model_history.history[metric]
    val_score = model_history.history['val_' + metric]
    loss_score = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # Plot training loss and training score
    plt.figure(figsize=(8, 5))
    plt.plot(loss_score, color='tab:blue')
    plt.plot(score, color='tab:red')
    plt.title('History of the model training')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend([loss, metric], loc='best')
    plt.grid(True)
    plt.show()

    # Plot training and validation scores
    plt.figure(figsize=(8, 5))
    plt.plot(score, label='Training ' + metric, color='tab:blue')
    plt.plot(val_score, label='Validation ' + metric, color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Plot training and validation losses
    plt.figure(figsize=(8, 5))
    plt.plot(loss_score, label='Training ' + loss, color='tab:blue')
    plt.plot(val_loss, label='Validation ' + loss, color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.title(f'Training and Validation {loss.capitalize()}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def display_sklearn_metrics(y_test, y_pred, SEED):
    """
    This function displays the values of the Scikit-learn metrics for
    regression and plots the actual and residuals values compared to
    predicted values.

    Parameters
    ----------
    - y_test: np.ndarray, pd.Series, or pd.DataFrame
              The test set.
    - y_pred: np.ndarray, pd.Series, or pd.DataFrame
              The forecast or predictions.
    - SEED: int or RandomState
            The random state value.
    """
    mse = metrics.mean_squared_error(y_test, y_pred)
    print('\nMSE: {:.3f}'.format(mse))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print('MAE: {:.3f}'.format(mae))
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    print('MAPE: {:.3f}'.format(mape))
    mdae = metrics.median_absolute_error(y_test, y_pred)
    print('MdAE: {:.3f}'.format(mdae))
    if np.where(y_test < 0)[0].size == 0 and np.where(y_pred < 0)[0].size == 0:
        print('MSLE: {:.3f}'.format(
            metrics.mean_squared_log_error(y_test, y_pred)))
    elif np.where(y_test < 0)[0].size > 0:
        print('Impossible to compute MSLE because the test set contains '
              'negative values.')
    elif np.where(y_pred < 0)[0].size > 0:
        print('Impossible to compute MSLE because forecasts or predictions '
              'contain negative values.')
    print('Maximum residual error: {:.3f}'.format(
        metrics.max_error(y_test, y_pred)))
    print('Explained variance score: {:.3f}'.format(
        metrics.explained_variance_score(y_test, y_pred)))
    print('R²: {:.3f}'.format(metrics.r2_score(y_test, y_pred)))

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    metrics.PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=SEED)
    axs[0].set_title('Actual vs Predicted values')
    axs[0].grid(True)
    metrics.PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=SEED)
    axs[1].set_title('Residuals vs Predicted Values')
    axs[1].grid(True)
    fig.suptitle('Plotting the results of predictions')
    plt.tight_layout()
    plt.show()


def display_sktime_metrics(y_test, y_pred, y_train, PERIOD):
    """
    This function calculates and displays the values of the Sktime metrics
    for regression.

    Parameters
    ----------
    - y_test: np.ndarray, pd.Series, or pd.DataFrame
              The test set.
    - y_pred: np.ndarray, pd.Series, or pd.DataFrame
              The forecast or predictions.
    - y_train: np.ndarray, pd.Series, or pd.DataFrame
               The train set.
    - PERIOD: int
              The seasonal period of the Time Series data.
    """
    print('\nMASE: {:.3f}'.format(forecasting.mean_absolute_scaled_error(
        y_true=y_test, y_pred=y_pred, y_train=y_train, sp=PERIOD)))
    print('MdASE: {:.3f}'.format(forecasting.median_absolute_scaled_error(
        y_true=y_test, y_pred=y_pred, y_train=y_train, sp=PERIOD)))
    print('MSSE: {:.3f}'.format(forecasting.mean_squared_scaled_error(
        y_true=y_test, y_pred=y_pred, y_train=y_train, sp=PERIOD)))
    print('MdSSE: {:.3f}'.format(forecasting.median_squared_scaled_error(
        y_true=y_test, y_pred=y_pred, y_train=y_train, sp=PERIOD)))
    print('MAE: {:.3f}'.format(forecasting.mean_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdAE: {:.3f}'.format(forecasting.median_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('MSE: {:.3f}'.format(forecasting.mean_squared_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdSE: {:.3f}'.format(forecasting.median_squared_error(
        y_true=y_test, y_pred=y_pred)))
    print('MAPE: {:.3f}'.format(forecasting.mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdAPE: {:.3f}'.format(forecasting.median_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('MSPE: {:.3f}'.format(forecasting.mean_squared_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('MdSPE: {:.3f}'.format(forecasting.median_squared_percentage_error(
        y_true=y_test, y_pred=y_pred)))
    print('GMAE: {:.3f}'.format(forecasting.geometric_mean_absolute_error(
        y_true=y_test, y_pred=y_pred)))
    print('GMSE: {:.3f}'.format(forecasting.geometric_mean_squared_error(
        y_true=y_test, y_pred=y_pred)))
