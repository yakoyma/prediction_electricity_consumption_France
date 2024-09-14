"""
===============================================================================
Time Series Regression project: Prediction of daily consumption of electricity
in France using AutoML
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 AutoKeras
   3.2 AutoGluon
   3.3 FLAML
   3.4 TPOT
   3.5 H2O
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
import sweetviz as sv
import sklearn
import sktime
import autokeras as ak
import tensorflow as tf
import autogluon
import flaml
import prophet
import tpot
import h2o

from sweetviz import analyze
from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series
from sktime.transformations.series.lag import Lag
from autokeras import StructuredDataRegressor
from tensorflow.keras.callbacks import EarlyStopping
from autogluon.tabular import TabularDataset, TabularPredictor
from flaml import AutoML
from tpot import TPOTRegressor
from h2o.automl import H2OAutoML
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('AutoKeras: {}'.format(ak.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('FLAML: {}'.format(flaml.__version__))
print('Prophet: {}'.format(prophet.__version__))
print('TPOT: {}'.format(tpot.__version__))
print('H2O: {}'.format(h2o.__version__))


# Constants
SEED = 0
PERIOD = 7
SP = 31
FOLDS = 5

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
report = analyze(source=raw_dataset, target_feat='Consommation (MW)')
report.show_html('raw_dataset_report.html')

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
report = analyze(source=dataset, target_feat='Consumption')
report.show_html('dataset_report.html')

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
ax.grid(True)
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

# Display the dataset's dimensions
print('\n\nDimensions of the dataset: {}'.format(X.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(X.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(round(X.describe(include='all'), 0))

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Time Series X dataset report
report = analyze(source=X, target_feat='Target')
report.show_html('X_dataset_report.html')

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
ax.grid(True)
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
report = analyze(source=train_set, target_feat='Target')
report.show_html('train_set_report.html')

# Time Series test set report
report = analyze(source=test_set, target_feat='Target')
report.show_html('test_set_report.html')

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
ax.grid(True)
plt.show()

# Standardisation
# Scale values between the range of 0 and 1
transformer = MinMaxScaler()
y_train = transformer.fit_transform(
    X=np.array(train_set.Target).reshape(-1, 1)).flatten()
y_test = transformer.transform(
    X=np.array(test_set.Target).reshape(-1, 1)).flatten()

scaler = MinMaxScaler()
X_train = pd.DataFrame(
    data=scaler.fit_transform(X=train_set.drop(['Target'], axis=1)),
    columns=train_set.drop(['Target'], axis=1).columns)
X_test = pd.DataFrame(
    data=scaler.transform(X=test_set.drop(['Target'], axis=1)),
    columns=test_set.drop(['Target'], axis=1).columns)
print(f'\ny_train shape: {y_train.shape}')
print(f'X_train shape: {X_train.shape}')
print(pd.concat([X_train.head(), X_train.tail()]))
print(f'\ny_test shape: {y_test.shape}')
print(f'X_test shape: {X_test.shape}')
print(pd.concat([X_test.head(), X_test.tail()]))

train = X_train.assign(Target=y_train, Date=train_set.index)
train = train.set_index('Date')
test = X_test.assign(Target=y_test, Date=test_set.index)
test = test.set_index('Date')
print(f'\ntrain shape: {train.shape}')
print(pd.concat([train.head(), train.tail()]))
print(f'\ntest shape: {test.shape}')
print(pd.concat([test.head(), test.tail()]))



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# 3.1 AutoKeras
# Train and evaluate the model
# Instantiate the model
model = StructuredDataRegressor(
    loss='mean_squared_error',
    max_trials=10,
    objective='val_loss',
    tuner='hyperband',
    overwrite=True,
    seed=SEED)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    verbose=0,
    mode='auto',
    patience=3)

# Train the model
model.fit(
    x=np.array(X_train),
    y=y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stopping_callback])

print('\n\nSummary of the model with optimised hyperparameters:')
model = model.export_model()
print(model.summary())

# Make predictions
y_pred = model.predict(np.array(X_test)).flatten()

# Evaluate the best model
scores = model.evaluate(np.array(X_test), y_test, verbose=0)
for metric, score in zip(['Loss (MSE)', 'MSE', 'MAE'], scores):
    print(f'{metric}: {score:.3f}')

# Display metrics for predictions
print('\nPredicted target shape: ', y_pred.shape)
display_sktime_metrics(y_test, np.array(y_pred), y_train, SP)
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
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    pred_target,
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), pred_target, SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target, index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()


# 3.2 AutoGluon
train_df = TabularDataset(data=train.reset_index(drop=True))
test_df = TabularDataset(data=test.reset_index(drop=True))

# Instantiate AutoML instance
automl = TabularPredictor(label='Target', eval_metric='mse').fit(
    train_data=train_df, time_limit=3600, presets='best_quality')

# Display the best model
print(f'\n\nThe best model:\n{automl.leaderboard(train_df)}')

# Make predictions
y_pred = np.array(automl.predict(test_df.drop(columns=['Target']))).flatten()

# Display metrics for predictions
print('\nPredicted target shape: ', y_pred.shape)
display_sktime_metrics(y_test, y_pred, y_train, SP)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test, index=test_set.index),
    pd.Series(data=y_pred, index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    pred_target,
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), pred_target, SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target, index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.grid(True)
plt.show()


# 3.3 FLAML
train_df = train.to_timestamp(freq='D').reset_index()
test_df = test.drop(['Target'], axis=1)
test_df = test_df.to_timestamp(freq='D').reset_index()

# Instantiate AutoML instance
automl = AutoML()
automl.fit(
    dataframe=train_df,
    label='Target',
    metric='mape',
    task='ts_forecast_regression',
    period=PERIOD,
    time_budget=300,
    n_jobs=-1,
    eval_method='auto',
    n_splits=FOLDS,
    split_type='time',
    seed=SEED,
    early_stop=True)

# Display information about the best model
print('\n\nBest estimator: {}'.format(automl.best_estimator))
print('Best hyperparameters:\n{}'.format(automl.best_config))
print('Best loss: {}'.format(automl.best_loss))
print('Training time: {}s'.format(automl.best_config_train_time))

# Make predictions
y_pred = np.array(automl.predict(test_df)).flatten()

# Display metrics for predictions
print('\nPredicted target shape: ', np.array(y_pred).shape)
display_sktime_metrics(y_test, np.array(y_pred), y_train, SP)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test, index=test.index),
    pd.Series(data=y_pred, index=test.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test.index.min()} to {test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    pred_target,
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), pred_target, SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target, index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()


# 3.4 TPOT
# Instantiate the TPOT model
model = TPOTRegressor(
    generations=100,
    population_size=100,
    cv=FOLDS,
    n_jobs=-1,
    random_state=SEED,
    early_stop=True,
    verbosity=2)
model.fit(np.array(X_train), y_train)

# Make predictions
y_pred = np.array(model.predict(np.array(X_test))).flatten()

# Display metrics for predictions
print('\n\nPredicted target shape: ', y_pred.shape)
display_sktime_metrics(y_test, y_pred, y_train, SP)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
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
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    pred_target,
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), pred_target, SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target, index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()


# 3.5 H2O
# Initialisation: start the cluster
h2o.init()

# Instantiate H2O AutoMl
aml = H2OAutoML(nfolds=FOLDS, max_models=100, seed=SEED)

# Training H2O AutoMl
aml.train(
    x=list(train.drop(['Target'], axis=1)),
    y='Target',
    training_frame=h2o.H2OFrame(train))

# Display the AutoML Leaderboard
lb = aml.leaderboard
print(f'\n\nThe Leaderboard:\n{lb.head(rows=lb.nrows)}')

# Display the best model
print(f'The best model:\n{aml.leader}')

# Display the model performance
performance = aml.leader.model_performance(h2o.H2OFrame(test))
print(performance)

# Make predictions
y_pred = aml.leader.predict(h2o.H2OFrame(test))

# Convert H2O frame into Pandas DataFrame
y_pred = np.array(y_pred.as_data_frame()).flatten()

# Display metrics for predictions
print('\nPredicted target shape: ', np.array(y_pred).shape)
display_sktime_metrics(y_test, np.array(y_pred), y_train, SP)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test, index=test.index),
    pd.Series(data=y_pred, index=test.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test.index.min()} to {test.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
print('\nPredicted target shape: ', pred_target.shape)
print('Actual target shape: ', np.array(test_set.Target).shape)

# Display metrics for actual values
display_sktime_metrics(
    np.array(test_set.Target),
    pred_target,
    np.array(train_set.Target),
    SP)
display_sklearn_metrics(np.array(test_set.Target), pred_target, SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target, index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Consumption (MW)',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True)
plt.show()
