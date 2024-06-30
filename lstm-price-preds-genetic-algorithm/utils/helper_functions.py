import zipfile
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as MSE

def unzip_file(filename:str):
  """
  Unzips filename into the current working directory.
  """
  try:
    with zipfile.ZipFile(filename) as zip_ref:
      zip_ref.extractall()
      print(f"Extract zip file completely from {filename}.")
  except FileNotFoundError:
      print(f" {FileNotFoundError.__name__} --> The filepath is not found! ")

# Load NASDAQ_100_Data_From_2010 dataset --> sep = '\t' because all information are in one column separated by whitespace
# parse_dates=True --> make "Date" column be available for calculation.
def load_data(csv_file:str, stock_code:str=None, test_data=False) -> pd.DataFrame:
  """
  Load data and return specific stock index.

  Args:
    csv_file: File path expected to be read.
    stock_code: Stock code to extract. 
  Returns:
    If train data, return DataFrame; if test data, return tuple (dates, DataFrame).
  """
  # Prepare dataset
  if not test_data:
    stock_df = pd.read_csv(csv_file, sep='\t', index_col='Date', parse_dates=True)
    dataset = stock_df[stock_df.Name == stock_code].loc[:, 'Adj Close'].values # Adj close
    dataset = np.reshape(dataset, (-1, 1))
    
    return dataset
  else:
    stock_df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
    dataset = dataset = stock_df.loc[:, 'Adj Close'].values
    dates = stock_df.index.values.astype('datetime64[D]')
    dataset = np.reshape(dataset, (-1, 1))

    return (dates, dataset)

class Data_Preprocessor:
  def __init__(self, dataset, test_dataset, look_back=1, rescaling='MinMaxScaling'):
    """
    Preprocess data on following steps:
      1. Create dataset: The forward days as feature, the last day as label.
      2. Data rescaling: Min-Max scaling or Standard scaling.
      3. Create training and testing datasets: Split train and test dataset.
      4. Tensor conversion: Convert data to be able to be fitted as input shape of LSTM.

    Args:
      dataset (DataFrame): One stock dataset including [Open	High	Low	Close] as columns and [Date] as index.
      look_back (int): Days to be as feature, range: [1, len(dataset)]
      rescaling (str): Rescale methods, MinMaxScaling or StandardScaling.
    """
    self.dataset = dataset
    self.test_dataset = test_dataset
    self.look_back = look_back
    self.rescaling = rescaling
    self.scaler = MinMaxScaler() if self.rescaling == 'MinMaxScaling' else StandardScaler()

  def _create_dataset(self, dataset):
    """
    Create features and labels.
    Series data uses previous data as features, last data as label.

    Args:
      dataset: Dataset expected to be created.
    Return:
      Numpy array for X and y datasets.
    """
    # features
    X_data = np.array([dataset[i: i+self.look_back] for i in range(len(dataset) - self.look_back)])

    # labels
    y_data = dataset[self.look_back:]

    return X_data, y_data

  def preprocessing_data(self, valid_size=30, test_data=False) -> tuple:
    """
    Preprocess data before training.

    Args:
      test_loc: Numbers of testing data.
    Return:
      Train and Test datasets: X_train, X_test, y_train, y_test
    """
    if not test_data:
      # Create dataset
      dataset_scaled = self.scaler.fit_transform(self.dataset) # data rescaling
      X, y = self._create_dataset(dataset_scaled)
      
      # Create training and testing dataset
      X_train, X_valid, y_train, y_valid = X[:-valid_size], X[-valid_size:], y[:-valid_size], y[-valid_size:]

      # Convert to (sample, time step, feature) tensor --> as input shape of LSTM model
      X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
      X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

      return (X_train, X_valid, y_train, y_valid)
    else:
      dataset_scaled = self.scaler.fit_transform(self.test_dataset)
      X_test, y_test = self._create_dataset(dataset_scaled)
      X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

      return (X_test, y_test)


def plot_result(y_test, y_pred, title, dates):
    """
    Plot a figure based on model predictions and true labels.

    Args:
        y_test: labels
        y_pred: model predictions
        title: title of the figure.
        dates: dates of data.
    """
    plt.plot(dates, y_test, color='red', label='Real Stock Price')
    plt.plot(dates, np.squeeze(y_pred), color='blue', label='Predicted Stock Price')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('time')
    plt.ylabel('AAPL Price')
    plt.legend()
    plt.show()


def plot_loss_lr(history:dict):
    """
    Plot training loss curve with learning rate.

    Args:
      history: History during model training.
    """
    # plot figure
    fig, ax1 = plt.subplots()

    # left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(history['loss'], label='Training Loss', color='blue', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', marker='o')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper right')

    # right y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Learning Rate')  # we already handled the x-label with ax1
    ax2.plot(history['lr'], label='Learning Rate', color='green', ls='--')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training History')
    plt.show()


def LSTM_model(input_shape):
  """
  Create a LSTM model and compile it.
  """
  # Input Layer
  input_layer = Input(shape=input_shape, name='input_layer')

  # LSTM and Dropout layers
  LSTM_1 = LSTM(32, return_sequences=True, activation='tanh', name='lstm_layer_1')(input_layer)
  Dropout_1 = Dropout(0.2, name='dropout_layer_1')(LSTM_1)
  LSTM_2 = LSTM(16, return_sequences=True, activation='tanh', name='lstm_layer_2')(Dropout_1)
  Dropout_2 = Dropout(0.2, name='dropout_layer_2')(LSTM_2)
  LSTM_3 = LSTM(8, activation='tanh', name='lstm_layer_4')(Dropout_2)
  Dropout_3 = Dropout(0.2, name='dropout_layer_4')(LSTM_3)

  # Fully-connected layer
  Dense_layer = Dense(32, name='dense_layer', activation='ReLU')(Dropout_3)
  Dropout_5 = Dropout(0.1, name='dropout_layer_5')(Dense_layer)
  Dense_layer_2 = Dense(16, name='dense_layer_2', activation='ReLU')(Dropout_5)
  Dropout_6 = Dropout(0.1, name='dropout_layer_6')(Dense_layer_2)
  Dense_layer_3 = Dense(8, name='dense_layer_3', activation='ReLU')(Dropout_6)

  # Output Layer
  output_layer = Dense(1, name='output_layer')(Dense_layer_3) # Prediction problem, only one unit of neuron

  # Build model
  model = Model(inputs=input_layer, outputs=output_layer, name='LSTM_Stock_Predictor')

  return model


def evaluate_model(dataset:np.ndarray,
                   test_dataset:np.ndarray,  
                   epochs:int, 
                   look_back:int) -> tuple:
  """
  Evaluate model by following steps:
  1. Create dataset: training data and testing data.
  2. Create and compile LSTM model.
  3. Fit the model
  4. Model prediction and evaluation.
  5. Calculate loss.

  Args:
    dataset: Data expected to be trained.
    epochs: Number of event that complete pass of the training dataset through algorithm.
    look_back: The window of the data for creating features.
  Returns:
    Model results containing loss value, predictions, and training history.
  """
  # Record training info
  model_results = {}

  # Data preprocessing
  data_preprocessor = Data_Preprocessor(dataset, test_dataset, look_back=look_back, rescaling='MinMaxScaling')
  X_train, X_valid, y_train, y_valid = data_preprocessor.preprocessing_data()
  X_test, y_test_scaled = data_preprocessor.preprocessing_data(test_data=True)

  # Build model
  model = LSTM_model(input_shape=X_train.shape[1:])

  # Compile model
  model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))

  # Reduce learning rate
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.6, patience=3, min_lr=1e-6, mode='min', verbose=1)

  # EarlyStopping
  early_stopper = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

  # Fit model
  model_history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=32, 
                            validation_data=(X_valid, y_valid), 
                            callbacks=[reduce_lr, early_stopper])

  # Evaluate model
  model_predictions = model.predict(X_test)
  y_pred = data_preprocessor.scaler.inverse_transform(model_predictions)
  y_test = data_preprocessor.scaler.inverse_transform(y_test_scaled)
  model_loss = MSE(y_test, y_pred)

  stopped_epoch = early_stopper.stopped_epoch
  if stopped_epoch:
    epochs = stopped_epoch

  model_results['Predictions'] = [y_pred] # model predictions
  model_results['Evaluation'] = np.round(model_loss, 3) # model evaluation
  model_results['History'] = model_history.history
  model_results['True_Labels'] = y_test
  model_results['Epochs'] = epochs
  model_results['Window'] = look_back

  # clear_session
  clear_session()
  del model
  return model_results


def save_results(results:dict, file_path:str):
  """
  Save training results as pickle file.

  Args:
    results: Results of training.
    file_path: File path for saving.
  """
  with open(file_path, 'wb') as file:
    pickle.dump(results, file)

def load_results(file_path:str) -> dict:
  """
  Load training results from pickle file.

  Args:
    file_path: File path of training results.
  """
  with open(file_path, 'rb') as file:
    results = pickle.load(file)
  
  return results


if __name__ ==  '__main__':
  from tensorflow import convert_to_tensor

  # Read data
  dataset = load_data('C:/Users/User/Desktop/tf_hyperparameters_searching/lstm-price-preds-genetic-algorithm/NASDAQ_100_Data_From_2010.csv', 'AAPL')

  # Data preprocessing
  data_preprocessor = Data_Preprocessor(dataset, look_back=7, rescaling='MinMaxScaling')
  X_train, X_test, y_train, y_test = data_preprocessor.preprocessing_data()
  print(X_train.shape)
  print(convert_to_tensor(X_train))