import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def show_random_image(dataset, labels, title:str=""):
  """
  Show random images from training and testing data from mnist dataset.

  Args:
    dataset: The dataset passed to plot to display.
    labels: The label dataset.
    title: The title displayed on each image.
  """
  # Select image
  img_indices = np.random.randint(0, len(dataset), size=16)

  # Plot a image
  plt.figure(figsize=(10, 10))
  for i, img_index in enumerate(img_indices):
    plt.subplot(4, 4, i+1)
    plt.imshow(dataset[img_index], cmap='gray')
    plt.title(f"{title}, label: {labels[img_index]}")
    plt.axis("off")
  plt.show()

def preprocessing_data(X_train, X_test, y_train, y_test):
  """
  Preprocess data on following steps:
  1. reshape every image size to [None, 28, 28, 1].
  2. Normalize pixel number of each images in range 0-255.
  3. Convert categorical labels into one-hot encoding.

  Args:
    X_train: Training set from mnist dataset.
    X_test: Testing set from mnist dataset.
    y_train: Training labels from mnist dataset.
    y_test: Testing labels from mnist dataset.
  """

  # Reshape data as needed by the model
  X_train = np.reshape(X_train, (-1, 28, 28, 1))
  X_test = np.reshape(X_test, (-1, 28, 28, 1))

  # print out
  datasets = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
  for title, data in datasets.items():
    print(f"{title} shape: {data.shape}")
  print('-'*30)

  # Normalization
  X_train = X_train / 255
  X_test = X_test / 255

  # One-hot encoding
  print(f"Before one-hot encoding: {y_train[0]}")
  y_train = to_categorical(y_train, num_classes=10)
  y_test = to_categorical(y_test, num_classes=10)
  print(f"After one-hot encoding: {y_train[0]}")

  return X_train, X_test, y_train, y_test

def plot_loss_curves(history):
  """
  Returns seperate loss curve for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(loss))

  plt.figure(figsize=(10, 5))
  # plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

def export_grid_results(grid_result):
  """
  Export a dataframe containing cv results in gridsearchcv object.

  Args:
    grid_result: gridsearch object after training.
  """

  # Information during training
  info = copy.deepcopy(grid_result.cv_results_)

  # Pop best params and insert them into dataframe
  parameter_combination = info.pop('params')
  parameter_index_names = []
  for parameters in parameter_combination:
    aseemble_name = ''
    for (key, value) in parameters.items():
      aseemble_name += str(value) + '_'
    parameter_index_names.append(aseemble_name)

  grid_df = pd.DataFrame(data=info, index=parameter_index_names)

  return grid_df

def plot_grid_results(grid_result_df, fit_time=False, score_time=False, figsize=(8, 6), lim=(0, 1)):
  """
  Plot horizontal bar plot from fitting time, scoring time or train/test score values.

  Args:
    grid_result_df: Gridsearch results
    fit_time (bool): Whether plot the fitting time
    score_time (bool): Whether plot the score time
    figsize: Set figure size
    lim: Set bar plot limitation
  """

  # mean fitting time
  if fit_time:
    data = grid_result_df.sort_values(by='mean_fit_time')
    height_fit = [x for x in range(len(data.index))]

    data.mean_fit_time.plot.barh(
        title='Fitting time with different hyperparameters',
        xlabel='Seconds',
        edgecolor='k')
    print(f"[{data.mean_fit_time.min()} - {data.mean_fit_time.max()}]")

    # Standard deveation
    plt.errorbar(
        x=data.mean_fit_time,
        y=height_fit,
        xerr=data.std_fit_time,
        fmt='>',
        color='k',
        alpha=0.8);

  # mean score time
  elif score_time:
    data = grid_result_df.sort_values(by='mean_score_time')
    height_score = [x for x in range(len(data.index))]

    data.mean_score_time.plot.barh(
        title='Scoring time with different hyperparameters',
        xlabel='Seconds',
        color='#3edef0',
        edgecolor='k')

    # Standard deveation
    plt.errorbar(
        x=data.mean_score_time,
        y=height_score,
        xerr=data.std_score_time,
        fmt='>',
        color='k',
        alpha=0.8);
    print(f"[{data.mean_score_time.min()} - {data.mean_score_time.max()}]")

  # mean train/test scores
  else:
    plt.figure(figsize=figsize)

    data = grid_result_df.sort_values(by='mean_test_score')
    height=0.25
    list_index = data.index
    height_train = [x + height/2 for x in range(len(list_index))]
    height_test = [x - height/2 for x in range(len(list_index))]


    # train score
    plt.barh(
        y=height_train,
        width=data.mean_train_score,
        height=height,
        label='train_score',
        color='#2d67e3',
        edgecolor='k',
        alpha=0.8)

    # Standard deveation for train score
    plt.errorbar(
        x=data.mean_train_score,
        y=height_train,
        xerr=data.std_train_score,
        fmt='>',
        color='k',
        alpha=0.8)

    # test score
    plt.barh(
        y=height_test,
        width=data.mean_test_score,
        height=height,
        label='test_score',
        color='#ffac59',
        edgecolor='k',
        alpha=0.8)

    # Standard deveation for test scores
    plt.errorbar(
        x=data.mean_test_score,
        y=height_test,
        xerr=data.std_test_score,
        fmt='>',
        color='k',
        alpha=0.8)
    print(f"Train: [{data.mean_train_score.min()} - {data.mean_train_score.max()}]")
    print(f"Test: [{data.mean_test_score.min()} - {data.mean_test_score.max()}]\n")

    plt.xlim(lim)
    plt.title('Scores of models with different parameters')
    plt.xlabel('Mean scores')
    plt.yticks(range(len(list_index)), labels=list_index)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.show()

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100

  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  round_fn = lambda x : np.round(x * 100, 2)
  model_results = {
      "accuracy": model_accuracy,
      "precision": round_fn(model_precision),
      "recall": round_fn(model_recall),
      "f1": round_fn(model_f1)
  }

  return model_results