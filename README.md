<h1 id='main-title'> Deep Learning Hyperparameter Tuning Projects </h1>

This github includes several projects which I interested in such as hyperparameters grid searching by `GridSearchCV`,  Genetic Algorithm (GA), or others.

| Project completed | Techniques | Introduction | Codes |
| :---------------: | :----------: | :--------: | :--------: |
| [CNN-keras-mnist-gridsearch](#project1) | Computer Vision, Classification, GridSearchCV, KerasClassifier | Use GridSearchCV with KerasClassifier to search best hyperparameters in simple CNN model trained on MNIST database. | [Link](https://github.com/Leohoji/deep-learning-for-hyperparameters-searching/blob/main/CNN-keras-mnist-gridsearch.ipynb) |
| [price-pred-using-LSTM-applying-genetic-algorithm](#project2) | Time-Series analysis, Prediction, Genetic Algorithm | Use genetic algorithm (GA) technique to search better hyperparameters in simple Long Short-Term Memory (LSTM) model trained on APPL stock database from [kaggle dataset](https://www.kaggle.com/datasets/kalilurrahman/nasdaq100-stock-price-data/data?authuser=0) | ---- |

<h2>Prerequisites</h2>

- python 3.8

- tensorflow 2.5.0 【 If you have GPU to accelerate, you could install tensorflow-gpu 2.5.0 with CUDA 11.2 and cuDNN 8.1 】

- Others in `requirements.txt`

💻 The GPU I use is **GeForce GTX 1050 Ti**


<h2 id='project1'>MNIST classification using CNN applying gridsearch</h2>

**Notebooks Here**  --> [CNN-keras-mnist-gridsearch.ipynb](https://github.com/Leohoji/machine-learning-deep-learning-projects/blob/main/cnn-keras-mnist-gridsearch/CNN-keras-mnist-gridsearch.ipynb)

Use GridSearchCV with KerasClassifier to search best hyperparameters in simple CNN model trained on MNIST database.

<h3>How Do I Complete This Project</h3>

#### Summary of experimental process
<p align='left'>
  <img alt="process of project" src="https://github.com/Leohoji/projects_for_hyperparameters_searching/blob/main/introduction_images/process_for_cnn_gridsearch.png?raw=true" width=700 height=400>
</p>

#### Hyperparameters to be searched
| Hyperparameters | Values to search |
| -- | -- |
| Activation functions | Sigmoid, ReLU, LeakyReLU, PReLU, tanh |
| Loss functions | MSE, Cross-Entropy |
| Batch size | 8, 32, 128 |

Epoch is fixed as **5** and Optimizer as **Adam** for convenience of experimentation.

#### Process of experiment
1. Create helper functions and import necessary libraries.
2. Load MNIST database and preprocess it, including create training and testing data.
3. Use KerasClassifier wrapper to pacckage CNN model.
4. Pass the wrapped model and hyperparameters expected to searched into GridSearchCV API.
5. Analize the results and conclude the experiment.

#### Results

Finally find some conditions that can make the model peform better, following conditions are the hyperparameters options that make the accuracy better than 90% within this project.

| Hyperparameters      | Better Performance   |
| -------------------- | -------------------- |
| Batch Size           | 8, 32                |
| Activation Functions | ReLU or Tanh         |
| Loss functions       | MSE or Cross-Entropy |

> [Back to outlines](#main-title)


<h2 id='project2'>Stock price prediction using LSTM applying genetic algorithm</h2>

**Notebooks Here**  --> 

Use genetic algorithm (GA) technique to search better hyperparameters in simple Long Short-Term Memory (LSTM) model trained on APPL stock database from [kaggle dataset](https://www.kaggle.com/datasets/kalilurrahman/nasdaq100-stock-price-data/data?authuser=0).

<h3>How Do I Complete This Project</h3>

#### Summary of experimental process
<p align='left'>
  <img alt="process of project" src="https://github.com/Leohoji/deep-learning-hyperparameter-tuning-projects/blob/main/introduction_images/process_of_GA_experiment.png?raw=true" width=700 height=400>
</p>

In this project I will follow the following steps to complete my project: 

<h3><ins>Problem definition</ins></h3>

This project is to conquer a prediction problem of time-series data. I would like to apply a evolution algorithms named Genetic Algorithm, short for GA, on LSTM model, a recursive neural network, to find a set of hyperparameters to let the predictions of stock by model be close to real price.

<h3><ins>Data and Features</ins></h3>

The dataset used in this project is a time-series data from from kaggle dataset: [NASDAQ-100 Stock Price Dataset](https://www.kaggle.com/datasets/kalilurrahman/nasdaq100-stock-price-data/data?authuser=0), this dataset contains stock prices of all NASDAQ-100 index stocks (as on Sep 2021) from 2010, I only choose AAPL index to predict.

Testing data: 2012/01 - 2012/06 APPL historiccal stocck price data.

A **structured data** with **271680** data points, all th columns are numerical.

| Columns   | Description                        |
| :-------- | :--------------------------------- |
| Open      | Opening Price on that day          |
| High      | High Price on that day             |
| Low       | Low Price on that day              |
| Close     | Closing Price on that day          |
| Adj Close | Adjusted Closing Price on that day |
| Volume    | Volume Traded on that day          |
| Name      | Stock Name (contains 100 stocks)   |

<p align='left'>
  <img alt="feature-and-labels" src="https://github.com/Leohoji/machine-learning-deep-learning-projects/blob/main/introduction_images/features-for-time-series.png?raw=true" width=800 height=200>
</p>

At present, most people's guess is that the stock price will be related to the previous day or previous data, hence, the previous data is feature, the current day is label. 

<h3><ins>Evaluation</ins></h3>

**Mean Square Error (MSE)** is a common metric for prediction problem, I use it to be the model's loss function and GA's fitness function.

$\text{MSE: } \frac{1}{N}\Sigma_{i=1}^{N}(y_{i} - {\hat{y}})^{2}$

<h3><ins>Modeling</ins></h3>

```python
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
```

<h3><ins>Experiments</ins></h3>

**Data Preprocessing**

1. Read and check the data, I will visualize the AAPL stock price.

2. Rescale the data into range [0, 1] by min-max rescaling using scikit-learn library [API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

3. Create feature and label from data, as present in **Data and Features**, the feature of the time series data is previous `N` day timestamp and the label is `N+1` day timestamp, that is to day, the task of this step is to convert a time series data into a `supervised learning`.

4. Split the dataset into `X` and `y`. Moreover, create the training and testing data ( `X_train`, `X_test`, `y_train`, `y_test` ).

**Experiments**

1. Train a baseline model to be the metric of genetic algorithm.

ExperimentaL conditions:

| Hyperparameters     | Values        |
| :-----------------: | :-----------: |
| Rescaling method    | MinMaxScaling |
| Loss function       | mse           |
| Optimizer           | Adam          |
| Learning rates      | 1e-4          |
| Batch size          | 32            |
| Epochs              | 30            |
| Look backs          | 10            |

I also use [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) API for learning rate reduction gradually during training.

2. Apply genetic algoruthm (GA) to find better `epoch` and `look_back` hyperparameters.

The detail introduction of GA is written on my medium (Traditional Chinese): [基因演算法 (Genetic Algorithm, GA) 介紹](https://medium.com/@leo122196/%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E4%BB%8B%E7%B4%B9-62df31c0e670)

ExperimentaL conditions:

| Hyperparameters     | Values        |
| :-----------------: | :-----------: |
| Rescaling method    | MinMaxScaling |
| Loss function       | mse           |
| Optimizer           | Adam          |
| Learning rates      | 1e-4          |
| Batch size          | 32            |
| Epochs              | `??`          |
| Look backs          | `??`          |

Following is the condition I used in GA:

**Population Initialization**: 10 `[epochs, look_back]` group to combine a population with randomly generation.

**Fitness Function**: **Mean Square Error (MSE)**, as well as the loss function used in LSTM model for evaluation.

**Selection**: 

(1) Rank Selection: Choose the top 50% to be offsprings .

(2) Roulette Wheel Selection: Assign the cumulative probability for each fitness individual based on the value.

(3) Tournament Selection: Randomly choose 2 individual to compete.

**Crossover**: The range of crossover rate is [0.5, 1.].

**Mutation**: The mutation rate is fixed to 0.1

**Survivor Selection**:

(1) If the value of offspring is zero or bigger than max value, it will be rejected to the next generation.

(2) If the length of the offspring is less than three, new population will be generated and join it.

**Termination Condition**

(1) Max iteration is set as 100.

(2) Fitness value is better than baseline model.

#### Results

All of selection methods can drive the GA to evolve the better hyperparameter searching.

> [Back to outlines](#main-title)
