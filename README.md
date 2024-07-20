<h1 id='main-title'> Stock Price Prediction Using LSTM Applying Genetic Algorithm </h1>

Use genetic algorithm (GA) technique to search better hyperparameters in simple Long Short-Term Memory (LSTM) model trained on APPL stock database from [kaggle dataset](https://www.kaggle.com/datasets/kalilurrahman/nasdaq100-stock-price-data/data?authuser=0).

**Notebooks Here**  --> [price-prediction-using-LSTM-applying-genetic-algorithm.ipynb](https://github.com/Leohoji/deep-learning-hyperparameter-tuning-projects/blob/main/lstm-price-preds-genetic-algorithm/price-prediction-using-LSTM-applying-genetic-algorithm.ipynb)

<h2>Prerequisites</h2>

- python 3.8

- tensorflow 2.5.0+ 【 If you have GPU to accelerate, you could install tensorflow-gpu 2.5.0 with CUDA 11.2 and cuDNN 8.1 】

- Others in `requirements.txt`

💻 The GPU I use is **GeForce GTX 1050 Ti**

<h2>How Do I Complete This Project</h2>

I will introduce the details of experiments in this project.

### Summary of experimental process
<p align='left'>
  <img alt="process of project" src="https://github.com/Leohoji/deep-learning-hyperparameter-tuning-projects/blob/main/introduction_images/process_of_GA_experiment.png?raw=true" width=700 height=400>
</p>

First, the LSTM model will fit the processed data, and calculate the exvaluation value, determine whether the value is met the temination metrics. If the temination metrics are triggered, the algorithm will shut down and return the best hyperparameters, on the other hand, the algorithm will continue to next generation if the termination metrics are not triggered, before going to next generation the chromosomes (hyperparameters) will pass to the GA operator including Selection, Crossover, and Mutation for generating offsprings, and iterate this process continuously until the metrics are met. For more detail information about genetic algorithm about my project please visit my **mediun blog**: [我如何利用基因演算法 (Genetic Algorithm, GA) 調整神經網路超參數](https://medium.com/@leo122196/%E6%88%91%E5%A6%82%E4%BD%95%E5%88%A9%E7%94%A8%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E8%AA%BF%E6%95%B4%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%E8%B6%85%E5%8F%83%E6%95%B8-470f64d6cf71)

In this project I will follow the following steps to complete it: 

<h3>Problem definition</h3>

First, I have to define my problem expected to be solved in this project:

The process of training a deep learning model is mainly a optimization problem, that it to say, during the training process, I would like to find a set of hyperparameters to let the predictions of stock by model be close to real price. However, there are many ways to search the best hyperparameters, the common one is using `GridSearchCV` from **sci-kit learn** library, I think whether I can apply a evolution algorithms such as Genetic Algorithm, named as GA, on model to figure out the problem?

<h3>Data</h3>

Second, I will check what data I have to deal with, and what features in data:

The dataset is a time-series data as well as a **structured data** with **271680** data points in `csv` format, all columns are numerical:

- **Training data**: [NASDAQ-100 Stock Price](https://www.kaggle.com/datasets/kalilurrahman/nasdaq100-stock-price-data/data?authuser=0) dataset contains stock prices of all NASDAQ-100 index stocks (as on Sep 2021) from 2010, I only choose AAPL index to predict.

- **Testing data**: 2012/01 - 2012/06 APPL historiccal stocck price data.

<h3>Features</h3>

| Columns   | Description                        |
| :-------- | :--------------------------------- |
| Open      | Opening Price on that day          |
| High      | High Price on that day             |
| Low       | Low Price on that day              |
| Close     | Closing Price on that day          |
| Adj Close | Adjusted Closing Price on that day |
| Volume    | Volume Traded on that day          |
| Name      | Stock Name (contains 100 stocks)   |

**The `N` days are features, the `N+1` day is lable**

<p align='left'>
  <img alt="feature-and-labels" src="https://github.com/Leohoji/machine-learning-deep-learning-projects/blob/main/introduction_images/features-for-time-series.png?raw=true" width=800 height=200>
</p>

At present, most people's guess is that the stock price will be related to the previous day or previous data which is named `autocoorelation`, hence, the previous data is feature, the current day is label. 

<h3>Evaluation</h3>

Third, I have to define an evaluation for stop metrics, either for GA or mdoel.

**Mean Square Error (MSE)** is a common metric for prediction problem, I use it to be the model's loss function and GA's fitness function.

$\text{MSE: } \frac{1}{N}\Sigma_{i=1}^{N}(y_{i} - {\hat{y}})^{2}$

<h3>Modeling</h3>

Fourth, what model do I use, is it suitable?

<p align='left'>
  <img alt="LSTM" src="https://www.mdpi.com/water/water-11-01387/article_deploy/html/images/water-11-01387-g004.png" width=550 height=461>
</p>

*The structure of the Long Short-Term Memory (LSTM) neural network. [(source)](https://blog.mlreview.com/understanding-lstm-and-its-diagrams-37e2f46f1714)*

I would like to conquer a prediction problem of time-series data, and the GA will be applied on LSTM model, a type of recursive neural network (RNN), to find a set of hyperparameters to let the predictions close to real price as possible as I can. However, LSTM neural network has the ability to learn time series related data, which requires long-term dependence and memory ability on information.

LSTMs are a special kind of RNN, capable of learning long-term dependencies and remembering information for prolonged periods of time as a default [(source)](https://www.mdpi.com/2073-4441/11/7/1387).

The following codes are the architecture of LSTM applied in project:

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

<h3>Experiments</h3>

Fifth, I have to design the process of experiments and recycle it until the performance is improved.

**Data Preprocessing**

1. Read and check the data, I will visualize the AAPL stock price.

2. Rescale the data into range [0, 1] by min-max rescaling using scikit-learn library [API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

3. Create feature and label from data, as present in **Data and Features**, the feature of the time series data is previous `N` day timestamp and the label is `N+1` day timestamp, that is to day, the task of this step is to convert a time series data into a `supervised learning`.

4. Split the dataset into `X` and `y`. Moreover, create the training and testing data ( `X_train`, `X_test`, `y_train`, `y_test` ).

**Experiments**

1. Train a baseline model to be the metric of genetic algorithm.

2. Apply genetic algorithm (GA) to find better `epoch` and `look_back` hyperparameters.

The detail introduction of GA is written on my medium (Traditional Chinese): [基因演算法 (Genetic Algorithm, GA) 介紹](https://medium.com/@leo122196/%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E4%BB%8B%E7%B4%B9-62df31c0e670)

ExperimentaL conditions for `baseline model`:

| Hyperparameters     | Values        |
| :-----------------: | :-----------: |
| Rescaling method    | MinMaxScaling |
| Loss function       | mse           |
| Optimizer           | Adam          |
| Learning rates      | 1e-4          |
| Batch size          | 32            |
| Epochs              | 30            |
| Look backs          | 10            |

ExperimentaL conditions for `GA searching`:

| Hyperparameters     | Values        |
| :-----------------: | :-----------: |
| Rescaling method    | MinMaxScaling |
| Loss function       | mse           |
| Optimizer           | Adam          |
| Learning rates      | 1e-4          |
| Batch size          | 32            |
| Epochs              | `??`          |
| Look backs          | `??`          |

I also use:
- [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) API for learning rate reduction gradually during training.
- [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) API to stop training when a monitored metric has stopped improving.

Following is the condition I used in GA:

| **Component**                 | **Description**                                                                                         |
| :---------------------------- | :------------------------------------------------------------------------------------------------------ |
| **Population Initialization** | 10 `[epochs, look_back]` groups to combine a population with randomly generation.                       |
| **Fitness Function**          | **Mean Square Error (MSE)**, as well as the loss function used in LSTM model for evaluation.            |
| **Selection**                 |                                                                                                         |
|   - Rank Selection            | Choose the top 50% to be offsprings.                                                                    |
|   - Roulette Wheel Selection  | Assign the cumulative probability for each fitness individual based on the value.                       |
|   - Tournament Selection      | Randomly choose 2 individual to compete.                                                                |
| **Crossover**                 | The range of crossover rate is [0.5, 1.].                                                               |
| **Mutation**                  | The mutation rate is fixed to 0.1.                                                                      |
| **Survivor Selection**        |                                                                                                         |
|   - Rejection Criteria        | If the value of offspring is zero or bigger than max value, it will be rejected to the next generation. |
|   - Population Regeneration   | If the length of the offspring is less than three, new population will be generated and join it.        |
| **Termination Condition**     |                                                                                                         |
|   - Max Iteration             | Max iteration is set as 100.                                                                            |
|   - Fitness Improvement       | Fitness value is better than baseline model.                                                            |

## Results

All of selection methods can drive the GA to evolve the better hyperparameter searching.

The evaluation value of baseline model is `0.683`, and it improves to `0.166`, `0.169`, and `0.15` by using GA with Rank Selection, Roulette Wheel Selection, and Tournament Selection respectively.

| Model                    | MSE value | Epochs | Windows (Look Back) |
| -----------------------  | --------- | ------ | ------------------- |
| Baseline                 | 0.683     | 28     | 10                  |
| Rank Selection           | 0.166     | 39     | 1                   |
| Roulette Wheel Selection | 0.169     | 31     | 1                   |
| Tournament Selection     | 0.15      | 35     | 1                   |

> [Back to outlines](#main-title)
