<h1> Deep learning for hyperparameters searching </h1>

This github includes several projects which I interested in such as hyperparameters grid searching by `GridSearchCV`,  Genetic Algorithm (GA), or others.

| Project completed | Techniques | Introduction |
| :---------------: | :----------: | :--------: |
| ☑ [CNN-keras-mnist-gridsearch](#project1) | Tensorflow, GridSearchCV, KerasClassifier | Use GridSearchCV with KerasClassifier to search best hyperparameters in simple CNN model trained on MNIST database. |

<h2>Prerequisites</h2>

- python 3.8

- tensorflow 2.5.0 【 If you have GPU to accelerate, you could install tensorflow-gpu 2.5.0 with CUDA 11.2 and cuDNN 8.1. 】

- Others in `requirements.txt`

💻 The GPU I use is **GeForce GTX 1050 Ti**

<h2 id='project1'>CNN-keras-mnist-gridsearch</h2>

Use GridSearchCV with KerasClassifier to search best hyperparameters in simple CNN model trained on MNIST database.

<h3>How Do I Complete This Project</h3>

#### Summary of process
<p align='left'>
  <img alt="process of project" src="https://github.com/Leohoji/projects_for_hyperparameters_searching/blob/main/introduction_images/process_for_cnn_gridsearch.png?raw=true" width=500 height=300>
</p>

#### Hyperparameters to be searched
| Hyperparameters | Values to search |
| -- | -- |
| Activation functions | Sigmoid, ReLU, LeakyReLU, PReLU, tanh |
| Loss functions | MSE, Cross-Entropy |
| Batch size | 8, 32, 128 |
| Epochs | **5** |
| Optimizer | **Adam** |

#### Process of experiment
1. Create helper functions and import necessary libraries.
2. Load MNIST database and preprocess it, including create training and testing data.
3. Use KerasClassifier wrapper to pacckage CNN model.
4. Pass the wrapped model and hyperparameters expected to searched into GridSearchCV API.
5. Analize the results and conclude the experiment.
