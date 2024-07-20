import os
import gc
import traceback
import numpy as np
from random import choices
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow import convert_to_tensor
from tensorflow.keras.backend import clear_session

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import mean_squared_error as MSE

from utils.helper_functions import load_data
from utils.helper_functions import Data_Preprocessor, LSTM_model, plot_loss_lr

class GeneticAlgorithm():
  def __init__(self,
               target_eval:float,
               epoch_range:list=[1, 101],
               look_back_range:list=[1, 61],
               size:tuple=(1, 5),
               max_iter:int=100,
               crossover_rate=0.6,
               mutation_rate=0.1,
               selection_method:str='Rank',
               model_save_to='./'):
      """
      A genetic algorithm for searching better epochs and look back of stock price data.

      Args:
        target_eval: Target value for fitness.
        epoch_max: Max value of epoch, default is 101.
        look_back_max: Max value of look_back, default is 61.
        size: Size of population, default is (1, 5).
        max_iter: A value for stop iteration.
        crossover_rate: Probability of gene crossover during reproduction.
        mutation_rate: Probability of gene mutation during reproduction.
        selection method: Methods for selection, there are three methods provided: 
                          1. Rank: Choose top 50% parents.
                          2. Roulette: Choose parents based on Roulette Wheel Selection.
                          3. Tournament: Choose parents based on Tournament Selection.
        model_save_to: File path for model saving if find the better one.
      """
      self.target_eval = target_eval
      self.epoch_min, self.epoch_max = epoch_range
      self.look_back_min, self.look_back_max = look_back_range
      self.size = size
      self.max_iter = max_iter
      self.crossover_rate = crossover_rate
      self.mutation_rate = mutation_rate
      self.selection_method = selection_method
      self.model_save_to = model_save_to
      self.bin2int = lambda *args: [int(arg, 2) for arg in args]
  
  def generate_population(self) -> list:
      """
      Generate initial population including epochs and look_backs.
       
      Returns:
          initial_chromosomes, a list with
      """ 
      epoch_candidates = np.random.randint(self.epoch_min, self.epoch_max, size=self.size)
      look_back_candidates = np.random.randint(self.look_back_min, self.look_back_max, size=self.size)
      initial_chromosomes = [[epoch, look_back] for epoch, look_back in zip(epoch_candidates[0], look_back_candidates[0])]

      return initial_chromosomes
  
  def evaluate_model(self,
                     dataset:np.ndarray, 
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
          Tuple containing model, loss value, and predictions.
      """
      try:
        # Data preprocessing
        data_preprocessor = Data_Preprocessor(dataset, test_dataset, look_back=look_back, rescaling='MinMaxScaling')
        X_train, X_valid, y_train, y_valid = data_preprocessor.preprocessing_data()
        X_test, y_test_scaled = data_preprocessor.preprocessing_data(test_data=True)
        X_train = convert_to_tensor(X_train)
        X_valid = convert_to_tensor(X_valid)
        X_test = convert_to_tensor(X_test)
        
        # Build model
        model = LSTM_model(input_shape=X_train.shape[1:])

        # Compile model
        model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))

        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.6, patience=3, min_lr=1e-6, mode='min', verbose=1)

        # EarlyStopping
        early_stopper = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

        # Fit model
        model_history = model.fit(X_train, 
                                  y_train, 
                                  epochs=epochs, 
                                  batch_size=32, 
                                  validation_data=(X_valid, y_valid), 
                                  callbacks=[reduce_lr, early_stopper])

        # Evaluate model
        model_predictions = model.predict(X_test)
        y_pred = data_preprocessor.scaler.inverse_transform(model_predictions)
        y_test = data_preprocessor.scaler.inverse_transform(y_test_scaled)
        model_loss = MSE(y_test, y_pred)

        # record model
        stopped_epoch = early_stopper.stopped_epoch
        if stopped_epoch:
            epochs = stopped_epoch

        return (model, model_loss, y_pred, y_test, epochs, model_history)
      except Exception as e:
        traceback.print_exc()
        raise ValueError(f'Error in evaluate_model: {e}')

  def selection(self,
                fitness:list,
                chromosomes:list) -> list:
      """
      Select the elites of population (chromosomes) based on fitness values.

      Args:
          fitness: Values for selection.
          chromosomes: Selected population.
      Returns:
          Selected population.
      """
      parents = []
      if self.selection_method == 'Rank':
        survive_index = int(np.ceil(len(fitness) * 0.5))
        top50_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[:survive_index]
        parents = [chromosomes[index] for index in top50_indices]
      
      elif self.selection_method == 'Roulette':
        # Sort population (chromosomes) and fitness
        fitness.sort()
        fitness_inverse = [(1/value) for value in fitness]   

        # Normalize fitness values
        total = sum(fitness_inverse)
        norm_fitness_values = [round((value/total), 2) for value in fitness_inverse]

        # Find cumulative fitness values for roulette wheel selection
        cumulative_fitness = []
        probability = 0
        for norm_value in norm_fitness_values:
            probability += norm_value
            cumulative_fitness.append(round(probability, 3))

        # Spin the wheel, choose parent population
        for _ in range(len(chromosomes)):
            random_number = np.random.uniform(0, 1)
            for individual_index, score in enumerate(cumulative_fitness):
                if (random_number <= score):
                   elite = chromosomes[individual_index]
                   if not (elite in parents):
                      parents.append(elite)
                   break
      
      elif self.selection_method == 'Tournament':
        counts = int(len(chromosomes) * 0.5)
        tournament_pool = list(zip(chromosomes, fitness))
        for _ in range(counts):
            p_1, p_2 = choices(tournament_pool, k=2)
            if p_1[1] >= p_2[1]:
               if not (p_2[0] in parents):
                  parents.append(p_2[0])
            else:
               if not (p_1[0] in parents):
                  parents.append(p_1[0])

      else:
         raise ValueError('Methods are Rank, Roulette, or Tournament!')
      
      return parents

  def crossover(self,
                candidate1:str, 
                candidate2:str) -> tuple:
      """
      Crossover the candidate1 and candidate2 "genes".
      
      Args:
          candidate1: list contains "0" or "1" present as "gene"
          candidate2: list contains "0" or "1" present as "gene"
      Returns:
          Two crossover candidates.
      """
      # Choose the position for crossover
      candidate_index_1 = np.random.randint(0, len(candidate1))
      candidate_index_2 = np.random.randint(0, len(candidate2))

      # Crossover on two candidates
      candidate_cross_1 = candidate1[:candidate_index_1] + candidate2[candidate_index_2:]
      candidate_cross_2 = candidate2[:candidate_index_2] + candidate1[candidate_index_1:]

      return (candidate_cross_1, candidate_cross_2)

  def mutate(self,
             candidate_epoch:str, 
             candidate_look_back:str) -> tuple:
      """
      Mutation on candidate genes by point mutation.

      Args:
          candidate_epoch: Candidate gene for Epoch, a hyperparameter on model.
          candidate_look_back: Candidate gene for Look Back, a hyperparameter on model.
      Returns:
          A tuple contains mutated genes.
      """
      # Randomly choose a mutation point for epoch
      epoch_mutation_index = np.random.randint(0, len(candidate_epoch))
      mutated_epoch = candidate_epoch[:epoch_mutation_index] +\
                      ('0' if candidate_epoch[epoch_mutation_index] == '1' else '1') +\
                      candidate_epoch[epoch_mutation_index+1:] 

      # Randomly choose a mutation point for look back
      look_back_mutation_index  = np.random.randint(0, len(candidate_look_back))
      mutated_look_back = candidate_look_back[:look_back_mutation_index] +\
                          ('0' if candidate_look_back[look_back_mutation_index] == '1' else '1') +\
                          candidate_look_back[look_back_mutation_index+1:] 

      return (mutated_epoch, mutated_look_back)

  def reproduction(self,
                   fitness:list, 
                   initial_chromosomes:list) -> list:
      """
      Perform "reproduction" calculations on elite groups.
      1. Select two candidates for crossover with all probabilities.
      2. Give some probabilities for gene mutation.
      
      Args:
          fitness: Values for selection.
          chromosomes: Selected population.
      Returns:
          A list after reproduction algorithm.
      """
      offspring_population = []

      # Selection
      parent_population = self.selection(fitness, initial_chromosomes)
      
      # Initial population in binary format
      elite_pop_binary = [[format(elite[0], 'b'), format(elite[1], 'b')] for elite in parent_population]

      # Take each individual to reproduce
      for i in range(len(elite_pop_binary)):
          for j in range(i+1, len(elite_pop_binary)):
            candidate_epoch_1, candidate1_look_back_1  = elite_pop_binary[i][0], elite_pop_binary[i][1]
            candidate_epoch_2, candidate1_look_back_2 = elite_pop_binary[j][0], elite_pop_binary[j][1]

            # Crossover on two genes
            random_crossover = np.random.uniform()
            if random_crossover <= self.crossover_rate:
                epoch_cross_1, epoch_cross_2 = self.crossover(candidate_epoch_1, candidate_epoch_2)
                look_back_cross_1, look_back_cross_2 = self.crossover(candidate1_look_back_1, candidate1_look_back_2)
                next_population = [[epoch_cross_1, look_back_cross_1], [epoch_cross_2, look_back_cross_2]]

                # Give some probability to mutation
                for epoch, look_back in next_population:
                    random_mutation = np.random.uniform()
                    if random_mutation <= self.mutation_rate:
                        offspring_epoch, offspring_look_back = self.mutate(epoch, look_back)
                    else:
                        offspring_epoch, offspring_look_back = epoch, look_back
                    
                    # Convert mutated binary strings back to integers
                    offspring_epoch_int, offspring_look_back_int = self.bin2int(offspring_epoch, offspring_look_back)

                    # Survivor selection
                    if (0 < offspring_epoch_int < self.epoch_max) and (0 < offspring_look_back_int < self.look_back_max):
                        offspring_population.append([offspring_epoch_int, offspring_look_back_int])

            else:
              # Convert mutated binary strings back to integers
              offspring_epoch_int_1, offspring_look_back_int_1 = self.bin2int(candidate_epoch_1, candidate1_look_back_1)
              if not ([offspring_epoch_int_1, offspring_look_back_int_1] in offspring_population):
                 offspring_population.append([offspring_epoch_int_1, offspring_look_back_int_1])
      
      # Prevent population to become small
      if len(offspring_population) < 3:
         print("New individuals adding..")
         offspring_population.extend(self.generate_population())
      elif len(offspring_population) > self.size[1]:
         offspring_population = offspring_population[:self.size[1]]

      return offspring_population

  def plot_every_fitness_value(self,
                               every_best_fitness:list):
     """
     Plot bar figure of fitness value per generation.

     Args:
        every_best_fitness: Best fitness value per generation
     """
     plt.figure(figsize=(8, 6))
     plt.xlabel("Iteration", fontsize=12)
     plt.ylabel("Fitness", fontsize=12)
     plt.title('Every Best Fitness Value of Generation', fontsize=12)
     plt.bar(range(len(every_best_fitness)), 
             every_best_fitness, 
             linewidth=2, 
             label="Best fitness convergence", 
             color='b')
     plt.legend()
     plt.show()

  def run(self, 
          dataset:list,
          test_dataset:list) -> dict:
      """
      Run the genetic algorithm:
      1. Generate initial population
      2. Evaluate LSTM model and calculate fitness
      3. Run selection of population with top 50% individuals
      4. Run reproduction with crossover and mutation.
      5. Repeat step 2 to 4 util find the better hyperparameters than target_val

      Args:
        dataset: Dataset for training.
        test_dataset: Dataset for testing.
      Returns:
        Dictionary contains information of better model.
      """
      print(f"Target value to beat: {self.target_eval}")
      # Save training results
      better_models = dict()

      # Initial population
      initial_chromosomes = self.generate_population()

      model_found = False
      generation = 1 # generation

      cur_iter = 1 # current iteration
      every_best_fitness = []
      
      # clear sessions
      clear_session() 

      while cur_iter < self.max_iter:
        print('='*40, f" Generation {str(generation)}", '='*40)
        print(f"[Epochs, Look_Back]: {initial_chromosomes}")
        print('='*95)
        fitness = []
        for idx, (epochs, look_back) in enumerate(initial_chromosomes, start=1):
          print(f"Training start: Epoch: {epochs}, Look Back: {look_back}")

          # clear sessions
          clear_session() 
          gc.collect()

          # Fitting model and calculate results
          model, model_loss, y_pred, y_true, epochs, model_history = self.evaluate_model(dataset, test_dataset, epochs, look_back)
          print(f"Iter: {cur_iter} | Loss: {model_loss} | Epochs: {epochs} | Windows: {look_back}")
          fitness.append(model_loss)

          # Stop condition
          if model_loss < self.target_eval:
            train_id = 'g' + str(generation) + '_cur' + str(idx)
            better_models['ID'] = train_id
            better_models['Epochs_Look'] = [epochs, look_back]
            better_models['Loss_Value'] = model_loss
            better_models['Predictions'] = y_pred
            better_models['True_Labels'] = y_true
            every_best_fitness.append(model_loss)
            model_found = True
            break

          del model
          cur_iter += 1
        
        # Termination
        if model_found:
          print(f"Stop iteration at generation-{str(generation)}_chromosome-{str(idx)} ||| Epochs_Look: {epochs}, {look_back} ||| Loss: {model_loss}")

          # Plot training history
          history = model_history.history
          better_models['history'] = history
          plot_loss_lr(history)
          
          # Plot best fitness value per generation
          self.plot_every_fitness_value(every_best_fitness)

          # Save model
          save_fname = "LSTM_GA_model_" + self.selection_method + "_" + train_id + '_' + datetime.now().strftime("%m%d%Y_%H%M%S") + ".h5"
          model_save_filepath = os.path.join(self.model_save_to, save_fname)
          print(f'Saving the best model at {model_save_filepath}')
          model.save(model_save_filepath)
          del model
          break

        # Record best fitness of each generation
        every_best_fitness.append(sorted(fitness)[0])

        # Reproduction
        next_generation = self.reproduction(fitness, initial_chromosomes)
        initial_chromosomes = next_generation[:]

        # Next generation
        generation += 1
      
      return better_models

if __name__ ==  '__main__':
    from tensorflow.random import set_seed as tf_set_seed
    from helper_functions import load_data
    from helper_functions import Data_Preprocessor, LSTM_model, plot_loss_lr
    
    # Set random seed
    seed = 42
    tf_set_seed(seed)

    # Load dataset
    train_path = 'C:/Users/User/Desktop/tf_hyperparameters_searching/lstm-price-preds-genetic-algorithm/NASDAQ_100_Data_From_2010.csv'
    test_path = 'C:/Users/User/Desktop/tf_hyperparameters_searching/lstm-price-preds-genetic-algorithm/AAPL_test_data.csv'
    dataset = load_data(train_path, 'AAPL')
    test_dates, test_dataset = load_data(test_path, stock_code=None, test_data=True)

    # Run genetic algorithm
    genetic_algorithm = GeneticAlgorithm(target_eval=1., 
                                        epoch_range=[1, 101],
                                        look_back_range=[10, 61],
                                        size=(1, 10),
                                        max_iter=100, 
                                        crossover_rate=0.8,
                                        mutation_rate=0.1,
                                        selection_method='Rank',
                                        model_save_to='./',)
    final_results = genetic_algorithm.run(dataset=dataset, test_dataset=test_dataset)