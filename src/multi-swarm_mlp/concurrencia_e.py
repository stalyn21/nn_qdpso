#---------------
# import modules
#---------------

import numpy as np
from timeit import default_timer
import csv
# Datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_circles
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
# Multi-layer Perceptron
from nn.swarm_perceptron_qpso_concurrencia_e import perceptron 

def load_data(data_name):
    if data_name == "iris":
        data = load_iris()
        return data.data, data.target
    elif data_name == "wine":
        data = load_wine()
        return data.data, data.target
    elif data_name == "breast":
        data = load_breast_cancer()
        return data.data, data.target
    else:
        n = 500 #register number
        #generating the input X and output y (binary vector)
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
        y = y[:, np.newaxis]
        return X, np.array(y).reshape((n,))
    
def save_timings_to_csv(data, data_name, training_type, algorithm, tipe):
    with open(f'./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_{tipe}_time.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Duration (seconds)"])
        for row in data:
            layer, start_time, end_time = row
            duration = end_time - start_time
            writer.writerow([layer, duration])

def main(data_name, training_type, algorithm, max_iter_pass, n_particulas_pass):
    X_org, y_org = load_data(data_name)
    print('Shape of X: ', X_org.shape)
    print('Shape of y: ', y_org.shape)

    # normalize data (0,1)
    t = MinMaxScaler()
    t.fit(X_org)
    X_org = t.transform(X_org)

    # Primero dividimos los datos en entrenamiento(80%) y testeo(20%)
    X_train_aux, X_test_bal, y_train_aux, y_test_bal = train_test_split(X_org, y_org, test_size=0.2, random_state=100)
    # Luego dividimos el conjunto de entrenamiento(80%) en entrenamiento(87.5% del 80% => 70% del total) y validación(12.5% del 80% => 10% del total)
    X_train_bal, X_val_bal, y_train_bal, y_val_bal = train_test_split(X_train_aux, y_train_aux, test_size=0.125, random_state=100)    

    print(f"Training target statistics: {Counter(y_train_bal), len(y_train_bal)}")
    print(f"Testing target statistics: {Counter(y_test_bal), len(y_test_bal)}")
    print(f"Validation target statistics: {Counter(y_val_bal), len(y_val_bal)}")

    # Global variables
    X_sample = len(X_train_bal)
    X_input = len(X_train_bal[1])
    X_class = len(np.unique(y_train_bal))

    n_particulas = n_particulas_pass
    max_iter = max_iter_pass
    n_training = 15

    X_train = X_train_bal
    y_train = y_train_bal
    X_test = X_test_bal
    y_test = y_test_bal
    X_val = X_val_bal
    y_val = y_val_bal

    #---------------------------------------
    # variable initialize for qpso algorithm
    #---------------------------------------
    beta = 1.13 #1.13

    gBest_value = []
    gBest = []
    cost_test = []
    cost_val = []
    metric_train_hidden = []
    metric_train_output = []
    test_time = []
    val_time = []

    # Save information about the training
    best_gBest = None
    best_g_best_value_iter = None
    best_avg_best_value = None
    best_std_best_value = None
    best_cost_test = float('inf')
    best_cost_val = float('inf')
    best_test_val_pred = float('inf')


    # load perceptron
    nn = perceptron(X_sample, X_input, X_class)
    
    for i in range(n_training):
        if (i==0):
            start_time_training = default_timer()
            gBest.append(nn.train(X_train, y_train, X_test, y_test, beta, n_particulas, max_iter, 100, 100))
            end_time_training = default_timer()
        else:
            print(">>>>>>>>>> Parameters from previous training <<<<<<<<<<<<")
            start_time_training = default_timer()
            gBest.append(nn.train(X_train, y_train, X_test, y_test, beta, n_particulas, max_iter, hidden_error, output_error, gBest[best_test_val_pred]))
            end_time_training = default_timer()
        
        nn.timings.append((f"total_training_{i}", start_time_training, end_time_training))
        
        hidden_error = nn.hidden_error
        output_error = nn.output_error

        gBest_value.append(nn.g_best_value)
        metric_train_hidden.append(nn.g_best_value_iter_hidden)
        metric_train_output.append(nn.g_best_value_iter_output)
        start_time_test = default_timer()
        y_test_pred = np.argmax(nn.forward(X_test, gBest[i]), axis=1)
        end_time_test = default_timer()
        test_time.append((f"total_test_{i}", start_time_test, end_time_test))        
        cost_test.append(mean_squared_error(y_test, y_test_pred))
        print ('>>> Test prediction cost with training: ', cost_test[i]) 
        min_cost_test = min(cost_test)
        print (">>> Min test prediction cost: ", min_cost_test)
        start_time_val = default_timer()
        y_val_pred = np.argmax(nn.forward(X_val, gBest[i]), axis=1)
        end_time_val = default_timer()
        val_time.append((f"total_val_{i}", start_time_val, end_time_val))
        cost_val.append(mean_squared_error(y_val, y_val_pred))
        min_cost_val = min(cost_val)
        print ('>>> Validation prediction cost with training: ', cost_val[i])
        print (">>> Min validation prediction cost: ", min_cost_val)
        if  (cost_test[i] <= min_cost_test and cost_val[i] <= min_cost_val):
            best_test_val_pred = i 
            best_cost_test = cost_test[i]
            best_cost_val = cost_val[i]
            best_gBest = gBest[i]
            best_g_best_value_iter = nn.g_best_value_iter
            best_avg_best_value = nn.avg_best_value
            best_std_best_value = nn.std_best_value
    
    #save the best training model  
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_gBest_113_{n_particulas}_{max_iter}_{X_input}.npy", best_gBest)
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_gBestIter_113_{n_particulas}_{max_iter}_{X_input}.npy", best_g_best_value_iter)
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_avgBest_113_{n_particulas}_{max_iter}_{X_input}.npy", best_avg_best_value)
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_stdBest_113_{n_particulas}_{max_iter}_{X_input}.npy", best_std_best_value)
    
    time_data.extend(nn.timings)
    save_timings_to_csv(test_time, data_name, training_type, algorithm, "test")
    save_timings_to_csv(val_time, data_name, training_type, algorithm, "val") 
            
    print("=====================================================================")
    print("=====================================================================")
    print(">>> Save train metric ....")
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_metric_hidden_113_{n_particulas}_{max_iter}_{X_input}.npy", np.array(metric_train_hidden))
    np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{algorithm}_{data_name}_metric_output_113_{n_particulas}_{max_iter}_{X_input}.npy", np.array(metric_train_output))
    print(">>> The best training is in iteration: ", best_test_val_pred)
    print(">>> The global best value is: ", gBest_value[best_test_val_pred])
    print(">>> Test prediction cost: ", best_cost_test)
    print(">>> Validation prediction cost: ", best_cost_val)
    print("=====================================================================")
    print("=====================================================================")

if __name__ == '__main__':
    datasets = ["breast"] #,  "iris", "wine", "circle"
    training_type = 'training_exchange_layers_e'
    algorithm = 'qdpso'
    
    iterations_list = [1000] # 20, 40, 60, 80, 100, 500,
    particles_list = [40, 60, 80, 100] # 20, 

    for data_name in datasets:
        for max_iter in iterations_list:
            for n_particulas in particles_list:
                time_data = []
                start_time = default_timer()
                
                main(data_name, training_type, algorithm, max_iter, n_particulas)
                
                end_time = default_timer()
                execution_time = end_time - start_time
                
                time_data.append((f"{max_iter}_{n_particulas}_execution_time", start_time, end_time))
                save_timings_to_csv(time_data, data_name, training_type, algorithm, "training")
                
                np.save(f"./output/{training_type}/{data_name}/{max_iter}/{n_particulas}/{data_name}_{max_iter}_{n_particulas}_start_end_execution_time.npy", 
                        np.array([start_time, end_time, execution_time]))
                
                print(f"Execution time for {data_name} with {max_iter} iterations and {n_particulas} particles: {execution_time} seconds")
    