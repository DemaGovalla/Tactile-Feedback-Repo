<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the RFMN algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different RFMN parameters and returns the classification
            report.  
"""


import numpy as np, pandas as pd, os, multiprocessing, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from RFMN import ReflexFuzzyNeuroNetwork
from time import time
import warnings
import matplotlib, shutil
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# np.set_printoptions(threshold=np.inf)

def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report_def = classification_report(y_test, y_pred)
    return mae, mse, rmse, r2, accuracy, precision, recall, f1, conf_matrix, class_report_def

def plot_accuracy_bar_graph(values_array, theta_labels, plot_title, save_filename, yaxis_label):
    directory_path_result = 'RFMN_result_def'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.06
    index = np.arange(num_gammas)
    colors = plt.cm.get_cmap('tab10', num_thetas)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas):
        label = f'Theta = {theta_labels[i]}' if isinstance(theta_labels[i], (int, float)) else f'Theta = {theta_labels[i]}'
        bars = plt.bar(index + i * bar_width, values_array[i], width=bar_width, label=label, color=colors(i))
        for bar, acc in zip(bars, values_array[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Gamma')
    plt.ylabel(yaxis_label)
    plt.title(plot_title)
    plt.xticks(index + (bar_width * (num_thetas - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.savefig(directory_path_result+ '/' + save_filename)  # Save as PNG file


def average_of_n_values(arr):
    if not arr:
        return "Input array is empty. Please provide values."
    return (sum(arr) / len(arr))*100

def convert_seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    result = f"{minutes} minute{'s' if minutes != 1 else ''}"
    if remaining_seconds > 0:
        result += f" and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    return result

def testAlgo(X_test, y_test, nn, y_predlr): 
    y_predlr1 = nn.test(X_test, y_test)
    y_predlr.send(y_predlr1)
 
def main():
    
    ''' Initiat loop counter '''
    startTime4 = time() 
    theta_loop = 0
    gamma_loop = 0
    fold_no_loop = 0
    
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData_def.csv')
    sensor_data = sensor_data.iloc[:,0:]

    # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]


    directory_path_param = 'cross_val_RFMN_param_def'

    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
        
    else:
        os.makedirs(directory_path_param)
        

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=42)

    fold_no_def = 1
    for train_index_def, val_index_def in skf.split(X_def, y_def):
        train_def = X_def.loc[train_index_def,:]
        val_def = X_def.loc[val_index_def,:]
        train_def.to_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no_def) + '.csv'))
        val_def.to_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no_def) + '.csv'))
        fold_no_def += 1

    count = 0
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

    theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    
    '''
    k-fold parameters for deformation
    '''
    k_fold_mae_def = []
    k_fold_mse_def = []
    k_fold_rmse_def = []
    k_fold_r2_def = []
    
    k_fold_accuracy_def = []
    k_fold_precision_def = []
    k_fold_recall_def = []
    k_fold_f1_def = []
    
    '''
    Metric value arrays for deformation
    '''
    mae_values_array_def = []
    mse_values_array_def = []
    rmse_values_array_def = []
    r2_values_array_def = []
    
    accuracy_values_array_def = []
    precision_values_array_def = []
    recall_values_array_def = []
    f1_values_array_def = []
    

    for theta in theta_values:
        
        '''
        Metric value for deformation
        '''
        mae_values_def = []
        mse_values_def = []
        rmse_values_def = []
        r2_values_def = []
        
        accuracy_values_def = []
        precision_values_def = []
        recall_values_def = []
        f1_values_def = []
        
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
            '''
            Metric ave for deformation
            '''
            mae_ave_def = []
            mse_ave_def = []
            rmse_ave_def = []
            r2_ave_def = []
            
            accuracy_ave_def = []
            precision_ave_def = []
            recall_ave_def = []
            f1_ave_def = []
        
            
            for fold_no in range(1,n_splits+1):
                
                startTime2 = time() # Add a timer. 
                scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
                
                '''
                Train the deformation data
                '''
                newtrain_def = pd.read_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no) + '.csv'))
                
                newtrain_def = newtrain_def.iloc[:,1:]
                new_X_train_def = newtrain_def.iloc[:,:-1].values
                new_y_train_def = newtrain_def.iloc[:,-1].values # Assign the label to the last column
                
                new_X_train_def = scaler_min_max.fit_transform(new_X_train_def)
                new_X_train_def = new_X_train_def.T

                nn_train_def = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta*.1)
                nn_train_def.train(new_X_train_def, new_y_train_def)
                print("Training is Done")
          
    
                '''
                Start the multiprocessing process for testing deformation.
                '''
                '''
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                newval_def = newval_def.iloc[:,1:]
          
                X_def_newval = newval_def.iloc[:, :4].values
                y_def_newval = newval_def.iloc[:, newval_def.shape[1]-1].values
                
                new_X_val_def = scaler_min_max.fit_transform(X_def_newval)
    
                X_train_def, X_test_def, y_train_def, y_test_def_newval = train_test_split(new_X_val_def, y_def_newval, train_size=1, random_state=42)

                num_parts_def = 94
                # num_parts_def = 2634
                # num_parts_def = 1580
                split_size_def = len(X_test_def) // num_parts_def
                x_split_def = []
                y_split_def = []
                for i in range(num_parts_def):
                    start_index_def = i * split_size_def
                    end_index_def = (i + 1) * split_size_def
                    x_part_def = X_test_def[start_index_def:end_index_def]
                    y_part_def = y_test_def_newval[start_index_def:end_index_def]
                    x_split_def.append(x_part_def)
                    y_split_def.append(y_part_def)
                x_split_def = np.array(x_split_def)
                y_split_def = np.array(y_split_def)

                transposed_arrays_def = []
                for i in range(num_parts_def):
                    transposed_arrays_def.append(x_split_def[i].T)
                transposed_arrays_def = np.array(transposed_arrays_def)


                '''
                Get the accuracy for defromation and texture
                '''
                end_range = 94
                
                y_test_def = np.array([])
                jobs_def = []
                pipe_list_def = []
    

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_def, send_end_def = multiprocessing.Pipe(False)
                    p_def = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_def[i], y_split_def[i], nn_train_def, send_end_def))
                
                    jobs_def.append(p_def)
                    pipe_list_def.append(recv_end_def)
                    p_def.start()

                    
                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_def = y_split_def[i]  
                    y_test_def = np.append(y_test_def, new_elements_def)
                y_test_def = y_test_def.reshape((end_range, -1))

    
                # Join all cores results into one
                for proc in jobs_def:
                    proc.join()

    
                '''
                Metrics For Deformation
                '''
                # Putting all cores results into an array    
                result_list_def = np.array([x.recv().tolist() for x in pipe_list_def])
                y_pred_flat_def = result_list_def.flatten()
                y_test_flat_def = y_test_def.flatten()

                mae_def, mse_def, rmse_def, r2_def, accuracy_def, precision_def, recall_def, f1_def, conf_matrix_def, class_report_def = calculate_metrics(y_test_flat_def, y_pred_flat_def)
                print("mae:", mae_def*100)
                print("mse:", mse_def*100)
                print("rmse:", rmse_def*100)
                print("r2:", r2_def*100)
                
                print("Accuracy:", accuracy_def*100)
                print("Precision:", precision_def*100)
                print("Recall:", recall_def*100)
                print("F1 Score:", f1_def*100)
                print("Confusion Matrix:\n", conf_matrix_def)
                print("Classification report:\n", class_report_def)
                
                mae_ave_def.append(mae_def)
                mse_ave_def.append(mse_def)
                rmse_ave_def.append(rmse_def)
                r2_ave_def.append(r2_def)
                
                accuracy_ave_def.append(accuracy_def)
                precision_ave_def.append(precision_def)
                recall_ave_def.append(recall_def)
                f1_ave_def.append(f1_def)
                        
                
                ''' Iterate fold_no_loop'''
                fold_no_loop += 1
                print("fold_no loop is at fold_no_loop = ", fold_no_loop)
                
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            '''
            Combine K-fold for deformation
            '''
            k_fold_mae_def.append(mae_ave_def)
            k_fold_mse_def.append(mse_ave_def)
            k_fold_rmse_def.append(rmse_ave_def)
            k_fold_r2_def.append(r2_ave_def)
            
            k_fold_accuracy_def.append(accuracy_ave_def)
            k_fold_precision_def.append(precision_ave_def)
            k_fold_recall_def.append(recall_ave_def)
            k_fold_f1_def.append(f1_ave_def)
            
    
            '''
            Combine values for deformation
            '''
            mae_values_def.append(average_of_n_values(mae_ave_def)) 
            mse_values_def.append(average_of_n_values(mse_ave_def)) 
            rmse_values_def.append(average_of_n_values(rmse_ave_def)) 
            r2_values_def.append(average_of_n_values(r2_ave_def)) 
            
            accuracy_values_def.append(average_of_n_values(accuracy_ave_def))  
            precision_values_def.append(average_of_n_values(precision_ave_def)) 
            recall_values_def.append(average_of_n_values(recall_ave_def)) 
            f1_values_def.append(average_of_n_values(f1_ave_def)) 
  
   
            ''' Iterate gamma_loop'''
            gamma_loop += 1
            print("gamma loop is at gamma_loop = ", gamma_loop)
                
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

        '''
        Combine values for deformation
        '''
        mae_values_array_def.append(mae_values_def)
        mse_values_array_def.append(mse_values_def)
        rmse_values_array_def.append(rmse_values_def)
        r2_values_array_def.append(r2_values_def)
        
        accuracy_values_array_def.append(accuracy_values_def)
        precision_values_array_def.append(precision_values_def)
        recall_values_array_def.append(recall_values_def)
        f1_values_array_def.append(f1_values_def)
        
        
        
        ''' Iterate theta_loop'''
        theta_loop += 1
        print("theta loop is at theta_loop = ", theta_loop)
               
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)

    '''
    Print K-fold for deformation
    '''
    # print("These are the def mae for each k-fold split", k_fold_mae_def, "\n")
    # print("These are the def mse for each k-fold split", k_fold_mse_def, "\n")
    # print("These are the def rmse for each k-fold split", k_fold_rmse_def, "\n")
    # print("These are the def r2 for each k-fold split", k_fold_r2_def, "\n")
    
    # print("These are the def accuracy for each k-fold split", k_fold_accuracy_def, "\n")
    # print("These are the def precision for each k-fold split", k_fold_precision_def, "\n")
    # print("These are the def recall for each k-fold split", k_fold_recall_def, "\n")
    # print("These are the def f1 for each k-fold split", k_fold_f1_def, "\n")

    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    '''
    Plot for deformation
    '''

    plot_accuracy_bar_graph(mae_values_array_def, theta_values, "mae_deformation_plot", "mae_deformation_plot", "MAE")
    plot_accuracy_bar_graph(mse_values_array_def, theta_values, "mse_deformation_plot", "mse_deformation_plot", "MSE")
    plot_accuracy_bar_graph(rmse_values_array_def, theta_values, "rmse_deformation_plot", "rmse_deformation_plot", "RMSE")
    plot_accuracy_bar_graph(r2_values_array_def, theta_values, "r2_deformation_plot", "r2_deformation_plot", "R2")
    
    plot_accuracy_bar_graph(accuracy_values_array_def, theta_values, "accuracy_deformation_plot", "accuracy_deformation_plot", "ACCURACY")
    plot_accuracy_bar_graph(precision_values_array_def, theta_values, "precision_deformation_plot", "precision_deformation_plot", "PRECISION")
    plot_accuracy_bar_graph(recall_values_array_def, theta_values, "recall_deformation_plot", "recall_deformation_plot", "RECALL")
    plot_accuracy_bar_graph(f1_values_array_def, theta_values, "f1_deformation_plot", "f1_deformation_plot", "F1")
    
    
 
    print("mae_values_array_def", mae_values_array_def, "\n")
    print("mse_values_array_def", mse_values_array_def, "\n")
    print("rmse_values_array_def", rmse_values_array_def, "\n")
    print("r2_values_array_def", r2_values_array_def, "\n")
    
    print("accuracy_values_array_def", accuracy_values_array_def, "\n")
    print("precision_values_array_def", precision_values_array_def, "\n")
    print("recall_values_array_def", recall_values_array_def, "\n")
    print("f1_values_array_def", f1_values_array_def, "\n")
    
    plt.show()
    
    
if __name__ == '__main__':
    main()
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_def.csv file using the RFMN algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different RFMN parameters and returns the classification
            report.  
"""


import numpy as np, pandas as pd, os, multiprocessing, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from RFMN import ReflexFuzzyNeuroNetwork
from time import time
import warnings
import matplotlib, shutil
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# np.set_printoptions(threshold=np.inf)

def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report_def = classification_report(y_test, y_pred)
    return mae, mse, rmse, r2, accuracy, precision, recall, f1, conf_matrix, class_report_def

def plot_accuracy_bar_graph(values_array, theta_labels, plot_title, save_filename, yaxis_label):
    directory_path_result = 'RFMN_result_def'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.06
    index = np.arange(num_gammas)
    colors = plt.cm.get_cmap('tab10', num_thetas)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas):
        label = f'Theta = {theta_labels[i]}' if isinstance(theta_labels[i], (int, float)) else f'Theta = {theta_labels[i]}'
        bars = plt.bar(index + i * bar_width, values_array[i], width=bar_width, label=label, color=colors(i))
        for bar, acc in zip(bars, values_array[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Gamma')
    plt.ylabel(yaxis_label)
    plt.title(plot_title)
    plt.xticks(index + (bar_width * (num_thetas - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.savefig(directory_path_result+ '/' + save_filename)  # Save as PNG file


def average_of_n_values(arr):
    if not arr:
        return "Input array is empty. Please provide values."
    return (sum(arr) / len(arr))*100

def convert_seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    result = f"{minutes} minute{'s' if minutes != 1 else ''}"
    if remaining_seconds > 0:
        result += f" and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    return result

def testAlgo(X_test, y_test, nn, y_predlr): 
    y_predlr1 = nn.test(X_test, y_test)
    y_predlr.send(y_predlr1)
 
def main():
    
    ''' Initiat loop counter '''
    startTime4 = time() 
    theta_loop = 0
    gamma_loop = 0
    fold_no_loop = 0
    
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData_def.csv')
    sensor_data = sensor_data.iloc[:,0:]

    # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]


    directory_path_param = 'cross_val_RFMN_param_def'

    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
        
    else:
        os.makedirs(directory_path_param)
        

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=42)

    fold_no_def = 1
    for train_index_def, val_index_def in skf.split(X_def, y_def):
        train_def = X_def.loc[train_index_def,:]
        val_def = X_def.loc[val_index_def,:]
        train_def.to_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no_def) + '.csv'))
        val_def.to_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no_def) + '.csv'))
        fold_no_def += 1

    count = 0
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

    theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    
    '''
    k-fold parameters for deformation
    '''
    k_fold_mae_def = []
    k_fold_mse_def = []
    k_fold_rmse_def = []
    k_fold_r2_def = []
    
    k_fold_accuracy_def = []
    k_fold_precision_def = []
    k_fold_recall_def = []
    k_fold_f1_def = []
    
    '''
    Metric value arrays for deformation
    '''
    mae_values_array_def = []
    mse_values_array_def = []
    rmse_values_array_def = []
    r2_values_array_def = []
    
    accuracy_values_array_def = []
    precision_values_array_def = []
    recall_values_array_def = []
    f1_values_array_def = []
    

    for theta in theta_values:
        
        '''
        Metric value for deformation
        '''
        mae_values_def = []
        mse_values_def = []
        rmse_values_def = []
        r2_values_def = []
        
        accuracy_values_def = []
        precision_values_def = []
        recall_values_def = []
        f1_values_def = []
        
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
            '''
            Metric ave for deformation
            '''
            mae_ave_def = []
            mse_ave_def = []
            rmse_ave_def = []
            r2_ave_def = []
            
            accuracy_ave_def = []
            precision_ave_def = []
            recall_ave_def = []
            f1_ave_def = []
        
            
            for fold_no in range(1,n_splits+1):
                
                startTime2 = time() # Add a timer. 
                scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
                
                '''
                Train the deformation data
                '''
                newtrain_def = pd.read_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no) + '.csv'))
                
                newtrain_def = newtrain_def.iloc[:,1:]
                new_X_train_def = newtrain_def.iloc[:,:-1].values
                new_y_train_def = newtrain_def.iloc[:,-1].values # Assign the label to the last column
                
                new_X_train_def = scaler_min_max.fit_transform(new_X_train_def)
                new_X_train_def = new_X_train_def.T

                nn_train_def = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta*.1)
                nn_train_def.train(new_X_train_def, new_y_train_def)
                print("Training is Done")
          
    
                '''
                Start the multiprocessing process for testing deformation.
                '''
                '''
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                newval_def = newval_def.iloc[:,1:]
          
                X_def_newval = newval_def.iloc[:, :4].values
                y_def_newval = newval_def.iloc[:, newval_def.shape[1]-1].values
                
                new_X_val_def = scaler_min_max.fit_transform(X_def_newval)
    
                X_train_def, X_test_def, y_train_def, y_test_def_newval = train_test_split(new_X_val_def, y_def_newval, train_size=1, random_state=42)

                num_parts_def = 94
                # num_parts_def = 2634
                # num_parts_def = 1580
                split_size_def = len(X_test_def) // num_parts_def
                x_split_def = []
                y_split_def = []
                for i in range(num_parts_def):
                    start_index_def = i * split_size_def
                    end_index_def = (i + 1) * split_size_def
                    x_part_def = X_test_def[start_index_def:end_index_def]
                    y_part_def = y_test_def_newval[start_index_def:end_index_def]
                    x_split_def.append(x_part_def)
                    y_split_def.append(y_part_def)
                x_split_def = np.array(x_split_def)
                y_split_def = np.array(y_split_def)

                transposed_arrays_def = []
                for i in range(num_parts_def):
                    transposed_arrays_def.append(x_split_def[i].T)
                transposed_arrays_def = np.array(transposed_arrays_def)


                '''
                Get the accuracy for defromation and texture
                '''
                end_range = 94
                
                y_test_def = np.array([])
                jobs_def = []
                pipe_list_def = []
    

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_def, send_end_def = multiprocessing.Pipe(False)
                    p_def = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_def[i], y_split_def[i], nn_train_def, send_end_def))
                
                    jobs_def.append(p_def)
                    pipe_list_def.append(recv_end_def)
                    p_def.start()

                    
                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_def = y_split_def[i]  
                    y_test_def = np.append(y_test_def, new_elements_def)
                y_test_def = y_test_def.reshape((end_range, -1))

    
                # Join all cores results into one
                for proc in jobs_def:
                    proc.join()

    
                '''
                Metrics For Deformation
                '''
                # Putting all cores results into an array    
                result_list_def = np.array([x.recv().tolist() for x in pipe_list_def])
                y_pred_flat_def = result_list_def.flatten()
                y_test_flat_def = y_test_def.flatten()

                mae_def, mse_def, rmse_def, r2_def, accuracy_def, precision_def, recall_def, f1_def, conf_matrix_def, class_report_def = calculate_metrics(y_test_flat_def, y_pred_flat_def)
                print("mae:", mae_def*100)
                print("mse:", mse_def*100)
                print("rmse:", rmse_def*100)
                print("r2:", r2_def*100)
                
                print("Accuracy:", accuracy_def*100)
                print("Precision:", precision_def*100)
                print("Recall:", recall_def*100)
                print("F1 Score:", f1_def*100)
                print("Confusion Matrix:\n", conf_matrix_def)
                print("Classification report:\n", class_report_def)
                
                mae_ave_def.append(mae_def)
                mse_ave_def.append(mse_def)
                rmse_ave_def.append(rmse_def)
                r2_ave_def.append(r2_def)
                
                accuracy_ave_def.append(accuracy_def)
                precision_ave_def.append(precision_def)
                recall_ave_def.append(recall_def)
                f1_ave_def.append(f1_def)
                        
                
                ''' Iterate fold_no_loop'''
                fold_no_loop += 1
                print("fold_no loop is at fold_no_loop = ", fold_no_loop)
                
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            '''
            Combine K-fold for deformation
            '''
            k_fold_mae_def.append(mae_ave_def)
            k_fold_mse_def.append(mse_ave_def)
            k_fold_rmse_def.append(rmse_ave_def)
            k_fold_r2_def.append(r2_ave_def)
            
            k_fold_accuracy_def.append(accuracy_ave_def)
            k_fold_precision_def.append(precision_ave_def)
            k_fold_recall_def.append(recall_ave_def)
            k_fold_f1_def.append(f1_ave_def)
            
    
            '''
            Combine values for deformation
            '''
            mae_values_def.append(average_of_n_values(mae_ave_def)) 
            mse_values_def.append(average_of_n_values(mse_ave_def)) 
            rmse_values_def.append(average_of_n_values(rmse_ave_def)) 
            r2_values_def.append(average_of_n_values(r2_ave_def)) 
            
            accuracy_values_def.append(average_of_n_values(accuracy_ave_def))  
            precision_values_def.append(average_of_n_values(precision_ave_def)) 
            recall_values_def.append(average_of_n_values(recall_ave_def)) 
            f1_values_def.append(average_of_n_values(f1_ave_def)) 
  
   
            ''' Iterate gamma_loop'''
            gamma_loop += 1
            print("gamma loop is at gamma_loop = ", gamma_loop)
                
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

        '''
        Combine values for deformation
        '''
        mae_values_array_def.append(mae_values_def)
        mse_values_array_def.append(mse_values_def)
        rmse_values_array_def.append(rmse_values_def)
        r2_values_array_def.append(r2_values_def)
        
        accuracy_values_array_def.append(accuracy_values_def)
        precision_values_array_def.append(precision_values_def)
        recall_values_array_def.append(recall_values_def)
        f1_values_array_def.append(f1_values_def)
        
        
        
        ''' Iterate theta_loop'''
        theta_loop += 1
        print("theta loop is at theta_loop = ", theta_loop)
               
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)

    '''
    Print K-fold for deformation
    '''
    # print("These are the def mae for each k-fold split", k_fold_mae_def, "\n")
    # print("These are the def mse for each k-fold split", k_fold_mse_def, "\n")
    # print("These are the def rmse for each k-fold split", k_fold_rmse_def, "\n")
    # print("These are the def r2 for each k-fold split", k_fold_r2_def, "\n")
    
    # print("These are the def accuracy for each k-fold split", k_fold_accuracy_def, "\n")
    # print("These are the def precision for each k-fold split", k_fold_precision_def, "\n")
    # print("These are the def recall for each k-fold split", k_fold_recall_def, "\n")
    # print("These are the def f1 for each k-fold split", k_fold_f1_def, "\n")

    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    '''
    Plot for deformation
    '''

    plot_accuracy_bar_graph(mae_values_array_def, theta_values, "mae_deformation_plot", "mae_deformation_plot", "MAE")
    plot_accuracy_bar_graph(mse_values_array_def, theta_values, "mse_deformation_plot", "mse_deformation_plot", "MSE")
    plot_accuracy_bar_graph(rmse_values_array_def, theta_values, "rmse_deformation_plot", "rmse_deformation_plot", "RMSE")
    plot_accuracy_bar_graph(r2_values_array_def, theta_values, "r2_deformation_plot", "r2_deformation_plot", "R2")
    
    plot_accuracy_bar_graph(accuracy_values_array_def, theta_values, "accuracy_deformation_plot", "accuracy_deformation_plot", "ACCURACY")
    plot_accuracy_bar_graph(precision_values_array_def, theta_values, "precision_deformation_plot", "precision_deformation_plot", "PRECISION")
    plot_accuracy_bar_graph(recall_values_array_def, theta_values, "recall_deformation_plot", "recall_deformation_plot", "RECALL")
    plot_accuracy_bar_graph(f1_values_array_def, theta_values, "f1_deformation_plot", "f1_deformation_plot", "F1")
    
    
 
    print("mae_values_array_def", mae_values_array_def, "\n")
    print("mse_values_array_def", mse_values_array_def, "\n")
    print("rmse_values_array_def", rmse_values_array_def, "\n")
    print("r2_values_array_def", r2_values_array_def, "\n")
    
    print("accuracy_values_array_def", accuracy_values_array_def, "\n")
    print("precision_values_array_def", precision_values_array_def, "\n")
    print("recall_values_array_def", recall_values_array_def, "\n")
    print("f1_values_array_def", f1_values_array_def, "\n")
    
    plt.show()
    
    
if __name__ == '__main__':
    main()
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    