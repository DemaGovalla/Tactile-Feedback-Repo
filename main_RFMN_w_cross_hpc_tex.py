<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_tex.csv file using the RFMN algorithm. 
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
    directory_path_result = 'RFMN_result_tex'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.07
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
 
    sensor_data = pd.read_csv('combined_sensorData_tex.csv')
    sensor_data = sensor_data.iloc[:,0:]


    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]

    directory_path_param = 'cross_val_RFMN_param_tex'

    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
        
    else:
        os.makedirs(directory_path_param)
        

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=42)


    fold_no_tex = 1
    for train_index_tex, val_index_tex in skf.split(X_tex, y_tex):
        train_tex = X_tex.loc[train_index_tex,:]
        val_tex = X_tex.loc[val_index_tex,:]
        train_tex.to_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no_tex) + '.csv'))
        val_tex.to_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no_tex) + '.csv'))
        fold_no_tex += 1

    count = 0
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

    theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    
    
    '''
    k-fold parameters for texture
    '''
    k_fold_mae_tex = []
    k_fold_mse_tex = []
    k_fold_rmse_tex = []
    k_fold_r2_tex = []
    
    k_fold_accuracy_tex = []
    k_fold_precision_tex = []
    k_fold_recall_tex = []
    k_fold_f1_tex = []

    
    '''
    Metric value arrays for texture
    '''
    mae_values_array_tex = []
    mse_values_array_tex = []
    rmse_values_array_tex = []
    r2_values_array_tex = []

    accuracy_values_array_tex = []
    precision_values_array_tex = []
    recall_values_array_tex = []
    f1_values_array_tex = []

    for theta in theta_values:
        

        '''
        Metric value for texture
        '''
        mae_values_tex = []
        mse_values_tex = []
        rmse_values_tex = []
        r2_values_tex = []
        
        accuracy_values_tex = []
        precision_values_tex = []
        recall_values_tex = []
        f1_values_tex = []
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
        
            '''
            Metric ave for texture
            '''
            mae_ave_tex = []
            mse_ave_tex = []
            rmse_ave_tex = []
            r2_ave_tex = []
            
            accuracy_ave_tex = []
            precision_ave_tex = []
            recall_ave_tex = []
            f1_ave_tex = []
            
            for fold_no in range(1,n_splits+1):
                
                startTime2 = time() # Add a timer. 
                scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
                
          
                '''
                Train the texture data
                '''
                newtrain_tex = pd.read_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no) + '.csv'))
                
                newtrain_tex = newtrain_tex.iloc[:,1:]
                new_X_train_tex = newtrain_tex.iloc[:,:-1].values
                new_y_train_tex = newtrain_tex.iloc[:,-1].values
                
                new_X_train_tex = scaler_min_max.fit_transform(new_X_train_tex)
                new_X_train_tex = new_X_train_tex.T
                
                nn_train_tex = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta*.1)
                nn_train_tex.train(new_X_train_tex, new_y_train_tex)
                print("Training is Done")
    
                '''
                Start the multiprocessing process for testing deformation.
                '''
    
                '''
                Test the texture data
                '''
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                newval_tex = newval_tex.iloc[:,1:]
                
                X_tex_newval = newval_tex.iloc[:, :-1].values
                y_tex_newval = newval_tex.iloc[:, newval_tex.shape[1]-1].values
                
                new_X_val_tex = scaler_min_max.fit_transform(X_tex_newval)
    
                X_train_tex, X_test_tex, y_train_tex, y_test_tex_newval = train_test_split(new_X_val_tex, y_tex_newval, train_size=1, random_state=42)
    
                num_parts_tex = 94
                # num_parts_tex = 2634
                # num_parts_tex = 1580
                
                split_size_tex = len(new_X_val_tex) // num_parts_tex
                x_split_tex = []
                y_split_tex = []
                for i in range(num_parts_tex):
                    start_index_tex = i * split_size_tex
                    end_index_tex = (i + 1) * split_size_tex
                    x_part_tex = X_test_tex[start_index_tex:end_index_tex]
                    y_part_tex = y_test_tex_newval[start_index_tex:end_index_tex]
                    x_split_tex.append(x_part_tex)
                    y_split_tex.append(y_part_tex)
                x_split_tex = np.array(x_split_tex)
                y_split_tex = np.array(y_split_tex)

                transposed_arrays_tex = []
                for i in range(num_parts_tex):
                    transposed_arrays_tex.append(x_split_tex[i].T)
                transposed_arrays_tex = np.array(transposed_arrays_tex)

                '''
                Get the accuracy for defromation and texture
                '''
                end_range = 94
        

                y_test_tex = np.array([])
                jobs_tex = []
                pipe_list_tex = []     

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_tex, send_end_tex = multiprocessing.Pipe(False)
                    p_tex = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_tex[i], y_split_tex[i], nn_train_tex, send_end_tex))
                

                    jobs_tex.append(p_tex)
                    pipe_list_tex.append(recv_end_tex)
                    p_tex.start()


                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_tex = y_split_tex[i]  
                    y_test_tex = np.append(y_test_tex, new_elements_tex)
                y_test_tex = y_test_tex.reshape((end_range, -1))


                # Join all cores results into one
                for proc in jobs_tex:
                    proc.join()
                    
                        
                '''
                Metrics For Texture
                '''
                # Putting all cores results into an array    
                result_list_tex = np.array([x.recv().tolist() for x in pipe_list_tex])
                y_pred_flat_tex = result_list_tex.flatten()
                y_test_flat_tex = y_test_tex.flatten()
            
                mae_tex, mse_tex, rmse_tex, r2_tex, accuracy_tex, precision_tex, recall_tex, f1_tex, conf_matrix_tex, class_report_tex = calculate_metrics(y_test_flat_tex, y_pred_flat_tex)
                print("mae:", mae_tex*100)
                print("mse:", mse_tex*100)
                print("rmse:", rmse_tex*100)
                print("r2:", r2_tex*100)
                
                print("Accuracy:", accuracy_tex*100)
                print("Precision:", precision_tex*100)
                print("Recall:", recall_tex*100)
                print("F1 Score:", f1_tex*100)
                print("Confusion Matrix:\n", conf_matrix_tex)
                print("Classification report:\n", class_report_tex)
 
                mae_ave_tex.append(mae_tex)
                mse_ave_tex.append(mse_tex)
                rmse_ave_tex.append(rmse_tex)
                r2_ave_tex.append(r2_tex)
                
                accuracy_ave_tex.append(accuracy_tex)
                precision_ave_tex.append(precision_tex)
                recall_ave_tex.append(recall_tex)
                f1_ave_tex.append(f1_tex)
                
                
                
                ''' Iterate fold_no_loop'''
                fold_no_loop += 1
                print("fold_no loop is at fold_no_loop = ", fold_no_loop)
                
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            
            '''
            Combine K-fold for texture
            '''    
            k_fold_mae_tex.append(mae_ave_tex)
            k_fold_mse_tex.append(mse_ave_tex)
            k_fold_rmse_tex.append(rmse_ave_tex)
            k_fold_r2_tex.append(r2_ave_tex)
            
            k_fold_accuracy_tex.append(accuracy_ave_tex)
            k_fold_precision_tex.append(precision_ave_tex)
            k_fold_recall_tex.append(recall_ave_tex)
            k_fold_f1_tex.append(f1_ave_tex) 
            
  
            '''
            Combine values for texture
            '''
            mae_values_tex.append(average_of_n_values(mae_ave_tex)) 
            mse_values_tex.append(average_of_n_values(mse_ave_tex)) 
            rmse_values_tex.append(average_of_n_values(rmse_ave_tex)) 
            r2_values_tex.append(average_of_n_values(r2_ave_tex)) 
            
            accuracy_values_tex.append(average_of_n_values(accuracy_ave_tex))  
            precision_values_tex.append(average_of_n_values(precision_ave_tex)) 
            recall_values_tex.append(average_of_n_values(recall_ave_tex)) 
            f1_values_tex.append(average_of_n_values(f1_ave_tex)) 
   
   
            ''' Iterate gamma_loop'''
            gamma_loop += 1
            print("gamma loop is at gamma_loop = ", gamma_loop)
                
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

        
        '''
        Combine values for texture
        '''    
        mae_values_array_tex.append(mae_values_tex)
        mse_values_array_tex.append(mse_values_tex)
        rmse_values_array_tex.append(rmse_values_tex)
        r2_values_array_tex.append(r2_values_tex)
        
        accuracy_values_array_tex.append(accuracy_values_tex)
        precision_values_array_tex.append(precision_values_tex)
        recall_values_array_tex.append(recall_values_tex)
        f1_values_array_tex.append(f1_values_tex)    
        
        
        ''' Iterate theta_loop'''
        theta_loop += 1
        print("theta loop is at theta_loop = ", theta_loop)
               
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)


    '''
    Print K-fold for texture
    '''
    # print("These are the tex mae for each k-fold split", k_fold_mae_tex, "\n")
    # print("These are the tex mse for each k-fold split", k_fold_mse_tex, "\n")
    # print("These are the tex rmse for each k-fold split", k_fold_rmse_tex, "\n")
    # print("These are the tex r2 for each k-fold split", k_fold_r2_tex, "\n")
    
    # print("These are the tex accuracy for each k-fold split", k_fold_accuracy_tex, "\n")
    # print("These are the tex precision for each k-fold split", k_fold_precision_tex, "\n")
    # print("These are the tex recall for each k-fold split", k_fold_recall_tex, "\n")
    # print("These are the tex f1 for each k-fold split", k_fold_f1_tex, "\n")
    
    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    
    '''
    Plot for texture
    '''
 
    plot_accuracy_bar_graph(mae_values_array_tex, theta_values, "mae_texture_plot", "mae_texture_plot", "MAE")
    plot_accuracy_bar_graph(mse_values_array_tex, theta_values, "mse_texture_plot", "mse_texture_plot", "MSE")
    plot_accuracy_bar_graph(rmse_values_array_tex, theta_values, "rmse_texture_plot", "rmse_texture_plot", "RMSE")
    plot_accuracy_bar_graph(r2_values_array_tex, theta_values, "r2_texture_plot", "r2_texture_plot", "R2")
    
    plot_accuracy_bar_graph(accuracy_values_array_tex, theta_values, "accuracy_texture_plot", "accuracy_texture_plot", "ACCURACY")
    plot_accuracy_bar_graph(precision_values_array_tex, theta_values, "precision_texture_plot", "precision_texture_plot", "PRECISION")
    plot_accuracy_bar_graph(recall_values_array_tex, theta_values, "recall_texture_plot", "recall_texture_plot", "RECALL")
    plot_accuracy_bar_graph(f1_values_array_tex, theta_values, "f1_texture_plot", "f1_texture_plot", "F1")
    
    
    print("mae_values_array_tex", mae_values_array_tex, "\n")
    print("mse_values_array_tex", mse_values_array_tex, "\n")
    print("rmse_values_array_tex", rmse_values_array_tex, "\n")
    print("r2_values_array_tex", r2_values_array_tex, "\n")
    
    print("accuracy_values_array_tex", accuracy_values_array_tex, "\n")
    print("precision_values_array_tex", precision_values_array_tex, "\n")
    print("recall_values_array_tex", recall_values_array_tex, "\n")
    print("f1_values_array_tex", f1_values_array_tex, "\n")
    
    plt.show()


if __name__ == '__main__':
    main()
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData_tex.csv file using the RFMN algorithm. 
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
    directory_path_result = 'RFMN_result_tex'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.07
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
 
    sensor_data = pd.read_csv('combined_sensorData_tex.csv')
    sensor_data = sensor_data.iloc[:,0:]


    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]

    directory_path_param = 'cross_val_RFMN_param_tex'

    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
        
    else:
        os.makedirs(directory_path_param)
        

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=42)


    fold_no_tex = 1
    for train_index_tex, val_index_tex in skf.split(X_tex, y_tex):
        train_tex = X_tex.loc[train_index_tex,:]
        val_tex = X_tex.loc[val_index_tex,:]
        train_tex.to_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no_tex) + '.csv'))
        val_tex.to_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no_tex) + '.csv'))
        fold_no_tex += 1

    count = 0
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

    theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
    
    
    '''
    k-fold parameters for texture
    '''
    k_fold_mae_tex = []
    k_fold_mse_tex = []
    k_fold_rmse_tex = []
    k_fold_r2_tex = []
    
    k_fold_accuracy_tex = []
    k_fold_precision_tex = []
    k_fold_recall_tex = []
    k_fold_f1_tex = []

    
    '''
    Metric value arrays for texture
    '''
    mae_values_array_tex = []
    mse_values_array_tex = []
    rmse_values_array_tex = []
    r2_values_array_tex = []

    accuracy_values_array_tex = []
    precision_values_array_tex = []
    recall_values_array_tex = []
    f1_values_array_tex = []

    for theta in theta_values:
        

        '''
        Metric value for texture
        '''
        mae_values_tex = []
        mse_values_tex = []
        rmse_values_tex = []
        r2_values_tex = []
        
        accuracy_values_tex = []
        precision_values_tex = []
        recall_values_tex = []
        f1_values_tex = []
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
        
            '''
            Metric ave for texture
            '''
            mae_ave_tex = []
            mse_ave_tex = []
            rmse_ave_tex = []
            r2_ave_tex = []
            
            accuracy_ave_tex = []
            precision_ave_tex = []
            recall_ave_tex = []
            f1_ave_tex = []
            
            for fold_no in range(1,n_splits+1):
                
                startTime2 = time() # Add a timer. 
                scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
                
          
                '''
                Train the texture data
                '''
                newtrain_tex = pd.read_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no) + '.csv'))
                
                newtrain_tex = newtrain_tex.iloc[:,1:]
                new_X_train_tex = newtrain_tex.iloc[:,:-1].values
                new_y_train_tex = newtrain_tex.iloc[:,-1].values
                
                new_X_train_tex = scaler_min_max.fit_transform(new_X_train_tex)
                new_X_train_tex = new_X_train_tex.T
                
                nn_train_tex = ReflexFuzzyNeuroNetwork(gamma=gamma, theta=theta*.1)
                nn_train_tex.train(new_X_train_tex, new_y_train_tex)
                print("Training is Done")
    
                '''
                Start the multiprocessing process for testing deformation.
                '''
    
                '''
                Test the texture data
                '''
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                newval_tex = newval_tex.iloc[:,1:]
                
                X_tex_newval = newval_tex.iloc[:, :-1].values
                y_tex_newval = newval_tex.iloc[:, newval_tex.shape[1]-1].values
                
                new_X_val_tex = scaler_min_max.fit_transform(X_tex_newval)
    
                X_train_tex, X_test_tex, y_train_tex, y_test_tex_newval = train_test_split(new_X_val_tex, y_tex_newval, train_size=1, random_state=42)
    
                num_parts_tex = 94
                # num_parts_tex = 2634
                # num_parts_tex = 1580
                
                split_size_tex = len(new_X_val_tex) // num_parts_tex
                x_split_tex = []
                y_split_tex = []
                for i in range(num_parts_tex):
                    start_index_tex = i * split_size_tex
                    end_index_tex = (i + 1) * split_size_tex
                    x_part_tex = X_test_tex[start_index_tex:end_index_tex]
                    y_part_tex = y_test_tex_newval[start_index_tex:end_index_tex]
                    x_split_tex.append(x_part_tex)
                    y_split_tex.append(y_part_tex)
                x_split_tex = np.array(x_split_tex)
                y_split_tex = np.array(y_split_tex)

                transposed_arrays_tex = []
                for i in range(num_parts_tex):
                    transposed_arrays_tex.append(x_split_tex[i].T)
                transposed_arrays_tex = np.array(transposed_arrays_tex)

                '''
                Get the accuracy for defromation and texture
                '''
                end_range = 94
        

                y_test_tex = np.array([])
                jobs_tex = []
                pipe_list_tex = []     

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_tex, send_end_tex = multiprocessing.Pipe(False)
                    p_tex = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_tex[i], y_split_tex[i], nn_train_tex, send_end_tex))
                

                    jobs_tex.append(p_tex)
                    pipe_list_tex.append(recv_end_tex)
                    p_tex.start()


                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_tex = y_split_tex[i]  
                    y_test_tex = np.append(y_test_tex, new_elements_tex)
                y_test_tex = y_test_tex.reshape((end_range, -1))


                # Join all cores results into one
                for proc in jobs_tex:
                    proc.join()
                    
                        
                '''
                Metrics For Texture
                '''
                # Putting all cores results into an array    
                result_list_tex = np.array([x.recv().tolist() for x in pipe_list_tex])
                y_pred_flat_tex = result_list_tex.flatten()
                y_test_flat_tex = y_test_tex.flatten()
            
                mae_tex, mse_tex, rmse_tex, r2_tex, accuracy_tex, precision_tex, recall_tex, f1_tex, conf_matrix_tex, class_report_tex = calculate_metrics(y_test_flat_tex, y_pred_flat_tex)
                print("mae:", mae_tex*100)
                print("mse:", mse_tex*100)
                print("rmse:", rmse_tex*100)
                print("r2:", r2_tex*100)
                
                print("Accuracy:", accuracy_tex*100)
                print("Precision:", precision_tex*100)
                print("Recall:", recall_tex*100)
                print("F1 Score:", f1_tex*100)
                print("Confusion Matrix:\n", conf_matrix_tex)
                print("Classification report:\n", class_report_tex)
 
                mae_ave_tex.append(mae_tex)
                mse_ave_tex.append(mse_tex)
                rmse_ave_tex.append(rmse_tex)
                r2_ave_tex.append(r2_tex)
                
                accuracy_ave_tex.append(accuracy_tex)
                precision_ave_tex.append(precision_tex)
                recall_ave_tex.append(recall_tex)
                f1_ave_tex.append(f1_tex)
                
                
                
                ''' Iterate fold_no_loop'''
                fold_no_loop += 1
                print("fold_no loop is at fold_no_loop = ", fold_no_loop)
                
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            
            '''
            Combine K-fold for texture
            '''    
            k_fold_mae_tex.append(mae_ave_tex)
            k_fold_mse_tex.append(mse_ave_tex)
            k_fold_rmse_tex.append(rmse_ave_tex)
            k_fold_r2_tex.append(r2_ave_tex)
            
            k_fold_accuracy_tex.append(accuracy_ave_tex)
            k_fold_precision_tex.append(precision_ave_tex)
            k_fold_recall_tex.append(recall_ave_tex)
            k_fold_f1_tex.append(f1_ave_tex) 
            
  
            '''
            Combine values for texture
            '''
            mae_values_tex.append(average_of_n_values(mae_ave_tex)) 
            mse_values_tex.append(average_of_n_values(mse_ave_tex)) 
            rmse_values_tex.append(average_of_n_values(rmse_ave_tex)) 
            r2_values_tex.append(average_of_n_values(r2_ave_tex)) 
            
            accuracy_values_tex.append(average_of_n_values(accuracy_ave_tex))  
            precision_values_tex.append(average_of_n_values(precision_ave_tex)) 
            recall_values_tex.append(average_of_n_values(recall_ave_tex)) 
            f1_values_tex.append(average_of_n_values(f1_ave_tex)) 
   
   
            ''' Iterate gamma_loop'''
            gamma_loop += 1
            print("gamma loop is at gamma_loop = ", gamma_loop)
                
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

        
        '''
        Combine values for texture
        '''    
        mae_values_array_tex.append(mae_values_tex)
        mse_values_array_tex.append(mse_values_tex)
        rmse_values_array_tex.append(rmse_values_tex)
        r2_values_array_tex.append(r2_values_tex)
        
        accuracy_values_array_tex.append(accuracy_values_tex)
        precision_values_array_tex.append(precision_values_tex)
        recall_values_array_tex.append(recall_values_tex)
        f1_values_array_tex.append(f1_values_tex)    
        
        
        ''' Iterate theta_loop'''
        theta_loop += 1
        print("theta loop is at theta_loop = ", theta_loop)
               
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)


    '''
    Print K-fold for texture
    '''
    # print("These are the tex mae for each k-fold split", k_fold_mae_tex, "\n")
    # print("These are the tex mse for each k-fold split", k_fold_mse_tex, "\n")
    # print("These are the tex rmse for each k-fold split", k_fold_rmse_tex, "\n")
    # print("These are the tex r2 for each k-fold split", k_fold_r2_tex, "\n")
    
    # print("These are the tex accuracy for each k-fold split", k_fold_accuracy_tex, "\n")
    # print("These are the tex precision for each k-fold split", k_fold_precision_tex, "\n")
    # print("These are the tex recall for each k-fold split", k_fold_recall_tex, "\n")
    # print("These are the tex f1 for each k-fold split", k_fold_f1_tex, "\n")
    
    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    
    '''
    Plot for texture
    '''
 
    plot_accuracy_bar_graph(mae_values_array_tex, theta_values, "mae_texture_plot", "mae_texture_plot", "MAE")
    plot_accuracy_bar_graph(mse_values_array_tex, theta_values, "mse_texture_plot", "mse_texture_plot", "MSE")
    plot_accuracy_bar_graph(rmse_values_array_tex, theta_values, "rmse_texture_plot", "rmse_texture_plot", "RMSE")
    plot_accuracy_bar_graph(r2_values_array_tex, theta_values, "r2_texture_plot", "r2_texture_plot", "R2")
    
    plot_accuracy_bar_graph(accuracy_values_array_tex, theta_values, "accuracy_texture_plot", "accuracy_texture_plot", "ACCURACY")
    plot_accuracy_bar_graph(precision_values_array_tex, theta_values, "precision_texture_plot", "precision_texture_plot", "PRECISION")
    plot_accuracy_bar_graph(recall_values_array_tex, theta_values, "recall_texture_plot", "recall_texture_plot", "RECALL")
    plot_accuracy_bar_graph(f1_values_array_tex, theta_values, "f1_texture_plot", "f1_texture_plot", "F1")
    
    
    print("mae_values_array_tex", mae_values_array_tex, "\n")
    print("mse_values_array_tex", mse_values_array_tex, "\n")
    print("rmse_values_array_tex", rmse_values_array_tex, "\n")
    print("r2_values_array_tex", r2_values_array_tex, "\n")
    
    print("accuracy_values_array_tex", accuracy_values_array_tex, "\n")
    print("precision_values_array_tex", precision_values_array_tex, "\n")
    print("recall_values_array_tex", recall_values_array_tex, "\n")
    print("f1_values_array_tex", f1_values_array_tex, "\n")
    
    plt.show()


if __name__ == '__main__':
    main()
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    