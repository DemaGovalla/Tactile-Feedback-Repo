<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData.csv file using the RFMN algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different RFMN parameters and returns the classification
            report. Done
"""

import numpy as np, pandas as pd, os, multiprocessing, matplotlib.pyplot as plt, matplotlib.cm as cm
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



def average_of_n_values(arr):
    if not arr:
        return "Input array is empty. Please provide values."
    return sum(arr) / len(arr)

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
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData.csv')
    sensor_data = sensor_data.iloc[:,0:]


    # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]

    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]


    directory_path_param = 'cross_val_RFMN_param'

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

    fold_no_tex = 1
    for train_index_tex, val_index_tex in skf.split(X_tex, y_tex):
        train_tex = X_tex.loc[train_index_tex,:]
        val_tex = X_tex.loc[val_index_tex,:]
        train_tex.to_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no_tex) + '.csv'))
        val_tex.to_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no_tex) + '.csv'))
        fold_no_tex += 1
        

    count = 0
    # Iterate directory
    for path in os.listdir(directory_path_param):
        # check if current path is a file
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)


    
    test_def = []
    test_tex = []
    

    # theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust these values according to your requirements
    # gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust these values according to your requirements
   
    theta_values = [1, 2, 3]  # Adjust these values according to your requirements
    gamma_values = [1, 2, 3, 4, 5, 6]  # Adjust these values according to your requirements 

    accuracy_values_array_def = []
    accuracy_values_array_tex = []

    for theta in theta_values:
        startTime4 = time() # Add a timer. 
        
        accuracy_values_def = []
        accuracy_values_tex = []
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
            accuracy_ave_def = []
            accuracy_ave_tex = []
            
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
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                newval_def = newval_def.iloc[:,1:]
          
                X_def_newval = newval_def.iloc[:, :4].values
                y_def_newval = newval_def.iloc[:, newval_def.shape[1]-1].values
                
                new_X_val_def = scaler_min_max.fit_transform(X_def_newval)
    
    
    
                X_train_def, X_test_def, y_train_def, y_test_def_newval = train_test_split(new_X_val_def, y_def_newval, train_size=1, random_state=42)
    

                num_parts_def = 94
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
                Test the texture data
                '''
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                newval_tex = newval_tex.iloc[:,1:]
                
                X_tex_newval = newval_tex.iloc[:, :-1].values
                y_tex_newval = newval_tex.iloc[:, newval_tex.shape[1]-1].values
                
                new_X_val_tex = scaler_min_max.fit_transform(X_tex_newval)
                
    
                X_train_tex, X_test_tex, y_train_tex, y_test_tex_newval = train_test_split(new_X_val_tex, y_tex_newval, train_size=1, random_state=42)
    
                num_parts_tex = 94
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
                
                y_test_def = np.array([])
                jobs_def = []
                pipe_list_def = []


                y_test_tex = np.array([])
                jobs_tex = []
                pipe_list_tex = []     

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_def, send_end_def = multiprocessing.Pipe(False)
                    recv_end_tex, send_end_tex = multiprocessing.Pipe(False)
                    p_def = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_def[i], y_split_def[i], nn_train_def, send_end_def))
                    p_tex = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_tex[i], y_split_tex[i], nn_train_tex, send_end_tex))
                
                    jobs_def.append(p_def)
                    pipe_list_def.append(recv_end_def)
                    p_def.start()

                    jobs_tex.append(p_tex)
                    pipe_list_tex.append(recv_end_tex)
                    p_tex.start()
                    
                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_def = y_split_def[i]  
                    y_test_def = np.append(y_test_def, new_elements_def)
                y_test_def = y_test_def.reshape((end_range, -1))

                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_tex = y_split_tex[i]  
                    y_test_tex = np.append(y_test_tex, new_elements_tex)
                y_test_tex = y_test_tex.reshape((end_range, -1))

                # Join all cores results into one
                for proc in jobs_def:
                    proc.join()

                # Join all cores results into one
                for proc in jobs_tex:
                    proc.join()
                    
                '''
                For Deformation
                '''
                # Putting all cores results into an array    
                result_list_def = np.array([x.recv().tolist() for x in pipe_list_def])
                y_pred_flat_def = result_list_def.flatten()
                y_test_flat_def = y_test_def.flatten()

                # Regression metrices
                mae_def = mean_absolute_error(y_test_flat_def, y_pred_flat_def)
                mse_def = mean_squared_error(y_test_flat_def, y_pred_flat_def)
                rmse_def = np.sqrt(mse_def)
                r2_def = r2_score(y_test_flat_def, y_pred_flat_def)
                
                print("mae:", mae_def)
                print("mse:", mse_def)
                print("rmse:", rmse_def)
                print("r2:", r2_def)
                
                # Classification metrices
                accuracy_def = accuracy_score(y_test_flat_def, y_pred_flat_def)
                precision_def = precision_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # recall_def = recall_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # f1_def = f1_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # auroc = roc_auc_score(y_test_flat, y_pred_flat)
                conf_matrix_def = confusion_matrix(y_test_flat_def, y_pred_flat_def)
                class_report_def = classification_report(y_test_flat_def, y_pred_flat_def)

                print("Accuracy:", accuracy_def*100)
                print("Precision:", precision_def)
                # print("Recall:", recall)
                # print("F1 Score:", f1)
                # print("AUROC:", auroc)
                print("Confusion Matrix:\n", conf_matrix_def)
                print("Classification report:\n", class_report_def)
                        
                '''
                For Texture
                '''
                # Putting all cores results into an array    
                result_list_tex = np.array([x.recv().tolist() for x in pipe_list_tex])
                y_pred_flat_tex = result_list_tex.flatten()
                y_test_flat_tex = y_test_tex.flatten()
    
            
                # Regression metrices
                mae_tex = mean_absolute_error(y_test_flat_tex, y_pred_flat_tex)
                mse_tex = mean_squared_error(y_test_flat_tex, y_pred_flat_tex)
                rmse_tex = np.sqrt(mse_tex)
                r2_tex = r2_score(y_test_flat_tex, y_pred_flat_tex)
                
                print("mae:", mae_tex)
                print("mse:", mse_tex)
                print("rmse:", rmse_tex)
                print("r2:", r2_tex)
                
                # Classification metrices
                accuracy_tex = accuracy_score(y_test_flat_tex, y_pred_flat_tex)
                precision_tex = precision_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # recall_tex = recall_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # f1_tex = f1_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # auroc = roc_auc_score(y_test_flat, y_pred_flat)
                conf_matrix_tex = confusion_matrix(y_test_flat_tex, y_pred_flat_tex)
                class_report_tex = classification_report(y_test_flat_tex, y_pred_flat_tex)

                print("Accuracy:", accuracy_tex*100)
                print("Precision:", precision_tex)
                # print("Recall:", recall)
                # print("F1 Score:", f1)
                # print("AUROC:", auroc)
                print("Confusion Matrix:\n", conf_matrix_tex)
                print("Classification report:\n", class_report_tex)
    
 
 
                accuracy_ave_def.append(accuracy_def)
                # print("This is the acuracy ave def array for 1 and 2 split", accuracy_ave_def, "\n")
                    
                accuracy_ave_tex.append(accuracy_tex)
                # print("This is the acuracy ave tex array for 1 and 2 split", accuracy_ave_tex, "\n")
                
            test_def.append(accuracy_ave_def)
            test_tex.append(accuracy_ave_tex)
    
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            
            total_values_accuracy_def = average_of_n_values(accuracy_ave_def)*100
            # print("The average of def values in the array", accuracy_ave_def, "is", total_values_accuracy_def)
            accuracy_values_def.append(total_values_accuracy_def)  
  
            total_values_accuracy_tex = average_of_n_values(accuracy_ave_tex)*100
            # print("The average of tex values in the array", accuracy_ave_tex, "is", total_values_accuracy_tex)
            accuracy_values_tex.append(total_values_accuracy_tex)  
        
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

            
        accuracy_values_array_def.append(accuracy_values_def)
        accuracy_values_array_tex.append(accuracy_values_tex)
        
        print("This is accuracy_values_array_def", accuracy_values_array_def, "\n")
        print("This is accuracy_values_array_tex", accuracy_values_array_tex, "\n")
    
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)



    print("These are the def accuracy for each k-fold split", test_def)
    print("These are the tex accuracy for each k-fold split", test_tex)

    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    '''
    For deformation
     '''
    # Extracting the number of gammas and thetas from the accuracy_values
    num_gammas_def = len(accuracy_values_array_def[0])
    num_thetas_def = len(accuracy_values_array_def)

    bar_width = 0.06
    index_def = np.arange(num_gammas_def)

    # Set custom colors for different thetas dynamically
    colors_def = cm.get_cmap('tab10', num_thetas_def)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas_def):
        bars = plt.bar(index_def + i * bar_width, accuracy_values_array_def[i], width=bar_width, label=f'Theta = {0.1 + i * 0.1}', color=colors_def(i))
        for bar, acc in zip(bars, accuracy_values_array_def[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Grouped Bar Graph with Different Gammas for Each Theta')
    plt.xticks(index_def + (bar_width * (num_thetas_def - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas_def)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    
    plt.savefig('deformation_plot_RFMN.png')  # Save as PNG file
    
    
    '''
    For texture
     '''
    # Extracting the number of gammas and thetas from the accuracy_values
    num_gammas_tex = len(accuracy_values_array_tex[0])
    num_thetas_tex = len(accuracy_values_array_tex)

    index_tex = np.arange(num_gammas_tex)

    # Set custom colors for different thetas dynamically
    colors_tex = cm.get_cmap('tab10', num_thetas_tex)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas_tex):
        bars = plt.bar(index_tex + i * bar_width, accuracy_values_array_tex[i], width=bar_width, label=f'Theta = {0.1 + i * 0.1}', color=colors_tex(i))
        for bar, acc in zip(bars, accuracy_values_array_tex[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')

    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Grouped Bar Graph with Different Gammas for Each Theta')
    plt.xticks(index_tex + (bar_width * (num_thetas_tex - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas_tex)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    plt.savefig('texture_plot_RFMN.png')  # Save as PNG file
    
    plt.show()

if __name__ == '__main__':
    main()
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData.csv file using the RFMN algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different RFMN parameters and returns the classification
            report. Done
"""

import numpy as np, pandas as pd, os, multiprocessing, matplotlib.pyplot as plt, matplotlib.cm as cm
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



def average_of_n_values(arr):
    if not arr:
        return "Input array is empty. Please provide values."
    return sum(arr) / len(arr)

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
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData.csv')
    sensor_data = sensor_data.iloc[:,0:]


    # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]

    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]


    directory_path_param = 'cross_val_RFMN_param'

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

    fold_no_tex = 1
    for train_index_tex, val_index_tex in skf.split(X_tex, y_tex):
        train_tex = X_tex.loc[train_index_tex,:]
        val_tex = X_tex.loc[val_index_tex,:]
        train_tex.to_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no_tex) + '.csv'))
        val_tex.to_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no_tex) + '.csv'))
        fold_no_tex += 1
        

    count = 0
    # Iterate directory
    for path in os.listdir(directory_path_param):
        # check if current path is a file
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)


    
    test_def = []
    test_tex = []
    

    # theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust these values according to your requirements
    # gamma_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust these values according to your requirements
   
    theta_values = [1, 2, 3]  # Adjust these values according to your requirements
    gamma_values = [1, 2, 3, 4, 5, 6]  # Adjust these values according to your requirements 

    accuracy_values_array_def = []
    accuracy_values_array_tex = []

    for theta in theta_values:
        startTime4 = time() # Add a timer. 
        
        accuracy_values_def = []
        accuracy_values_tex = []
        
        for gamma in gamma_values:
            
            startTime3 = time() # Add a timer. 
            
            accuracy_ave_def = []
            accuracy_ave_tex = []
            
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
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                newval_def = newval_def.iloc[:,1:]
          
                X_def_newval = newval_def.iloc[:, :4].values
                y_def_newval = newval_def.iloc[:, newval_def.shape[1]-1].values
                
                new_X_val_def = scaler_min_max.fit_transform(X_def_newval)
    
    
    
                X_train_def, X_test_def, y_train_def, y_test_def_newval = train_test_split(new_X_val_def, y_def_newval, train_size=1, random_state=42)
    

                num_parts_def = 94
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
                Test the texture data
                '''
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                newval_tex = newval_tex.iloc[:,1:]
                
                X_tex_newval = newval_tex.iloc[:, :-1].values
                y_tex_newval = newval_tex.iloc[:, newval_tex.shape[1]-1].values
                
                new_X_val_tex = scaler_min_max.fit_transform(X_tex_newval)
                
    
                X_train_tex, X_test_tex, y_train_tex, y_test_tex_newval = train_test_split(new_X_val_tex, y_tex_newval, train_size=1, random_state=42)
    
                num_parts_tex = 94
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
                
                y_test_def = np.array([])
                jobs_def = []
                pipe_list_def = []


                y_test_tex = np.array([])
                jobs_tex = []
                pipe_list_tex = []     

                # for i in range(0, os.cpu_count()):
                # for i in range(0, 94):
                for i in range(0, end_range):
                    startTime1 = time() # Add a timer. 
        
                    # range(0, os.cpu_count())
                    recv_end_def, send_end_def = multiprocessing.Pipe(False)
                    recv_end_tex, send_end_tex = multiprocessing.Pipe(False)
                    p_def = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_def[i], y_split_def[i], nn_train_def, send_end_def))
                    p_tex = multiprocessing.Process(target=testAlgo, args=(transposed_arrays_tex[i], y_split_tex[i], nn_train_tex, send_end_tex))
                
                    jobs_def.append(p_def)
                    pipe_list_def.append(recv_end_def)
                    p_def.start()

                    jobs_tex.append(p_tex)
                    pipe_list_tex.append(recv_end_tex)
                    p_tex.start()
                    
                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_def = y_split_def[i]  
                    y_test_def = np.append(y_test_def, new_elements_def)
                y_test_def = y_test_def.reshape((end_range, -1))

                # Create a new numpy array the same as y_split_def
                for i in range(end_range): 
                    new_elements_tex = y_split_tex[i]  
                    y_test_tex = np.append(y_test_tex, new_elements_tex)
                y_test_tex = y_test_tex.reshape((end_range, -1))

                # Join all cores results into one
                for proc in jobs_def:
                    proc.join()

                # Join all cores results into one
                for proc in jobs_tex:
                    proc.join()
                    
                '''
                For Deformation
                '''
                # Putting all cores results into an array    
                result_list_def = np.array([x.recv().tolist() for x in pipe_list_def])
                y_pred_flat_def = result_list_def.flatten()
                y_test_flat_def = y_test_def.flatten()

                # Regression metrices
                mae_def = mean_absolute_error(y_test_flat_def, y_pred_flat_def)
                mse_def = mean_squared_error(y_test_flat_def, y_pred_flat_def)
                rmse_def = np.sqrt(mse_def)
                r2_def = r2_score(y_test_flat_def, y_pred_flat_def)
                
                print("mae:", mae_def)
                print("mse:", mse_def)
                print("rmse:", rmse_def)
                print("r2:", r2_def)
                
                # Classification metrices
                accuracy_def = accuracy_score(y_test_flat_def, y_pred_flat_def)
                precision_def = precision_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # recall_def = recall_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # f1_def = f1_score(y_test_flat_def, y_pred_flat_def, average='micro')
                # auroc = roc_auc_score(y_test_flat, y_pred_flat)
                conf_matrix_def = confusion_matrix(y_test_flat_def, y_pred_flat_def)
                class_report_def = classification_report(y_test_flat_def, y_pred_flat_def)

                print("Accuracy:", accuracy_def*100)
                print("Precision:", precision_def)
                # print("Recall:", recall)
                # print("F1 Score:", f1)
                # print("AUROC:", auroc)
                print("Confusion Matrix:\n", conf_matrix_def)
                print("Classification report:\n", class_report_def)
                        
                '''
                For Texture
                '''
                # Putting all cores results into an array    
                result_list_tex = np.array([x.recv().tolist() for x in pipe_list_tex])
                y_pred_flat_tex = result_list_tex.flatten()
                y_test_flat_tex = y_test_tex.flatten()
    
            
                # Regression metrices
                mae_tex = mean_absolute_error(y_test_flat_tex, y_pred_flat_tex)
                mse_tex = mean_squared_error(y_test_flat_tex, y_pred_flat_tex)
                rmse_tex = np.sqrt(mse_tex)
                r2_tex = r2_score(y_test_flat_tex, y_pred_flat_tex)
                
                print("mae:", mae_tex)
                print("mse:", mse_tex)
                print("rmse:", rmse_tex)
                print("r2:", r2_tex)
                
                # Classification metrices
                accuracy_tex = accuracy_score(y_test_flat_tex, y_pred_flat_tex)
                precision_tex = precision_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # recall_tex = recall_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # f1_tex = f1_score(y_test_flat_tex, y_pred_flat_tex, average='micro')
                # auroc = roc_auc_score(y_test_flat, y_pred_flat)
                conf_matrix_tex = confusion_matrix(y_test_flat_tex, y_pred_flat_tex)
                class_report_tex = classification_report(y_test_flat_tex, y_pred_flat_tex)

                print("Accuracy:", accuracy_tex*100)
                print("Precision:", precision_tex)
                # print("Recall:", recall)
                # print("F1 Score:", f1)
                # print("AUROC:", auroc)
                print("Confusion Matrix:\n", conf_matrix_tex)
                print("Classification report:\n", class_report_tex)
    
 
 
                accuracy_ave_def.append(accuracy_def)
                # print("This is the acuracy ave def array for 1 and 2 split", accuracy_ave_def, "\n")
                    
                accuracy_ave_tex.append(accuracy_tex)
                # print("This is the acuracy ave tex array for 1 and 2 split", accuracy_ave_tex, "\n")
                
            test_def.append(accuracy_ave_def)
            test_tex.append(accuracy_ave_tex)
    
                
                # end_time1 = time()
                # totalTime1 = convert_seconds_to_minutes(end_time1 - startTime1)
                # print("Total time 1 is: ", totalTime1)
            
            
            total_values_accuracy_def = average_of_n_values(accuracy_ave_def)*100
            # print("The average of def values in the array", accuracy_ave_def, "is", total_values_accuracy_def)
            accuracy_values_def.append(total_values_accuracy_def)  
  
            total_values_accuracy_tex = average_of_n_values(accuracy_ave_tex)*100
            # print("The average of tex values in the array", accuracy_ave_tex, "is", total_values_accuracy_tex)
            accuracy_values_tex.append(total_values_accuracy_tex)  
        
            # end_time2 = time()
            # totalTime2 = convert_seconds_to_minutes(end_time2 - startTime2)
            # print("Total time 2 is: ", totalTime2)

            
        accuracy_values_array_def.append(accuracy_values_def)
        accuracy_values_array_tex.append(accuracy_values_tex)
        
        print("This is accuracy_values_array_def", accuracy_values_array_def, "\n")
        print("This is accuracy_values_array_tex", accuracy_values_array_tex, "\n")
    
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)



    print("These are the def accuracy for each k-fold split", test_def)
    print("These are the tex accuracy for each k-fold split", test_tex)

    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    '''
    For deformation
     '''
    # Extracting the number of gammas and thetas from the accuracy_values
    num_gammas_def = len(accuracy_values_array_def[0])
    num_thetas_def = len(accuracy_values_array_def)

    bar_width = 0.06
    index_def = np.arange(num_gammas_def)

    # Set custom colors for different thetas dynamically
    colors_def = cm.get_cmap('tab10', num_thetas_def)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas_def):
        bars = plt.bar(index_def + i * bar_width, accuracy_values_array_def[i], width=bar_width, label=f'Theta = {0.1 + i * 0.1}', color=colors_def(i))
        for bar, acc in zip(bars, accuracy_values_array_def[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Grouped Bar Graph with Different Gammas for Each Theta')
    plt.xticks(index_def + (bar_width * (num_thetas_def - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas_def)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    
    plt.savefig('deformation_plot_RFMN.png')  # Save as PNG file
    
    
    '''
    For texture
     '''
    # Extracting the number of gammas and thetas from the accuracy_values
    num_gammas_tex = len(accuracy_values_array_tex[0])
    num_thetas_tex = len(accuracy_values_array_tex)

    index_tex = np.arange(num_gammas_tex)

    # Set custom colors for different thetas dynamically
    colors_tex = cm.get_cmap('tab10', num_thetas_tex)

    plt.figure(figsize=(10, 6))
    for i in range(num_thetas_tex):
        bars = plt.bar(index_tex + i * bar_width, accuracy_values_array_tex[i], width=bar_width, label=f'Theta = {0.1 + i * 0.1}', color=colors_tex(i))
        for bar, acc in zip(bars, accuracy_values_array_tex[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')

    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.title('Grouped Bar Graph with Different Gammas for Each Theta')
    plt.xticks(index_tex + (bar_width * (num_thetas_tex - 1)) / 2, [f'Gamma {i+1}' for i in range(num_gammas_tex)])  # Custom x-ticks for gammas
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend outside the plot
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    plt.savefig('texture_plot_RFMN.png')  # Save as PNG file
    
    plt.show()

if __name__ == '__main__':
    main()
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    