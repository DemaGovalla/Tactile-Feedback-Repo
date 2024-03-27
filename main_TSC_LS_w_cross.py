<<<<<<< HEAD
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

'''
Start of Cross validation results
'''
import numpy as np, pandas as pd, random, sys, seaborn as sns, os, statistics, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from torch import nn, optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.multiclass import unique_labels
from TSC_LS import LearningShapelets
from time import time
import warnings, matplotlib.cm as cm
import matplotlib, shutil
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
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

def plot_accuracy_bar_graph(values_array, dist_measure_strings, plot_title, save_filename):
    directory_path_result = 'TSC_LS_result'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.06
    index = np.arange(num_gammas)
    colors = plt.cm.get_cmap('tab10', num_thetas)
    
    plt.figure(figsize=(10, 6))
    for i in range(num_thetas):
        bars = plt.bar(index + i * bar_width, values_array[i], width=bar_width, label=f'Dist_measure = {dist_measure_strings[i]}', color=colors(i))
        for bar, acc in zip(bars, values_array[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')

    plt.xlabel('shapelets length')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.xticks(index + (bar_width * (num_thetas - 1)) / 2, [f'shap_len {i+1}' for i in range(num_gammas)])  
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(directory_path_result+ '/' + save_filename)  

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

def eval_accuracy(model, X, Y):
    result = []
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
        print(predictions)
    print(type(predictions))
    print(predictions.shape)

    result.append(predictions)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return result

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

def main():
    
    ''' Initiat loop counter '''
    startTime4 = time() 
    dist_measure_loop = 0
    shapelets_len_loop = 0
    fold_no_loop = 0
    
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData.csv')
    sensor_data = sensor_data.iloc[:,0:]

   # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]
    
    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]

    directory_path_param = 'cross_val_TSC_LS_param'
    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
    else:
        os.makedirs(directory_path_param)

    n_splits=2
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
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

   
    dist_measure_strings = ["euclidean", "cosine"]  
    shapelets_len_values = [2, 3]  
    
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
    
    for dist_measure in dist_measure_strings:
        
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
        
        for shapelets_len in shapelets_len_values:
            
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

                '''
                Train the deformation data
                '''
                newtrain_def = pd.read_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no) + '.csv'))
                columns_to_combine_newtrain_def = newtrain_def.columns[1:-1]
                combined_array_newtrain_def = []
                
                for column in columns_to_combine_newtrain_def:
                    new_array_def = newtrain_def[column].to_numpy()
                    combined_array_newtrain_def.append(new_array_def)

                combined_array_newtrain_def = np.column_stack(combined_array_newtrain_def)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_train_def = np.array([label_map[label] for label in newtrain_def.iloc[:,-1].to_numpy()])
                new_X_train_def = combined_array_newtrain_def.reshape(new_y_train_def.size, columns_to_combine_newtrain_def.size,1)
                new_X_train_def, scaler_def = normalize_data(new_X_train_def)
                
            
                n_ts_def, n_channels_def, len_ts_def = new_X_train_def.shape
                loss_func_def = nn.CrossEntropyLoss()
                num_classes_def = len(set(new_y_train_def))
                # learn 2 shapelets of length 130
                shapelets_size_and_len_def = {1: shapelets_len}
                # dist_measure_def = "euclidean"
                dist_measure_def = dist_measure
                
                lr_def = 1e-2
                wd_def = 1e-3
                epsilon_def = 1e-7
            
                learning_shapelets_def = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_def,
                                                    in_channels=n_channels_def,
                                                    num_classes=num_classes_def,
                                                    loss_func=loss_func_def,
                                                    to_cuda=False,
                                                    verbose=1,
                                                    dist_measure=dist_measure_def)

                for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_def.items()):
                    weights_block = get_weights_via_kmeans(new_X_train_def, shapelets_size, num_shapelets)
                    learning_shapelets_def.set_shapelet_weights_of_block(i, weights_block)
            
                optimizer = optim.Adam(learning_shapelets_def.model.parameters(), lr=lr_def, weight_decay=wd_def, eps=epsilon_def)
                learning_shapelets_def.set_optimizer(optimizer)
                losses = learning_shapelets_def.fit(new_X_train_def, new_y_train_def, epochs=2000, batch_size=256, shuffle=False, drop_last=False)
                print("Training is Done")
                
                
                '''
                Train the texture data
                '''
                # Used for training the model
                newtrain_tex = pd.read_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no) + '.csv'))
                columns_to_combine_newtrain_tex = newtrain_tex.columns[1:-1]
                combined_array_newtrain_tex = []

                for column in columns_to_combine_newtrain_tex:
                    new_array_tex = newtrain_tex[column].to_numpy()
                    combined_array_newtrain_tex.append(new_array_tex)

                combined_array_newtrain_tex = np.column_stack(combined_array_newtrain_tex)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_train_tex = np.array([label_map[label] for label in newtrain_tex.iloc[:,-1].to_numpy()])
                new_X_train_tex = combined_array_newtrain_tex.reshape(new_y_train_tex.size, columns_to_combine_newtrain_tex.size,1)
                new_X_train_tex, scaler_tex = normalize_data(new_X_train_tex)
                
            
                n_ts_tex, n_channels_tex, len_ts_tex = new_X_train_tex.shape
                loss_func_tex = nn.CrossEntropyLoss()
                num_classes_tex = len(set(new_y_train_tex))
                # learn 2 shapelets of length 130
                shapelets_size_and_len_tex = {1: shapelets_len}
                # dist_measure_tex = "euclidean"
                # dist_measure_tex = "cross-correlation"
                dist_measure_tex = dist_measure
                
                
                lr_tex = 1e-2
                wd_tex = 1e-3
                epsilon_tex = 1e-7
                learning_shapelets_tex = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_tex,
                                                    in_channels=n_channels_tex,
                                                    num_classes=num_classes_tex,
                                                    loss_func=loss_func_tex,
                                                    to_cuda=False,
                                                    verbose=1,
                                                    dist_measure=dist_measure_tex)
                
                for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_tex.items()):
                    weights_block = get_weights_via_kmeans(new_X_train_tex, shapelets_size, num_shapelets)
                    learning_shapelets_tex.set_shapelet_weights_of_block(i, weights_block)
            
                optimizer = optim.Adam(learning_shapelets_tex.model.parameters(), lr=lr_tex, weight_decay=wd_tex, eps=epsilon_tex)
                learning_shapelets_tex.set_optimizer(optimizer)

                losses = learning_shapelets_tex.fit(new_X_train_tex, new_y_train_tex, epochs=2000, batch_size=256, shuffle=False, drop_last=False)
                print("Training is Done")
                  
                '''
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                columns_to_combine_newval_def = newval_def.columns[1:-1]
                combined_array_newval_def = []
                for column in columns_to_combine_newval_def:
                    new_array_def = newval_def[column].to_numpy()
                    combined_array_newval_def.append(new_array_def)
                combined_array_newval_def = np.column_stack(combined_array_newval_def)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_val_def = np.array([label_map[label] for label in newval_def.iloc[:,-1].to_numpy()])
                new_X_val_def = combined_array_newval_def.reshape(new_y_val_def.size,columns_to_combine_newval_def.size,1)
                new_X_val_def, scaler = normalize_data(new_X_val_def)

                y_predlr_val_def = np.array(eval_accuracy(learning_shapelets_def,new_X_val_def,new_y_val_def)).reshape(-1)

                '''
                Test the texture data
                '''                          
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                columns_to_combine_newval_tex = newval_tex.columns[1:-1]
                combined_array_newval_tex = []
                for column in columns_to_combine_newval_tex:
                    new_array_tex = newval_tex[column].to_numpy()
                    combined_array_newval_tex.append(new_array_tex)
                combined_array_newval_tex = np.column_stack(combined_array_newval_tex)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_val_tex = np.array([label_map[label] for label in newval_tex.iloc[:,-1].to_numpy()])
                new_X_val_tex = combined_array_newval_tex.reshape(new_y_val_tex.size,columns_to_combine_newval_tex.size,1)
                new_X_val_tex, scaler = normalize_data(new_X_val_tex)
                
                y_predlr_val_tex = np.array(eval_accuracy(learning_shapelets_tex,new_X_val_tex,new_y_val_tex)).reshape(-1)
           
                '''
                Metrics For Deformation
                '''   
                mae_def, mse_def, rmse_def, r2_def, accuracy_def, precision_def, recall_def, f1_def, conf_matrix_def, class_report_def = calculate_metrics(new_y_val_def, y_predlr_val_def)
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
                
                '''
                Metrics For Texture
                '''
                mae_tex, mse_tex, rmse_tex, r2_tex, accuracy_tex, precision_tex, recall_tex, f1_tex, conf_matrix_tex, class_report_tex = calculate_metrics(new_y_val_tex, y_predlr_val_tex)
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
        
        
            ''' Iterate fold_no_loop'''
            dist_measure_loop += 1
            print("dist_measure loop is at dist_measure_loop = ", dist_measure_loop)
            
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
    
        ''' Iterate fold_no_loop'''
        shapelets_len_loop += 1
        print("shapelets_len loop is at shapelets_len_loop = ", shapelets_len_loop)
        
                              
    
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)                    
        
        
    '''
    Print K-fold for deformation
    '''
    print("These are the def mae for each k-fold split", k_fold_mae_def, "\n")
    print("These are the def mse for each k-fold split", k_fold_mse_def, "\n")
    print("These are the def rmse for each k-fold split", k_fold_rmse_def, "\n")
    print("These are the def r2 for each k-fold split", k_fold_r2_def, "\n")
    
    print("These are the def accuracy for each k-fold split", k_fold_accuracy_def, "\n")
    print("These are the def precision for each k-fold split", k_fold_precision_def, "\n")
    print("These are the def recall for each k-fold split", k_fold_recall_def, "\n")
    print("These are the def f1 for each k-fold split", k_fold_f1_def, "\n")

    '''
    Print K-fold for texture
    '''
    print("These are the tex mae for each k-fold split", k_fold_mae_tex, "\n")
    print("These are the tex mse for each k-fold split", k_fold_mse_tex, "\n")
    print("These are the tex rmse for each k-fold split", k_fold_rmse_tex, "\n")
    print("These are the tex r2 for each k-fold split", k_fold_r2_tex, "\n")
    
    print("These are the tex accuracy for each k-fold split", k_fold_accuracy_tex, "\n")
    print("These are the tex precision for each k-fold split", k_fold_precision_tex, "\n")
    print("These are the tex recall for each k-fold split", k_fold_recall_tex, "\n")
    print("These are the tex f1 for each k-fold split", k_fold_f1_tex, "\n")
    
    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    
    '''
    Plot for deformation
    '''
    plot_accuracy_bar_graph(mae_values_array_def, dist_measure_strings, "mae_deformation_plot", "mae_deformation_plot")
    plot_accuracy_bar_graph(mse_values_array_def, dist_measure_strings, "mse_deformation_plot", "mse_deformation_plot")
    plot_accuracy_bar_graph(rmse_values_array_def, dist_measure_strings, "rmse_deformation_plot", "rmse_deformation_plot")
    plot_accuracy_bar_graph(r2_values_array_def, dist_measure_strings, "r2_deformation_plot", "r2_deformation_plot")
    
    plot_accuracy_bar_graph(accuracy_values_array_def, dist_measure_strings, "accuracy_deformation_plot", "accuracy_deformation_plot")
    plot_accuracy_bar_graph(precision_values_array_def, dist_measure_strings, "precision_deformation_plot", "precision_deformation_plot")
    plot_accuracy_bar_graph(recall_values_array_def, dist_measure_strings, "recall_deformation_plot", "recall_deformation_plot")
    plot_accuracy_bar_graph(f1_values_array_def, dist_measure_strings, "f1_deformation_plot", "f1_deformation_plot")
    
    '''
    Plot for texture
    '''
    plot_accuracy_bar_graph(mae_values_array_tex, dist_measure_strings, "mae_texture_plot", "mae_texture_plot")
    plot_accuracy_bar_graph(mse_values_array_tex, dist_measure_strings, "mse_texture_plot", "mse_texture_plot")
    plot_accuracy_bar_graph(rmse_values_array_tex, dist_measure_strings, "rmse_texture_plot", "rmse_texture_plot")
    plot_accuracy_bar_graph(r2_values_array_tex, dist_measure_strings, "r2_texture_plot", "r2_texture_plot")
    
    plot_accuracy_bar_graph(accuracy_values_array_tex, dist_measure_strings, "accuracy_texture_plot", "accuracy_texture_plot")
    plot_accuracy_bar_graph(precision_values_array_tex, dist_measure_strings, "precision_texture_plot", "precision_texture_plot")
    plot_accuracy_bar_graph(recall_values_array_tex, dist_measure_strings, "recall_texture_plot", "recall_texture_plot")
    plot_accuracy_bar_graph(f1_values_array_tex, dist_measure_strings, "f1_texture_plot", "f1_texture_plot")
    
    plt.show()
    
                    
if __name__ == '__main__':
=======
"""
Module: main_TSC_LS.ipynb
Author: Dema N. Govalla
Date: December 4, 2023
Description: The file trains and test the combined_sensorData.csv file using the TSC-LS algorithm. 
            After traning and testing, it returns the algorithms metrics such as accuracy, presision and more.
            The file performs cross validation for different TSC-LS parameters and returns the classification
            report. 
"""

'''
Start of Cross validation results
'''
import numpy as np, pandas as pd, random, sys, seaborn as sns, os, statistics, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from torch import nn, optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.multiclass import unique_labels
from TSC_LS import LearningShapelets
from time import time
import warnings, matplotlib.cm as cm
import matplotlib, shutil
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
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

def plot_accuracy_bar_graph(values_array, dist_measure_strings, plot_title, save_filename):
    directory_path_result = 'TSC_LS_result'
    if not os.path.exists(directory_path_result):
        os.makedirs(directory_path_result)
        
    num_gammas = len(values_array[0])
    num_thetas = len(values_array)

    bar_width = 0.06
    index = np.arange(num_gammas)
    colors = plt.cm.get_cmap('tab10', num_thetas)
    
    plt.figure(figsize=(10, 6))
    for i in range(num_thetas):
        bars = plt.bar(index + i * bar_width, values_array[i], width=bar_width, label=f'Dist_measure = {dist_measure_strings[i]}', color=colors(i))
        for bar, acc in zip(bars, values_array[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{acc:.2f}%', ha='center', va='bottom')

    plt.xlabel('shapelets length')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.xticks(index + (bar_width * (num_thetas - 1)) / 2, [f'shap_len {i+1}' for i in range(num_gammas)])  
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(directory_path_result+ '/' + save_filename)  

def normalize_standard(X, scaler=None):
    shape = X.shape
    data_flat = X.flatten()
    if scaler is None:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    else:
        data_transformed = scaler.transform(data_flat.reshape(np.prod(shape), 1)).reshape(shape)
    return data_transformed, scaler

def normalize_data(X, scaler=None):
    if scaler is None:
        X, scaler = normalize_standard(X)
    else:
        X, scaler = normalize_standard(X, scaler)
    return X, scaler

def sample_ts_segments(X, shapelets_size, n_segments=10000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments

def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=10000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

def eval_accuracy(model, X, Y):
    result = []
    predictions = model.predict(X)
    if len(predictions.shape) == 2:
        predictions = predictions.argmax(axis=1)
        print(predictions)
    print(type(predictions))
    print(predictions.shape)

    result.append(predictions)
    print(f"Accuracy: {(predictions == Y).sum() / Y.size}")
    return result

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

def main():
    
    ''' Initiat loop counter '''
    startTime4 = time() 
    dist_measure_loop = 0
    shapelets_len_loop = 0
    fold_no_loop = 0
    
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
 
    sensor_data = pd.read_csv('combined_sensorData.csv')
    sensor_data = sensor_data.iloc[:,0:]

   # separate the independent and dependent features
    X_def = sensor_data.iloc[:, np.r_[0:4, 8]]
    y_def = sensor_data.iloc[:, sensor_data.shape[1]-1]
    
    # separate the independent and dependent features
    X_tex = sensor_data.iloc[:, np.r_[2:9]] 
    y_tex = sensor_data.iloc[:, sensor_data.shape[1]-1]

    directory_path_param = 'cross_val_TSC_LS_param'
    if os.path.exists(directory_path_param):
        shutil.rmtree(directory_path_param)
        os.makedirs(directory_path_param)
    else:
        os.makedirs(directory_path_param)

    n_splits=2
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
    for path in os.listdir(directory_path_param):
        if os.path.isfile(os.path.join(directory_path_param, path)):
            count += 1
    print('File count:', count)

   
    dist_measure_strings = ["euclidean", "cosine"]  
    shapelets_len_values = [2, 3]  
    
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
    
    for dist_measure in dist_measure_strings:
        
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
        
        for shapelets_len in shapelets_len_values:
            
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

                '''
                Train the deformation data
                '''
                newtrain_def = pd.read_csv(os.path.join(directory_path_param, 'train_fold_def_' + str(fold_no) + '.csv'))
                columns_to_combine_newtrain_def = newtrain_def.columns[1:-1]
                combined_array_newtrain_def = []
                
                for column in columns_to_combine_newtrain_def:
                    new_array_def = newtrain_def[column].to_numpy()
                    combined_array_newtrain_def.append(new_array_def)

                combined_array_newtrain_def = np.column_stack(combined_array_newtrain_def)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_train_def = np.array([label_map[label] for label in newtrain_def.iloc[:,-1].to_numpy()])
                new_X_train_def = combined_array_newtrain_def.reshape(new_y_train_def.size, columns_to_combine_newtrain_def.size,1)
                new_X_train_def, scaler_def = normalize_data(new_X_train_def)
                
            
                n_ts_def, n_channels_def, len_ts_def = new_X_train_def.shape
                loss_func_def = nn.CrossEntropyLoss()
                num_classes_def = len(set(new_y_train_def))
                # learn 2 shapelets of length 130
                shapelets_size_and_len_def = {1: shapelets_len}
                # dist_measure_def = "euclidean"
                dist_measure_def = dist_measure
                
                lr_def = 1e-2
                wd_def = 1e-3
                epsilon_def = 1e-7
            
                learning_shapelets_def = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_def,
                                                    in_channels=n_channels_def,
                                                    num_classes=num_classes_def,
                                                    loss_func=loss_func_def,
                                                    to_cuda=False,
                                                    verbose=1,
                                                    dist_measure=dist_measure_def)

                for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_def.items()):
                    weights_block = get_weights_via_kmeans(new_X_train_def, shapelets_size, num_shapelets)
                    learning_shapelets_def.set_shapelet_weights_of_block(i, weights_block)
            
                optimizer = optim.Adam(learning_shapelets_def.model.parameters(), lr=lr_def, weight_decay=wd_def, eps=epsilon_def)
                learning_shapelets_def.set_optimizer(optimizer)
                losses = learning_shapelets_def.fit(new_X_train_def, new_y_train_def, epochs=2000, batch_size=256, shuffle=False, drop_last=False)
                print("Training is Done")
                
                
                '''
                Train the texture data
                '''
                # Used for training the model
                newtrain_tex = pd.read_csv(os.path.join(directory_path_param, 'train_fold_tex_' + str(fold_no) + '.csv'))
                columns_to_combine_newtrain_tex = newtrain_tex.columns[1:-1]
                combined_array_newtrain_tex = []

                for column in columns_to_combine_newtrain_tex:
                    new_array_tex = newtrain_tex[column].to_numpy()
                    combined_array_newtrain_tex.append(new_array_tex)

                combined_array_newtrain_tex = np.column_stack(combined_array_newtrain_tex)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_train_tex = np.array([label_map[label] for label in newtrain_tex.iloc[:,-1].to_numpy()])
                new_X_train_tex = combined_array_newtrain_tex.reshape(new_y_train_tex.size, columns_to_combine_newtrain_tex.size,1)
                new_X_train_tex, scaler_tex = normalize_data(new_X_train_tex)
                
            
                n_ts_tex, n_channels_tex, len_ts_tex = new_X_train_tex.shape
                loss_func_tex = nn.CrossEntropyLoss()
                num_classes_tex = len(set(new_y_train_tex))
                # learn 2 shapelets of length 130
                shapelets_size_and_len_tex = {1: shapelets_len}
                # dist_measure_tex = "euclidean"
                # dist_measure_tex = "cross-correlation"
                dist_measure_tex = dist_measure
                
                
                lr_tex = 1e-2
                wd_tex = 1e-3
                epsilon_tex = 1e-7
                learning_shapelets_tex = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len_tex,
                                                    in_channels=n_channels_tex,
                                                    num_classes=num_classes_tex,
                                                    loss_func=loss_func_tex,
                                                    to_cuda=False,
                                                    verbose=1,
                                                    dist_measure=dist_measure_tex)
                
                for i, (shapelets_size, num_shapelets) in enumerate(shapelets_size_and_len_tex.items()):
                    weights_block = get_weights_via_kmeans(new_X_train_tex, shapelets_size, num_shapelets)
                    learning_shapelets_tex.set_shapelet_weights_of_block(i, weights_block)
            
                optimizer = optim.Adam(learning_shapelets_tex.model.parameters(), lr=lr_tex, weight_decay=wd_tex, eps=epsilon_tex)
                learning_shapelets_tex.set_optimizer(optimizer)

                losses = learning_shapelets_tex.fit(new_X_train_tex, new_y_train_tex, epochs=2000, batch_size=256, shuffle=False, drop_last=False)
                print("Training is Done")
                  
                '''
                Test the deformation data
                '''
                newval_def = pd.read_csv(os.path.join(directory_path_param, 'val_fold_def_' + str(fold_no) + '.csv'))
                columns_to_combine_newval_def = newval_def.columns[1:-1]
                combined_array_newval_def = []
                for column in columns_to_combine_newval_def:
                    new_array_def = newval_def[column].to_numpy()
                    combined_array_newval_def.append(new_array_def)
                combined_array_newval_def = np.column_stack(combined_array_newval_def)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_val_def = np.array([label_map[label] for label in newval_def.iloc[:,-1].to_numpy()])
                new_X_val_def = combined_array_newval_def.reshape(new_y_val_def.size,columns_to_combine_newval_def.size,1)
                new_X_val_def, scaler = normalize_data(new_X_val_def)

                y_predlr_val_def = np.array(eval_accuracy(learning_shapelets_def,new_X_val_def,new_y_val_def)).reshape(-1)

                '''
                Test the texture data
                '''                          
                newval_tex = pd.read_csv(os.path.join(directory_path_param, 'val_fold_tex_' + str(fold_no) + '.csv'))
                columns_to_combine_newval_tex = newval_tex.columns[1:-1]
                combined_array_newval_tex = []
                for column in columns_to_combine_newval_tex:
                    new_array_tex = newval_tex[column].to_numpy()
                    combined_array_newval_tex.append(new_array_tex)
                combined_array_newval_tex = np.column_stack(combined_array_newval_tex)
                label_map = {1: 0, 2: 1, 3: 2, 4: 3}

                new_y_val_tex = np.array([label_map[label] for label in newval_tex.iloc[:,-1].to_numpy()])
                new_X_val_tex = combined_array_newval_tex.reshape(new_y_val_tex.size,columns_to_combine_newval_tex.size,1)
                new_X_val_tex, scaler = normalize_data(new_X_val_tex)
                
                y_predlr_val_tex = np.array(eval_accuracy(learning_shapelets_tex,new_X_val_tex,new_y_val_tex)).reshape(-1)
           
                '''
                Metrics For Deformation
                '''   
                mae_def, mse_def, rmse_def, r2_def, accuracy_def, precision_def, recall_def, f1_def, conf_matrix_def, class_report_def = calculate_metrics(new_y_val_def, y_predlr_val_def)
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
                
                '''
                Metrics For Texture
                '''
                mae_tex, mse_tex, rmse_tex, r2_tex, accuracy_tex, precision_tex, recall_tex, f1_tex, conf_matrix_tex, class_report_tex = calculate_metrics(new_y_val_tex, y_predlr_val_tex)
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
        
        
            ''' Iterate fold_no_loop'''
            dist_measure_loop += 1
            print("dist_measure loop is at dist_measure_loop = ", dist_measure_loop)
            
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
    
        ''' Iterate fold_no_loop'''
        shapelets_len_loop += 1
        print("shapelets_len loop is at shapelets_len_loop = ", shapelets_len_loop)
        
                              
    
        # end_time3 = time()
        # totalTime3 = convert_seconds_to_minutes(end_time3 - startTime3)
        # print("Total time 3 is: ", totalTime3)                    
        
        
    '''
    Print K-fold for deformation
    '''
    print("These are the def mae for each k-fold split", k_fold_mae_def, "\n")
    print("These are the def mse for each k-fold split", k_fold_mse_def, "\n")
    print("These are the def rmse for each k-fold split", k_fold_rmse_def, "\n")
    print("These are the def r2 for each k-fold split", k_fold_r2_def, "\n")
    
    print("These are the def accuracy for each k-fold split", k_fold_accuracy_def, "\n")
    print("These are the def precision for each k-fold split", k_fold_precision_def, "\n")
    print("These are the def recall for each k-fold split", k_fold_recall_def, "\n")
    print("These are the def f1 for each k-fold split", k_fold_f1_def, "\n")

    '''
    Print K-fold for texture
    '''
    print("These are the tex mae for each k-fold split", k_fold_mae_tex, "\n")
    print("These are the tex mse for each k-fold split", k_fold_mse_tex, "\n")
    print("These are the tex rmse for each k-fold split", k_fold_rmse_tex, "\n")
    print("These are the tex r2 for each k-fold split", k_fold_r2_tex, "\n")
    
    print("These are the tex accuracy for each k-fold split", k_fold_accuracy_tex, "\n")
    print("These are the tex precision for each k-fold split", k_fold_precision_tex, "\n")
    print("These are the tex recall for each k-fold split", k_fold_recall_tex, "\n")
    print("These are the tex f1 for each k-fold split", k_fold_f1_tex, "\n")
    
    end_time4 = time()
    totalTime4 = convert_seconds_to_minutes(end_time4 - startTime4)
    print("Total completion time is: ", totalTime4)
    
    
    '''
    Plot for deformation
    '''
    plot_accuracy_bar_graph(mae_values_array_def, dist_measure_strings, "mae_deformation_plot", "mae_deformation_plot")
    plot_accuracy_bar_graph(mse_values_array_def, dist_measure_strings, "mse_deformation_plot", "mse_deformation_plot")
    plot_accuracy_bar_graph(rmse_values_array_def, dist_measure_strings, "rmse_deformation_plot", "rmse_deformation_plot")
    plot_accuracy_bar_graph(r2_values_array_def, dist_measure_strings, "r2_deformation_plot", "r2_deformation_plot")
    
    plot_accuracy_bar_graph(accuracy_values_array_def, dist_measure_strings, "accuracy_deformation_plot", "accuracy_deformation_plot")
    plot_accuracy_bar_graph(precision_values_array_def, dist_measure_strings, "precision_deformation_plot", "precision_deformation_plot")
    plot_accuracy_bar_graph(recall_values_array_def, dist_measure_strings, "recall_deformation_plot", "recall_deformation_plot")
    plot_accuracy_bar_graph(f1_values_array_def, dist_measure_strings, "f1_deformation_plot", "f1_deformation_plot")
    
    '''
    Plot for texture
    '''
    plot_accuracy_bar_graph(mae_values_array_tex, dist_measure_strings, "mae_texture_plot", "mae_texture_plot")
    plot_accuracy_bar_graph(mse_values_array_tex, dist_measure_strings, "mse_texture_plot", "mse_texture_plot")
    plot_accuracy_bar_graph(rmse_values_array_tex, dist_measure_strings, "rmse_texture_plot", "rmse_texture_plot")
    plot_accuracy_bar_graph(r2_values_array_tex, dist_measure_strings, "r2_texture_plot", "r2_texture_plot")
    
    plot_accuracy_bar_graph(accuracy_values_array_tex, dist_measure_strings, "accuracy_texture_plot", "accuracy_texture_plot")
    plot_accuracy_bar_graph(precision_values_array_tex, dist_measure_strings, "precision_texture_plot", "precision_texture_plot")
    plot_accuracy_bar_graph(recall_values_array_tex, dist_measure_strings, "recall_texture_plot", "recall_texture_plot")
    plot_accuracy_bar_graph(f1_values_array_tex, dist_measure_strings, "f1_texture_plot", "f1_texture_plot")
    
    plt.show()
    
                    
if __name__ == '__main__':
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
    main()