import torch
import pandas as pd
import numpy as np
import json
import os

import argparse
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import sys
sys.path.append('..\data_loader')
sys.path.append('..\ecg_respiration_sleep_staging-master')
from tqdm import tqdm
import random
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight



random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
from torch import optim

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def calculate_metrics(y_true,y_pred):
    metric_dict = {}
    metric_dict["accuracy"] = accuracy_score(y_true,y_pred)    
    metric_dict["f1_score"] = f1_score(y_true,y_pred,average="weighted")    
    metric_dict["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    return metric_dict    
    
def gatherBatchDataToSubjectMetrics(batch_dict):
    temp_subject = []
    temp_prediction = []
    temp_labels = []

    temp_subject = [item for sublist in batch_dict['subject'] for item in sublist]
    temp_prediction = [item for sublist in batch_dict['prediction'] for item in sublist]
    temp_labels = [item for sublist in batch_dict['labels'] for item in sublist]    
    df = pd.DataFrame({"subject":temp_subject,
                       "predicted":temp_prediction,
                       "labels":temp_labels})
    #print(df)
        
    subjects = df["subject"].unique()
    print
    subject_metric_dict = {}
    for subject in subjects:
        y_pred = df["predicted"][df["subject"]==subject]
        y_true = df["labels"][df["subject"]==subject]        
        subject_metric_dict[subject] = calculate_metrics(y_true,y_pred)
    return subject_metric_dict, df   

def getDataFrameOfRecordingMetric(metric_list:list,metric:str='accuracy'):
    recordings = list(metric_list[0].keys())
    d =  {k: [] for k in recordings}
    #print(d)
    for recordings_in_epoch in metric_list:
            for recording in recordings_in_epoch:         
                #print(recordings_in_epoch[recording])
                d[recording].append(recordings_in_epoch[recording][metric])
    return pd.DataFrame(d)

def getMetricPerEpoch(metric_list:list,metric_name:str=''):
    """
    Calculates the mean value of the metrics for each epoch. The input is a list of nested dicts in the structe of [{"recordingX":{"metric1":[] ,"metric2":[]...}} , ...]

    Parameters
    ----------
    metric_list : list
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if metric_name == '':
        metrics_names = list(metric_list[0][list(metric_list[0].keys())[0]].keys()) # unpack the names for the metrics
    else:
        metrics_names = [metric_name]
        
    mean_metric_per_epoch = {key: [] for key in metrics_names}
    #num_of_epochs = len(metric_list)
    for recordings_in_epoch in metric_list:
        for metric in metrics_names:
            metric_value = 0
            for recording in recordings_in_epoch:        
                metric_value = metric_value + recordings_in_epoch[recording][metric]
            mean_metric_per_epoch[metric].append(metric_value/len(recordings_in_epoch))
            
    return mean_metric_per_epoch        

    recordings = list(metric_list[0].keys())
    d =  {k: [] for k in recordings}
    #print(d)
    for recordings_in_epoch in metric_list:
            for recording in recordings_in_epoch:         
                #print(recordings_in_epoch[recording])
                d[recording].append(recordings_in_epoch[recording][metric])
    return pd.DataFrame(d)

def getValidationLabelsAndPredictions(validation_set, model,batch_size = 16,model_name = ''):
    y_pred = []
    y_true = []
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=batch_size)
    # iterate over test data
    for inputs, labels, val_file_name in validation_generator:
            if torch.cuda.is_available():   
                #inputs, labels = inputs.to(device), labels.to(device)   
                inputs, labels = inputs.to('cuda'), labels.to('cuda')  
            inputs = inputs.squeeze(0)
            labels = labels.squeeze()        
            if model_name == 'ECGSleepNet':
                forward_pass_outputs,H_ = model(inputs)
            else:
                forward_pass_outputs = model(inputs)
            #forward_pass_outputs,H_ = model(inputs)
            _, predicted = forward_pass_outputs.max(1)
            #predicted_batch.append(predicted)     
            #output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(predicted.data.cpu().numpy()) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth    
    return y_true, y_pred

def setCWDToScriptLocation():
    pathname = os.path.dirname(sys.argv[0])
    if pathname == "":
        print("No filename")
        return
    os.chdir(pathname)
    print("Current working directory set to: ",os.getcwd())    
    return


def get_train_valid_test_files_recreate(root, file_json_path):
    ff = open(file_json_path) 
    files = json.load(ff)
    ff.close()        
    tfiles = files['training']
    tfiles = [x.split('\\')[-1] for x in tfiles]
    tfiles = [os.path.join(root,x) for x in tfiles]
    vfiles = files['validation']
    vfiles = [x.split('\\')[-1] for x in vfiles]
    vfiles = [os.path.join(root,x) for x in vfiles]    
    testfiles = files['testing']
    testfiles = [x.split('\\')[-1] for x in testfiles]
    testfiles = [os.path.join(root,x) for x in testfiles] 
    test_fileNames = testfiles
    training_fileNames = tfiles
    validation_fileNames = vfiles    
    return training_fileNames, validation_fileNames, test_fileNames

def get_train_valid_test_files_recreate_small(root, file_json_path, train_number = 20):
    training_fileNames, validation_fileNames, test_fileNames = get_train_valid_test_files_recreate(root, file_json_path)
    training_fileNames = training_fileNames[0:train_number]
    return training_fileNames, validation_fileNames, test_fileNames


    

def get_optimzer_and_scheduler(model,lr = 0.001,decay=0):
    optimizer = optim.RMSprop(filter(lambda x:x.requires_grad, model.parameters()), lr=lr,weight_decay=decay)#, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=2)   
    return optimizer, scheduler


def getCrossEntropyWeights(dataset,num_of_classes = 5):
    if (type(dataset) == torch.utils.data.dataset.ConcatDataset) or (type(dataset) == torch.utils.data.dataset.ChainDataset):
        all_labels = []
        for recording in dataset.datasets:
            all_labels.append(recording.labels.squeeze().data.cpu().numpy())
        
        all_labels = [item for sublist in all_labels for item in sublist]
        all_labels = np.array(all_labels)
        label_counter = Counter(all_labels)
        weights =  []
        for i in np.arange(0,num_of_classes):
            weights.append(1./label_counter[i])
        weights = np.array(weights)
        weights = weights/np.mean(weights)
        return weights, label_counter
    else:
        #raise Exception("Not type torch.utils.data.dataset.ConcatDataset")
        list_of_labels = []
        generator = torch.utils.data.DataLoader(dataset,batch_size=6)
    
        for x, y, file_name in generator:
            if torch.cuda.is_available():   
                x, y = x.to('cuda'), y.to('cuda')
    
            x = x.squeeze(0)
            #y = y.squeeze()
            if  y.size() != torch.Size([1]):
                y =  y.squeeze()
            list_of_labels.append(y.cpu().numpy())
        flat_list = [item for sublist in list_of_labels for item in sublist]
        all_labels = np.array(flat_list)
        label_counter = Counter(all_labels)
        weights =  []
        for i in np.arange(0,num_of_classes):
            weights.append(1./label_counter[i])
        weights = np.array(weights)
        weights = weights/np.mean(weights)
        return weights, label_counter
        #labels = np.array(flat_list)
        #weights = class_weight.compute_class_weight(class_weight ="balanced", classes= np.unique(labels),y=labels)
        #return weights, None
        
# Old create dataset
def create_datasets(files:list,dataloader,**kwargs):
    dataset_list = []
    for file in tqdm(files,desc = 'Creating dataset using '+str(dataloader.__name__),total=len(files)):
        try:
            dataset_list.append(dataloader(file,**kwargs))
        except Exception as e:
            print("\nFailed to load:",file)
            print("Original error:", str(e))
    return torch.utils.data.ConcatDataset(dataset_list)
"""
def create_datasets(files:list,dataloader,**kwargs):
    #print(files[0])
    if str(dataloader.__name__) == 'ECGPeakMask3':
        print("Using iterable-style dataset")
        #d = torch.utils.data.ChainDataset([dataloader(files[0],**kwargs)])
        # This is a generator function that creates instances of your Dataloader
        # as they are requested
        def dataset_generator():
            for file in files:
                yield dataloader(file, **kwargs)
        
        # We pass the generator (not the generator function) to ChainDataset
        d = torch.utils.data.ChainDataset(dataset_generator())     
        return d
    else:
        d = torch.utils.data.ConcatDataset([dataloader(files[0],**kwargs)])
    
        if len(files)>1:
            for file in tqdm(files[1:],desc = 'Creating dataset using '+str(dataloader.__name__)):
                d.datasets.append(dataloader(file,**kwargs))
        return d   

"""

def getSubjectPaths(data_type='rawecg',USING_CLAAUDIA = False,train_split = 0.2, test_split = 0.5):
    ALREADY_PROCESSED = False
    def ecgmask_path(USING_CLAAUDIA=False):
        if USING_CLAAUDIA == False:
            all_files = glob.glob(os.path.join('H:\processed_mesa_ecg_mask',"*"))
            if len(all_files) ==0:
                all_files = glob.glob(os.path.join(r'C:\Users\Shagen\OneDrive - Aalborg Universitet\Dokumenter\PhD\data_loader\ecg_mask',"*"))
        else:
            all_files = glob.glob(os.path.join('/home/cs.aau.dk/fq73oo/data_loader',"sleep_data","processed_mesa_ecg_mask","*"))
        return all_files
    
    def rawecg_path(USING_CLAAUDIA=False):
        if USING_CLAAUDIA == False:
            all_files = glob.glob(r"H:\processed_mesa\mesa\*")
            if len(all_files) ==0:
                all_files = glob.glob(r'C:\Users\Shagen\OneDrive - Aalborg Universitet\Dokumenter\PhD\data_loader\non-augmented\processed_data_train\mesa\*')
        else:
            all_files = glob.glob(r"/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa/mesa/*")
        return all_files
    
    def ibi_path(USING_CLAAUDIA=False):
        if USING_CLAAUDIA == False:
            all_files = os.path.join('H:\processed_mesa_ibi',"*")
        else:
            all_files = os.path.join(r"/home/cs.aau.dk/fq73oo/data_loader/","sleep_data","processed_mesa_ibi","*")
        return all_files
    
    def recreate_path(USING_CLAAUDIA=False):
        if USING_CLAAUDIA == False:
            file_dict_path = r'F:\OneDrive - Aalborg Universitet\Dokumenter\PhD\ecg_respiration_sleep_staging-master\training_code_still_messy\my_models\training_files.json'
            root_of_mask_files = r'H:\processed_mesa_ecg_mask\\'         
            training_fileNames, validation_fileNames, test_fileNames = get_train_valid_test_files_recreate(root_of_mask_files, file_dict_path)
        else:
            file_dict_path =  os.path.join('/home/cs.aau.dk/fq73oo/data_loader',"sleep_data","training_files.json")
            root_of_mask_files = os.path.join('/home/cs.aau.dk/fq73oo/data_loader',"sleep_data","processed_mesa_ecg_mask")
            training_fileNames, validation_fileNames, test_fileNames = get_train_valid_test_files_recreate(root_of_mask_files, file_dict_path)
        return training_fileNames, validation_fileNames, test_fileNames    
    
    if data_type == 'ecgmask':
        all_files = ecgmask_path(USING_CLAAUDIA)
    elif data_type == 'rawecg':
        all_files = rawecg_path(USING_CLAAUDIA)
    elif data_type == 'ibi':
        ALREADY_PROCESSED = True
        all_files = ibi_path(USING_CLAAUDIA)
    elif data_type == 'recreate':
        training_fileNames, validation_fileNames, test_fileNames = recreate_path(USING_CLAAUDIA)
        ALREADY_PROCESSED = True
    
    if 'all_files' in locals():
        print("Splitting into train, test and valid with ratio:", train_split," for train and ", test_split)
        training_fileNames ,val_test_files = train_test_split(all_files,test_size=train_split)
        test_fileNames ,validation_fileNames = train_test_split(val_test_files,test_size=test_split)
    if len(training_fileNames) == 0:
        raise Exception("No training files found")
    if len(validation_fileNames) == 0:
        raise Exception("No validation files found")
    if len(test_fileNames) == 0:
        raise Exception("No test files found")
    return training_fileNames, validation_fileNames, test_fileNames, ALREADY_PROCESSED


def getSubjectPathsContinuedTraining(resume_training_folder_name:str,root_folder = r"H:\processed_mesa\mesa"):
        # Opening JSON file
        f = open(os.path.join(resume_training_folder_name,'training_files.json'))
          
        # returns JSON object as 
        # a dictionary
        def __fix_path_to_local(root,files):
            d = [os.path.join(root,f.split("/")[-1]) for f in files]
            return d
        #root_folder = 
        files_names = json.load(f)
        #training_fileNames = __fix_path_to_local(root_folder,files_names['training']) # Used for switching between local machine cloud computing
        #test_fileNames = __fix_path_to_local(root_folder,files_names['testing'])
        #validation_fileNames = __fix_path_to_local(root_folder,files_names['validation'])
        training_fileNames = files_names['training']
        test_fileNames = files_names['testing']
        validation_fileNames = files_names['validation']
        f.close()    
        return training_fileNames, validation_fileNames, test_fileNames
    
def calculateBaselinePrClass(target,label_mapping):        
    baseline_dict = {}
    for key, item in label_mapping.items():
        baseline_dict['baseline_'+str(key)] = calculate_metrics(target,[item]*len(target))
    return baseline_dict

def createBaselineAndPerformanceDf(metric_per_epoch:dict,baselines:dict,model_name:str):
    model_performance = pd.DataFrame(metric_per_epoch)
    def extract_last_element(lst):
        return lst[-1]
    
    if type(model_performance['training'][-1])==list:
        model_performance = model_performance.applymap(extract_last_element) # Get last epoch of the dataset
    model_performance['model'] = model_name
    model_performance = model_performance.reset_index().rename(columns={'index': 'metric'})
    
    base_train = pd.DataFrame(baselines['training'])
    base_train = base_train.stack().reset_index()
    base_train.columns = ['metric', 'model', 'training']
    
    base_test = pd.DataFrame(baselines['test'])
    base_test = base_test.stack().reset_index()
    base_test.columns = ['metric', 'model', 'test']

    base_validation = pd.DataFrame(baselines['validation'])
    base_validation = base_validation.stack().reset_index()
    base_validation.columns = ['metric', 'model', 'validation']
    
    _merged_df = pd.merge(base_train, base_test, on=['metric', 'model'])
    merged_df = pd.merge(_merged_df, base_validation, on=['metric', 'model'])
    
    model_and_base_performance = pd.concat([merged_df, model_performance], ignore_index=True)
    
    df_long = pd.melt(model_and_base_performance, id_vars=['metric', 'model'], var_name='dataset', value_name='value')
    return df_long
    

def getAAUWSSDLECGPPGPathPairs(aauwss_root_folder):
    ecg_folder = os.path.join(aauwss_root_folder,'ecg',"*")
    ppg_folder = os.path.join(aauwss_root_folder,'ppg',"*")
    ecg_paths = glob.glob(ecg_folder)
    ppg_paths = glob.glob(ppg_folder)
    ecg_dict = {os.path.basename(file).split('_')[1]: file for file in ecg_paths}
    ppg_dict = {os.path.basename(file).split('_')[1]: file for file in ppg_paths}
    
    # Find common subjects and pair them
    paired_files = []
    for subject in ecg_dict.keys():
        if subject in ppg_dict:
            paired_files.append((subject, ecg_dict[subject], ppg_dict[subject]))        
    return paired_files