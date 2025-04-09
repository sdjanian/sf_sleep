import sys
sys.path
sys.path.append('./code')
sys.path.append('./model_library')
sys.path.append('./preprocessing')

import random
import torch
from torch import nn
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import os
import glob
import time
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim

from deep_learning_loop_utilities import str2bool 
from deep_learning_loop_utilities import get_train_valid_test_files_recreate,get_train_valid_test_files_recreate_small,getSubjectPaths, getSubjectPathsContinuedTraining, getAAUWSSDLECGPPGPathPairs
from deep_learning_loop_utilities import create_datasets, getCrossEntropyWeights, get_optimzer_and_scheduler
from deep_learning_plots import create_loss_plot, create_confusion_matrix, create_hypnograms,getDistributionOfGeneratorLabels,DistributionPlot, MetricPlots, PlotMetricDistribution, PlotClassAccuracies, createBaselineComparisonPlot
from deep_learning_loop_utilities import calculate_metrics, gatherBatchDataToSubjectMetrics, getDataFrameOfRecordingMetric, getMetricPerEpoch
from deep_learning_loop_utilities import calculateBaselinePrClass,createBaselineAndPerformanceDf


from dataset_densenet import ECGDataSetSingle2
from ecg_mask_loader_simple import ECGPeakMask2
from data_loader_aauwss_sleep_study_dataset import AAUWSSDL
from dense_net_utils import setCWDToScriptLocation, experiment_folder_path, get_weights 
from resnet1d import ResNet1D
from dense_net_model import create_DenseNetmodel
from fcn import FCN
from mymodel import ECGSleepNet
from ECGSleepNetAdaptable import ECGSleepNetAdaptable
from simple_cnn import Simple1DCNN
from deep_sleep_bcg_model import DeepSleep
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
plt.style.use('seaborn')

import shutil
#Helper functions
##############################################################################################################



    

def run_one_epoch(model, optimizer, scheduler, loss_fn, data_loader,train = True,verboose=0):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.  # Initialize total loss for the epoch
    if train==True:
        description ="Training model"
    else:
        description = "Evaluation model"

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    prediction_batch_dict = {'prediction':[],'labels':[],'subject':[]}
    for i, data in tqdm(enumerate(data_loader),position=0,leave=True,desc=description):#,total =len(data_loader),desc=description):
        # Every data instance is an input + label pair
        inputs, labels,filename = data
        # If the size is already 1, squeezing wil ruin the dimensions
        if labels.size() != torch.Size([1]):
            labels = labels.squeeze()
        #labels = labels.squeeze() # Old squeeze, commented out
        if torch.cuda.is_available():
            labels = labels.to('cuda')
            inputs = inputs.to('cuda')
        if train==True:
            # Zero your gradients for every batch!
            optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _h = outputs # unfold the tuple to contain the prediction and the last layer before
        #outputs, _h  = model(inputs)

        # Compute the loss and its gradients
        #print(labels.min().cpu().detach().numpy(),labels.max().cpu().detach().numpy(),filename[0])
        loss = loss_fn(outputs, labels)
        if train==True:
            loss.backward()
    
            # Adjust learning weights
            if optimizer != None:
                optimizer.step()
            #print(i)
        # Accumulate the total loss for the entire epoch
        total_loss += loss.item()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            if verboose == 1:
                print('  batch {} loss: {}'.format(i + 1, last_loss))

            running_loss = 0.
        predicted_labels = outputs.max(1).indices
        #print(predicted_labels)
        prediction_batch_dict['prediction'].append(predicted_labels.cpu().detach().numpy().tolist())
        prediction_batch_dict['labels'].append(labels.cpu().detach().numpy().tolist())
        prediction_batch_dict['subject'].append(filename)
        
    average_epoch_loss = total_loss / len(data_loader)    
    if train == False and scheduler!=None:
        scheduler.step(average_epoch_loss)        
    return average_epoch_loss, prediction_batch_dict

    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    setCWDToScriptLocation()
    current_dir = os.getcwd()
    #bleh = create_datasets(['H:\\processed_mesa\\mesa\\mesa-sleep-0789.pkl'],ECGDataSetSingle,shuffle_recording=False, number_of_sleep_stage=5, normalize=True, augment_data = True)

    # Parses the commandline arguments for the various configurations and stores them
    ##############################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-ss', '--sleep_stages',type=int,default=2,help='Number of sleep stages for classification. From 2-5. Defaults to 5.',dest='number_of_sleep_stages')
    parser.add_argument('-e', '--epochs',type=int,default=5,help='Number of epochs to train',dest='n_epochs')    
    parser.add_argument('-bs', '--batch_size',type=int,default=6,help='Batch size',dest='batch_size')    
    parser.add_argument('-CAAU', '--USING_CLAAUDIA',type=str2bool,default=False,help='Loading data from Claaudia AI Cloud. Only use if connected to Claaudia.',dest='USING_CLAAUDIA')    
    parser.add_argument('-m', '--model',type=str,default='fcn',choices=['fcn','dnet','simplecnn','rnet','rnext','DeepSleep','ECGSleepNet','ECGSleepNetAdaptable'],help='Which model to use. Choose between: \n rnet = ResNet \n rnext = ResNext \n dnet = DenseNet \n fcn = Fully Connected CNN',
                        dest='model_type')    
    parser.add_argument('-wloss', '--weighted_loss',type=str2bool,default=False,help='Use a weighted loss function',dest='use_weighted_loss')    
    parser.add_argument('-wsample', '--weighted_random_samples',type=str2bool,default=False,help='Use a weighted random sampling',dest='use_weighted_sampler')    
    parser.add_argument('-fname', '--folder_name_prefix',type=str,default='',help='String to add before the output folder of the experiment',
                        dest='folder_name_prefix')          
    parser.add_argument('-rtrain', '--resume_training',type=str2bool,default=False,help='Resume training of a previous model',dest='resume_training')    
    parser.add_argument('-rname', '--resume_folder_name',type=str,default="",help='Name of the folder that training is to be resumed in',
                        dest='resume_folder_name')   
    parser.add_argument('-augment', '--augment_training_data',type=str2bool,default=False,help='Perform augmentation of training data',dest='augment')    
    parser.add_argument('-scheduler', '--user_scheduler',type=str2bool,default=False,help='Use the training scheduler',dest='use_scheduler')    
    parser.add_argument('-save_i_models', '--save_intermediate_models',type=str2bool,default=False,help='Save models after every epoch',dest='save_intermediate_models')    
    parser.add_argument('-features', '--use_features',type=str2bool,default=False,help='Use HRV features instead of signal',dest='use_features')    
    parser.add_argument('-kernel_size', '--kernel_size',type=int,default=0,help='Size of the initial kernel. The default is 0, which is whatever is the default of the model',dest='kernel_size')    
    parser.add_argument('-ibi', '--use_ibi',type=str2bool,default=False,help='Use Interbeat Interval as input',dest='use_IBI')    
    parser.add_argument('-loss', '--loss_type',type=str,default='crossentropy',choices = ['crossentropy','focal'],help='Which loss function to use to use. Choose between: \n crossentropy = Categorical Cross Entropy \n focal = Focal Loss',dest='loss_type') 
    parser.add_argument('-lr', '--learning_rate',type=float,default=0.001,help='The learning rate of the optimizer',dest='learning_rate')  
    parser.add_argument('-dtype', '--data_type',type=str,default='rawecg',choices = ['rawecg','ecgmask'],help='Which input data to use. \necg = raw ECG signal \nfeatures = HRV features \nibi = Interbeat Interval Sequence \necgmask = A mask of ECG peaks ',dest='data_type') 
    parser.add_argument('-shuffle', '--shuffle_training',type=str2bool,default=False,help='Shuffle the training data. Validation and test are not shuffled',dest='shuffle_bool')    
    parser.add_argument('-optim', '--optimizer',type=str,default='rmsprop',choices = ['rmsprop','adam'],help='Which optimizer to use. adam or rmsprop',dest='optimizer') 
    parser.add_argument('-dec', '--decay',type=float,default=0.0,help='The decay of the learning rate',dest='weight_decay')  
    parser.add_argument('-layers', '--number_of_layers',type=int,default=1,help='Number of layers in the CNN',dest='number_of_layers')    
    parser.add_argument('-norm_type', '--normalize_type',type=str,default='zscore',choices=["zscore","paper"],help='Choice of normalization for raw ecg data',dest='normalize_type')
    parser.add_argument('-norm', '--normalize_data',type=str2bool,default=True,help='Normalize input to neural networks',dest='normalize_data')
    parser.add_argument('-resample', '--resample_signal',type=str2bool,default=True,help='Resample input signal',dest='resample_signal')
    parser.add_argument('-resample_hz', '--resample_frequency',type=int,default=64,help='The new frequency of the input signal',dest='resample_frequency')
    parser.add_argument('-w_size', '--window_size',type=int,default=None,help='Changes the epoch size from 30 seconds to window_size',dest='window_size')
    parser.add_argument('-pat', '--patience',type=int,default=0,help='Amount of patience compared to validation set',dest='patience')
    parser.add_argument('-s_best', '--save_best_model',type=str2bool,default=True,help='Normalize input to neural networks',dest='save_best_model')
    parser.add_argument('-l_vs_rest', '-light_sleep_vs_all_bool',type=str2bool,default=False,help='Collapses N1 and N2 into Light Sleep and N3,REM and Wake into Not Light Sleep. Only works when sleep stages set to 2',dest='light_sleep_vs_all_bool')
    parser.add_argument('-dset', '--data_set',type=str,default='',choices = ['AAUWSS','example'],help='Which dataset to use. \nAAUWSS = The AAUWSS dataset loader that contain ECG and PPGs ',dest='data_set') 


    
    args = parser.parse_args()
    
    # args.number_of_sleep_stages=5
    # args.n_epochs = 2
    # args.USING_CLAAUDIA = False
    # args.data_type = 'rawecg'
    # args.model_type = 'ECGSleepNetAdaptable'#'ECGSleepNet' #fcn, dnt
    # #args.shuffle_bool =False# True
    # #args.augment = False
    # args.use_weighted_loss = False
    # #args.use_weighted_sampler = False
    # args.learning_rate = 0.001
    # args.optimizer = 'rmsprop'
    # args.weight_decay = 1e-6#0.000001
    # #args.number_of_layers = 1
    # #args.shuffle_bool=True
    # args.normalize_type = 'zscore'
    # args.resample_frequency = 64#200
    # args.resample_signal = True
    # args.light_sleep_vs_all_bool = False
    # args.use_scheduler = False
    # args.data_set = 'example'
    # args.window_size = 270


    # args.window_size = 270
    
    


    CONTINUE_TRAINING = args.resume_training
    
    if CONTINUE_TRAINING == True:
        resume_training_folder_name = args.resume_folder_name
        #del args
        parser2 = argparse.ArgumentParser()
        args2, unknown2 = parser2.parse_known_args()
        with open(os.path.join(resume_training_folder_name,'commandline_args.txt'), 'r') as f:
            args2.__dict__ = json.load(f)
            args2.patience = 0
        print(args.__dict__)
        for key, value in args.__dict__.items():
            #print(key,value)
            if key in args2.__dict__:
                args.__dict__[key] = args2.__dict__[key]
        #args.resume_training = True
        #args.resume_folder_name = resume_training_folder_name
    #args.USING_CLAAUDIA = False

    USE_SCHEDULER = args.use_scheduler
    NUM_CLASSES = args.number_of_sleep_stages
    n_epochs = args.n_epochs    
    batch_size = args.batch_size    
    model_type = args.model_type #'ECGSleepNet'
    use_weighted_loss = args.use_weighted_loss
    use_weighted_random_samlper = args.use_weighted_sampler
    folder_name_prefix = args.folder_name_prefix
    AUGMENT_DATA = args.augment        
    USING_CLAAUDIA = args.USING_CLAAUDIA
    save_intermediate_models = args.save_intermediate_models
    USE_FEATURES = args.use_features
    USE_IBI = args.use_IBI
    arg_kernel_size = args.kernel_size
    loss_type = args.loss_type
    learning_rate = args.learning_rate
    data_type = args.data_type #'ecgmask' 'recreate'#'rawecg'#'recreate'
    shuffle_bool = args.shuffle_bool
    #split_test_train = True
    ALREADY_PROCESSED = False
    optimizer_choice =args.optimizer
    decay = args.weight_decay
    number_of_layers = args.number_of_layers
    normalize_type = args.normalize_type
    normalize_data_bool = args.normalize_data
    resample_frequency = args.resample_frequency
    resample_data = args.resample_signal
    window_size = args.window_size
    patience = args.patience
    save_best_model_bool = args.save_best_model
    light_sleep_vs_all_bool = args.light_sleep_vs_all_bool
    data_set = args.data_set
    #print(window_size)
    #exit

    if NUM_CLASSES == 2:
        AUGMENT_DATA = False
    ##############################################################################################################
    
    
    print("Current directory: ",current_dir)   
       
    #experiment_folder = experiment_folder_path(subfolder="resnet1d"+"_num_of_sleep_stage_"+str(NUM_CLASSES))

    # Loads the files used for training and validating the model. If USING_CLAAUDIA is true it uses the full dataset, otherwise it uses a smaller subsample
    ##############################################################################################################
    if CONTINUE_TRAINING == False:
        training_fileNames, validation_fileNames, test_fileNames, ALREADY_PROCESSED = getSubjectPaths(data_type=data_type,USING_CLAAUDIA=USING_CLAAUDIA,train_split=0.2,test_split=0.5) # train_split should be 0.2. set to 0.5 for prototype purpose
    else: 
        training_fileNames, validation_fileNames, test_fileNames = getSubjectPathsContinuedTraining(resume_training_folder_name)
    
    #training_fileNames = training_fileNames[0:10]
    #validation_fileNames = validation_fileNames[0:10]
    #test_fileNames = test_fileNames[0:10]

    files_dict = {'training':training_fileNames,'validation':validation_fileNames,'testing':test_fileNames}
    #d = ECGDataSetSingle2(training_fileNames[0],shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = 250,resample_bool=resample_data,window_size=window_size,sampling_frequency=256,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    #d = ECGPeakMask2(training_fileNames[0],shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,light_sleep_vs_all_bool=light_sleep_vs_all_bool)

    for key, value in files_dict.items():
        print('Number of',key,'files:',len(value))
    
    if data_type == 'recreate' or data_type == 'ecgmask':
        training_dataset = create_datasets(training_fileNames,ECGPeakMask2,shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
        validation_dataset = create_datasets(validation_fileNames,ECGPeakMask2,shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
        test_dataset = create_datasets(test_fileNames,ECGPeakMask2,shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    elif data_type == 'rawecg':
        if data_set =='AAUWSS':
            AAUWSS_root_folder = r'./aligned_sleep_data_set'

            
            paired_paths=getAAUWSSDLECGPPGPathPairs(AAUWSS_root_folder)               
            dataset_list = []
            for subject,ecg_path,ppg_path in tqdm(paired_paths,desc='Loading ECG and PPG dataset',total=len(paired_paths)):
                #print(ecg_path,"\n",ppg_path)
                #print(f"Subject: {subject}, ECG File: {ecg_path}, PPG File: {ppg_path}")
                one_subject = AAUWSSDL(file_path=(ecg_path,ppg_path),signal_source='both',mask = False,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
                if data_type == 'rawecg':
                    one_subject.getitm_output = 'ecg'
                if data_type == 'rawppg':
                    one_subject.getitm_output = 'ppg'            
                dataset_list.append(one_subject)
            training_dataset = torch.utils.data.ConcatDataset(dataset_list) # Quickfix to test if train, val and test work with AAUWSS
            validation_dataset = torch.utils.data.ConcatDataset(dataset_list)
            test_dataset = torch.utils.data.ConcatDataset(dataset_list)
        
        elif data_set =='example':
            example_files = glob.glob('./example_data/*')
            
            training_fileNames = example_files[0:2]
            validation_fileNames = example_files[2:4]
            test_fileNames = example_files[4:6]
            sampling_frequency = 200
            training_dataset = create_datasets(training_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
            validation_dataset = create_datasets(validation_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
            test_dataset = create_datasets(test_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency= resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
               
        else:
            sampling_frequency = 256
            training_dataset = create_datasets(training_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
            validation_dataset = create_datasets(validation_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
            test_dataset = create_datasets(test_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency= resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    elif data_type == 'ibi':
        #training_dataset = create_datasets(training_fileNames,HRVFeaturesDataset,number_of_sleep_stage=NUM_CLASSES,hz=256,window_length_in_min=5,get_ibi=True,resample_ibi=True,normalize_ibi=True,already_processed=ALREADY_PROCESSED)
        #validation_dataset = create_datasets(validation_fileNames,HRVFeaturesDataset,number_of_sleep_stage=NUM_CLASSES,hz=256,window_length_in_min=5,get_ibi=True,resample_ibi=True,normalize_ibi=True,already_processed=ALREADY_PROCESSED)
        #test_dataset = create_datasets(test_fileNames,HRVFeaturesDataset,number_of_sleep_stage=NUM_CLASSES,hz=256,window_length_in_min=5,get_ibi=True,resample_ibi=True,normalize_ibi=True,already_processed=ALREADY_PROCESSED)
        None

    
    # If use_weighted_random_samlper is true it oversamples the less common classes. The weights are calculated based on 1/frequncy of each class.
    ##############################################################################################################
    if use_weighted_random_samlper == True:
        print("Use weighted random sampler: ", use_weighted_random_samlper )
        sample_calculator = torch.utils.data.DataLoader(training_dataset,batch_size=batch_size,drop_last=True)   
        y_true = []
        
        for val_local_batch, val_local_labels, val_file_name in sample_calculator:                     
            val_local_labels = val_local_labels.squeeze()    
            y_true.append(val_local_labels.cpu().detach().numpy().tolist())
        all_y_training_labels = [item for sublist in y_true for item in sublist]
        all_y_training_labels = np.array(all_y_training_labels)
        class_sample_count = np.array([len(np.where(all_y_training_labels == t)[0]) for t in np.unique(all_y_training_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in all_y_training_labels])
        print("Weigths from sampler: ", weight)
        samples_weight = torch.from_numpy(samples_weight)    
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),replacement=True)
        sampled_training_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(torch.utils.data.DataLoader(training_dataset,batch_size=batch_size,sampler=sampler)),index=[0])
        training_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(torch.utils.data.DataLoader(training_dataset,batch_size=batch_size)),index=[0])
        print("With weighted random samples distribution of training:\n",sampled_training_distribution,
              "\nWith regular samplig distribution:\n",training_distribution)
        shuffle_bool = False

    else:
        sampler = None    
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True,sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True)


    
    #Determine which model to use based on model_type and it's parameters
    ##############################################################################################################

    if model_type == 'ECGSleepNet':
        model = ECGSleepNet(nb_classes = NUM_CLASSES)
        model_name = "ECGSleepNet"            

        model_paramaters = {"NUM_CLASSES":NUM_CLASSES,
                            'loss_type':loss_type,
                            'data_type':data_type}  
    elif model_type == "dnet":
        Hz = 64
        N_DENSE_BLOCK = 4
        model = create_DenseNetmodel(Hz= Hz,N_DENSE_BLOCK= N_DENSE_BLOCK,NUM_CLASSES = NUM_CLASSES)
        model_name = "DenseNet1d"            
        model_paramaters = {"conv_kernel":15,
                            "Hz":Hz,
                            "N_DENSE_BLOCK":N_DENSE_BLOCK,
                            "NUM_CLASSES":NUM_CLASSES,
                            'data_type':data_type}      
    elif model_type == "ECGSleepNetAdaptable":
        n_timestep = resample_frequency*window_size # 64*270 = 17280 and 200*270=54000

        model = ECGSleepNetAdaptable(nb_classes = NUM_CLASSES,n_timestep=n_timestep)
        model_name = "ECGSleepNetAdaptable"            
        model_paramaters = {"NUM_CLASSES":NUM_CLASSES,
                            "n_timestep":n_timestep,
                            'loss_type':loss_type,
                            'data_type':data_type}        
    elif model_type == "fcn":
        if arg_kernel_size == 0:
            kernel_size = 8
        else:
            kernel_size = arg_kernel_size       
        input_shape = (training_dataset.datasets[0].processed_ECG_dataset[0].shape[1],1) #Default should be (1920, 1)
        model = FCN(input_shape = input_shape , nb_classes = NUM_CLASSES,kernel_size=kernel_size)
        model_name = "FCN"            
        model_paramaters = {"NUM_CLASSES":NUM_CLASSES,
                            'data_type':data_type}     
    elif model_type == 'simplecnn':
        if number_of_layers == 1:
            num_layers = 1
            kernel_size = [5]
        else:
            kernel_list = [5,3,5,3,5,3]
            num_layers=number_of_layers
            kernel_size=kernel_list[0:number_of_layers]
        model = Simple1DCNN(num_of_sleep_stages = NUM_CLASSES,num_layers=num_layers,kernel_size=kernel_size)
        model_name = "Simple1DCNN"            

        model_paramaters = {"NUM_CLASSES":NUM_CLASSES,
                            'kernel_size':kernel_size,
                            'num_layers':num_layers,
                            'loss_type':loss_type,
                            'data_type':data_type}          
    elif model_type == "rnet":
        if arg_kernel_size == 0:
            kernel_size = 16
        else:
            kernel_size = arg_kernel_size            
        
        #kernel_size = 16
        stride = 2
        n_block = 48
        downsample_gap = 6
        increasefilter_gap = 12
        base_filters = 64 # 64 for ResNet1D, 352 for ResNeXt1D

        model = ResNet1D(
            in_channels=1, 
            base_filters=base_filters, 
            kernel_size=kernel_size, 
            stride=stride, 
            groups=32, 
            n_block=n_block, 
            n_classes=NUM_CLASSES, 
            downsample_gap=downsample_gap, 
            increasefilter_gap=increasefilter_gap, 
            use_do=True)
        
        
        model_name = "ResNet1d"
        model_paramaters = {"kernel_size":kernel_size,
                            "stride":stride,
                            "n_block":n_block,
                            "downsample_gap":downsample_gap,
                            "base_filters":base_filters,
                            "n_classes":NUM_CLASSES,
                            'loss_type':loss_type,
                            'data_type':data_type}            
    elif model_type == "DeepSleep":
        if arg_kernel_size == 0:
            kernel_size = 25
        else:
            kernel_size = arg_kernel_size            
        num_residual_blocks = 8
        model = DeepSleep(num_residual_blocks = num_residual_blocks ,nb_classes =NUM_CLASSES)
        model_name = "DeepSleep"
        model_paramaters = {"kernel_size":kernel_size,
                            "num_residual_blocks":num_residual_blocks,
                            "n_classes":NUM_CLASSES,
                            'loss_type':loss_type,
                            'data_type':data_type}                 
    else:
        raise Exception("No model was chosen") 

    print("Chosen model: ",model_name )
    if torch.cuda.is_available():
        model = model.cuda()

       
         

    if loss_type == 'crossentropy':    
        # Calculates the weights for a weighted cross entropy loss
        if use_weighted_loss == True:
            #  weights:  tensor([ 1.0938,  0.3975, 37.1875,  1.8421])
            # tensor([0.1080, 0.0392, 3.6710, 0.1818]) # correct one
            #weights:  tensor([0.1080, 0.0392, 3.6710, 0.1818])
            weights, counter = getCrossEntropyWeights(training_dataset,num_of_classes = NUM_CLASSES)
            weights = torch.FloatTensor(weights)
            if torch.cuda.is_available():
                weights = weights.to('cuda')
            print("Using weighted loss: ", use_weighted_loss, "\n weights: ", weights)
            loss_fn = nn.CrossEntropyLoss(weights) 
        else:
            loss_fn = nn.CrossEntropyLoss() 
    '''
    if loss_type == 'focal':
        if use_weighted_loss == True:
            weights = get_weights(training_dataset,batch_size=batch_size)
            weights = torch.FloatTensor(weights) 
            if torch.cuda.is_available():
                weights = weights.to('cuda')
            print("Using weighted loss: ", use_weighted_loss, "\n weights: ", weights)
            loss_fn = FocalLoss(gamma=0.7,weights = weights) 
        else:
            loss_fn = FocalLoss(gamma=0.7)    
    '''
        
    epoch_starting_number = 0


    if optimizer_choice == 'rmsprop':
        optimizer, scheduler = get_optimzer_and_scheduler(model,lr=learning_rate,decay=decay)
    elif optimizer_choice=='adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)#, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=5,min_lr=1e-6)

    if CONTINUE_TRAINING == True:
        try:
            #__folder_name = "test_code_resnet1d_num_of_sleep_stage_2_2023-02-22_15-07"
            #__model_name = "densenet_epoch_state_dict_1.pt"
            print("Looking in folder: ",os.path.join(resume_training_folder_name,"*.pt"))                
            list_of_trained_epochs = glob.glob(os.path.join(resume_training_folder_name,"*.pt"))
            list_of_trained_epochs = [_x for _x in list_of_trained_epochs if "best_model" not in _x] # Remove best model from list
            print("Found: ", list_of_trained_epochs)
            __model_name = list_of_trained_epochs[0].split(os.sep)[-1].split("_")

            #list_of_trained_epochs = [int(x.split("_")[-1].split(".")[0]) for x in list_of_trained_epochs]
            current_epoch = max([int(x.split("_")[-1].split(".")[0]) for x in list_of_trained_epochs])
            __model_name[-1] = str(current_epoch)+".pt"
            __model_name = "_".join(__model_name)
            
            
            experiment_folder = resume_training_folder_name #os.path.join(current_dir,"experiment",__folder_name)
            unfinished_epoch_data_path = os.path.join(experiment_folder,"epoch_"+str(current_epoch+1))
            if os.path.exists(unfinished_epoch_data_path):
                shutil.rmtree(unfinished_epoch_data_path) # Delete GPU output of the unfinished epoch
            #model = keras.models.load_model(os.path.join(experiment_folder,__model_name))
            if torch.cuda.is_available()==False:
                loaded_dict = torch.load(os.path.join(experiment_folder,__model_name),map_location=torch.device('cpu'))
            else:
                loaded_dict = torch.load(os.path.join(experiment_folder,__model_name))
            model_state_dict = loaded_dict["model_state_dict"]
            epoch_starting_number = loaded_dict["epoch"]
            optimizer_state_dict = loaded_dict["optimizer_state_dict"]
            loss_dict = loaded_dict["loss"]
            subject_metric = loaded_dict['subject_metric']
            subject_predictions = loaded_dict['subject_predictions']

            #per_recording_dict = loaded_dict["per_recording_dict"]
            epoch_starting_number =current_epoch
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            best_validation_loss = loaded_dict["best_validation_loss"]

            print("\nSuccesfully loaded model: ",__model_name)
        except:
            print("No models found. Training from epoch 0")
            experiment_folder = resume_training_folder_name
            loss_dict = {"train":[],"validation":[]}
            epoch_starting_number=0
            best_validation_loss = 100

        
        
    else:            
        #Create the folder to save the model, epoch and performance data. Also instanciates the loss dict and saves the commandline arguments
        experiment_folder = experiment_folder_path(subfolder=folder_name_prefix+
                                                   model_name+
                                                   "_full_dataset_"+str(USING_CLAAUDIA)+
                                                   "_datatype_"+str(data_type)+
                                                   "_augment_data_"+str(AUGMENT_DATA)+
                                                   "_weighted_loss_" + str(use_weighted_loss) +
                                                   "_num_of_sleep_stage_"+str(NUM_CLASSES))
        loss_dict = {'training':[],'validation':[],'test':[],'control_training':[]}
        subject_metric = {'training':[],'validation':[],'test':[],'control_training':[]}
        subject_predictions = {'training':[],'validation':[],'test':[],'control_training':[]}

        epoch_starting_number=0
        best_validation_loss = 100

        with open(os.path.join(experiment_folder,'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)     
            
        with open(os.path.join(experiment_folder,'training_files.json'), "w") as fp:
            json.dump(files_dict, fp)  # encode dict into JSON               
    
    try:
        label_mapping = training_dataset.datasets[0].label_dict
        train_len =len(training_dataset.datasets)
        valid_len=len(validation_dataset.datasets)
        test_len =len(test_dataset.datasets)
    except:
        pass
    try:
        label_mapping = training_dataset.cached_dataset.label_dict
        train_len =len(training_dataset.file_paths)
        valid_len=len(validation_dataset.file_paths)
        test_len =len(test_dataset.file_paths)
    except:
        pass

    training_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(training_loader),index=[0])


    fig = DistributionPlot(training_distribution,title='Training Distribution \n Recordings = '+str(train_len),label_mapping_dict=label_mapping)
    plt.savefig(os.path.join(experiment_folder,"training_distribution"+".png"))
    
    validation_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(validation_loader),index=[0])
    fig = DistributionPlot(validation_distribution,title='Validation Distribution \n Recordings = '+str(valid_len),label_mapping_dict=label_mapping)
    plt.savefig(os.path.join(experiment_folder,"validation_distribution"+".png"))

    test_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(test_loader),index=[0])
    fig = DistributionPlot(test_distribution,title='Test Distribution \n Recordings = '+str(test_len),label_mapping_dict=label_mapping)
    plt.savefig(os.path.join(experiment_folder,"test_distribution"+".png"))    
    
    # Loop over epochs
    pbar = tqdm(range(epoch_starting_number,n_epochs),desc="Epoch loop")
    # Early stopping parameters
    #patience = 27
    #best_loss = float('inf')
    #counter = 0
    #exit()
    for epoch in pbar:
        #break # to exit training loop
        print('EPOCH {}:'.format(epoch + 1))
        t0 = time.time()
    
        # Make sure gradient tracking is on, and do a pass over the data
        print('Training: \n')
        model.train(True)
        avg_loss,training_prediction_batch_dict = run_one_epoch(model, optimizer, None, loss_fn,training_loader,train=True)
        train_scores,train_predictions = gatherBatchDataToSubjectMetrics(training_prediction_batch_dict)
        subject_predictions['training'].append(train_predictions)
        subject_metric['training'].append(train_scores)
        
        model.eval()
    
        # Disable gradient computation and reduce memory consumption.
        print('Validation: \n')
    
        with torch.no_grad():
            avg_vloss,validation_prediction_batch_dict = run_one_epoch(model, None, scheduler, loss_fn,validation_loader,train=False)
            val_scores,val_predictions = gatherBatchDataToSubjectMetrics(validation_prediction_batch_dict)
            subject_metric['validation'].append(val_scores)
            subject_predictions['validation'].append(val_predictions)


        # TEST SET
        print('Testing: \n')
        with torch.no_grad():
            avg_test_loss,testing_prediction_batch_dict = run_one_epoch(model, None, None, loss_fn, test_loader,train=False)
            test_scores,test_predictions = gatherBatchDataToSubjectMetrics(testing_prediction_batch_dict)
            subject_metric['test'].append(test_scores)
            subject_predictions['test'].append(test_predictions)
            
        with torch.no_grad():
            control_avg_loss,control_training_prediction_batch_dict = run_one_epoch(model, None, None, loss_fn,training_loader,train=False)
            control_train_scores,control_train_predictions = gatherBatchDataToSubjectMetrics(control_training_prediction_batch_dict)
            subject_predictions['control_training'].append(control_train_predictions)
            subject_metric['control_training'].append(control_train_scores)


        print('\nTraining time: {:.1f} minutes LOSS train {:.3f} valid {:.3f} test {:.3f} control{:.3f}'.format((time.time() - t0)/60, avg_loss, avg_vloss,avg_test_loss,control_avg_loss))
    
        loss_dict['training'].append(avg_loss)
        loss_dict['validation'].append(avg_vloss)
        loss_dict['test'].append(avg_test_loss)    
        loss_dict['control_training'].append(control_avg_loss)


        

                   
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_dict,
                    'model_parameters':model_paramaters,
                    'subject_metric':subject_metric,
                    'subject_predictions':subject_predictions,
                    'best_validation_loss':best_validation_loss},
                   os.path.join(experiment_folder,model_name+"_state_dict_epoch_"+str(epoch)+".pt"))     
        
        if save_intermediate_models == False:
            try:
                os.remove(os.path.join(experiment_folder,model_name+"_state_dict_epoch_"+str(epoch-1)+".pt"))
            except:
                None
        
        if (avg_vloss < best_validation_loss) and (save_best_model_bool==True):
            best_validation_loss = avg_vloss
            print("Saved best model at epoch:",epoch)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dict,
                        'model_parameters':model_paramaters,'best_validation_loss':best_validation_loss}, os.path.join(experiment_folder,model_name+"_best_model.pt"))        
            
            with open(os.path.join(experiment_folder,'best_epoch_performance.json'), "w") as fp:
                d = {"best_epoch":epoch,
                     'subject_metric':subject_metric}
                json.dump(d, fp)  
            file = open(os.path.join(experiment_folder,'best_epoch_subject_predictions.pkl'), 'wb')
            pickle.dump(subject_predictions,file )
            file.close()
        """
        if avg_vloss < best_loss:
            best_loss = avg_vloss
            counter = 0
        else:
            counter += 1
        
        if (counter >= patience) and (patience != 0):
            print("Early stopping due to no improvement in validation loss!")
            break
        """
        
    print("Finished training")
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict,
                'model_parameters':model_paramaters}, os.path.join(experiment_folder,model_name+"_final_model.pt"))

    
    file = open(os.path.join(experiment_folder,'subject_predictions.pkl'), 'wb')
    pickle.dump(subject_predictions,file )
    file.close()

    with open(os.path.join(experiment_folder,'subject_metric.json'), "w") as fp:
        json.dump(subject_metric, fp)  

    with open(os.path.join(experiment_folder,'loss.json'), "w") as fp:
        json.dump(loss_dict, fp)  #         
        
        
    #####################################################################################


    baselines = {}
    baselines['training'] = calculateBaselinePrClass(subject_predictions['training'][-1]['labels'],label_mapping['mapping'])
    baselines['validation'] = calculateBaselinePrClass(subject_predictions['validation'][-1]['labels'],label_mapping['mapping'])
    baselines['test'] = calculateBaselinePrClass(subject_predictions['test'][-1]['labels'],label_mapping['mapping'])
    with open(os.path.join(experiment_folder,'baseline_metrics.json'), "w") as fp:
        json.dump(baselines, fp)  
    ##################################################################################### PLOTTING

    p1 = create_loss_plot(loss_dict)
    plt.savefig(os.path.join(experiment_folder,"loss.png"))
    
    cm1, d = create_confusion_matrix(validation_dataset,model,title='Validation set')
    plt.savefig(os.path.join(experiment_folder,'confusion_matix_validation.png'))
    
    cm2, d = create_confusion_matrix(test_dataset,model,title='Test set')
    plt.savefig(os.path.join(experiment_folder,'confusion_matix_test.png'))
    
    cm3, d = create_confusion_matrix(training_dataset,model,title = 'Training set')
    plt.savefig(os.path.join(experiment_folder,'confusion_matix_training.png'))
    
    metric_per_epoch = {'training': getMetricPerEpoch(subject_metric['training']),
                        'validation':getMetricPerEpoch(subject_metric['validation']),
                        'test':getMetricPerEpoch(subject_metric['test'])}
    
    with open(os.path.join(experiment_folder,'metric_per_epoch.json'), "w") as fp:
        json.dump(metric_per_epoch, fp)  #        
    
    
    g = PlotMetricDistribution(subject_metric)
    plt.savefig(os.path.join(experiment_folder,"metric_distribution.png"))
    

    fig, axs, accuracy_df_output = PlotClassAccuracies(subject_predictions,label_mapping)
    accuracy_df_output.to_pickle(os.path.join(experiment_folder,'accuracy_df.pkl'))
    plt.savefig(os.path.join(experiment_folder,"class_accuracies_epochs.png"))
    
    
    fig, axs = MetricPlots(metric_per_epoch,NUM_CLASSES=NUM_CLASSES)
    plt.savefig(os.path.join(experiment_folder,"train_accuracy.png"))
    
    df_long = createBaselineAndPerformanceDf(metric_per_epoch,baselines,model_name)
    df_long.to_pickle(os.path.join(experiment_folder,'baseline_performance_long.pkl'))
    fig, axs = createBaselineComparisonPlot(df_long)
    plt.savefig(os.path.join(experiment_folder,"baseline_comparison.png"))


    create_hypnograms(model = model,dataloader=validation_dataset,experiment_folder = os.path.join(experiment_folder,'validation'),NUM_CLASSES=NUM_CLASSES,batch_size=batch_size)
    create_hypnograms(model = model,dataloader=test_dataset,experiment_folder = os.path.join(experiment_folder,'testing'),NUM_CLASSES=NUM_CLASSES,batch_size=batch_size)
