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
from torch.utils.data import Subset
from torch.utils.data import random_split

from deep_learning_loop_utilities import str2bool 
from deep_learning_loop_utilities import get_train_valid_test_files_recreate,get_train_valid_test_files_recreate_small,getSubjectPaths, getSubjectPathsContinuedTraining
from deep_learning_loop_utilities import create_datasets, getCrossEntropyWeights, get_optimzer_and_scheduler
from deep_learning_plots import create_loss_plot, create_confusion_matrix, create_hypnograms,getDistributionOfGeneratorLabels,DistributionPlot, MetricPlots, PlotMetricDistribution, PlotClassAccuracies, createBaselineComparisonPlot
from deep_learning_loop_utilities import calculate_metrics, gatherBatchDataToSubjectMetrics, getDataFrameOfRecordingMetric, getMetricPerEpoch
from deep_learning_loop_utilities import calculateBaselinePrClass,createBaselineAndPerformanceDf


from dataset_densenet import ECGDataSetSingle2
from ecg_mask_loader_simple import ECGPeakMask2
from dense_net_utils import setCWDToScriptLocation, experiment_folder_path, get_weights 
from resnet1d import ResNet1D
from dense_net_model import create_DenseNetmodel
from fcn import FCN
from mymodel import ECGSleepNet
from ECGSleepNetAdaptable import ECGSleepNetAdaptable
from simple_cnn import Simple1DCNN
from deep_sleep_bcg_model import DeepSleep
from data_loader_sleep_study_dataset import SFDL
from deep_learning_training_loop import run_one_epoch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
plt.style.use('seaborn')

import shutil
#Helper functions
##############################################################################################################


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def getSFDLECGPPGPathPairs(soundfocus_root_folder):
    ecg_folder = os.path.join(soundfocus_root_folder,'ecg',"*")
    ppg_folder = os.path.join(soundfocus_root_folder,'ppg',"*")
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
    parser.add_argument('-CAAU', '--USING_CLAAUDIA',type=str2bool,default=False,help='Using Claaudia AI Cloud',dest='USING_CLAAUDIA')    
    parser.add_argument('-m', '--model',type=str,default='fcn',choices=['fcn','dnet','simplecnn','rnet','rnext','DeepSleep','ECGSleepNet','ECGSleepNetAdaptable'],help='Which model to use. Choose between: \n rnet = ResNet \n rnext = ResNext \n dnet = DenseNet \n fcn = Fully Connected CNN \n ibi_cnn = CNN with IBI input ',
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
    parser.add_argument('-dtype', '--data_type',type=str,default='rawecg',choices = ['rawecg','ecgmask','rawppg'],help='Which input data to use. \necg = raw ECG signal \nfeatures = HRV features \nibi = Interbeat Interval Sequence \necgmask = A mask of ECG peaks ',dest='data_type') 
    parser.add_argument('-shuffle', '--shuffle_training',type=str2bool,default=False,help='Shuffle the training data. Validation and test are not shuffled',dest='shuffle_bool')    
    parser.add_argument('-optim', '--optimizer',type=str,default='rmsprop',choices = ['rmsprop','adam'],help='Which optimizer to use. adam or rmsprop',dest='optimizer') 
    parser.add_argument('-dec', '--decay',type=float,default=0.0,help='The learning rate of the optimizer',dest='weight_decay')  
    parser.add_argument('-layers', '--number_of_layers',type=int,default=1,help='Number of layers in the CNN',dest='number_of_layers')    
    parser.add_argument('-norm_type', '--normalize_type',type=str,default='zscore',choices=["zscore","paper"],help='Choice of normalization for raw ecg data',dest='normalize_type')
    parser.add_argument('-norm', '--normalize_data',type=str2bool,default=True,help='Normalize input to neural networks',dest='normalize_data')
    parser.add_argument('-resample', '--resample_signal',type=str2bool,default=True,help='Resample input signal',dest='resample_signal')
    parser.add_argument('-resample_hz', '--resample_frequency',type=int,default=64,help='The new frequency of the input signal',dest='resample_frequency')
    parser.add_argument('-w_size', '--window_size',type=int,default=None,help='Changes the epoch size from 30 seconds to window_size',dest='window_size')
    parser.add_argument('-pat', '--patience',type=int,default=0,help='Amount of patience compared to validation set',dest='patience')
    parser.add_argument('-s_best', '--save_best_model',type=str2bool,default=True,help='Normalize input to neural networks',dest='save_best_model')
    parser.add_argument('-l_vs_rest', '-light_sleep_vs_all_bool',type=str2bool,default=False,help='Collapses N1 and N2 into Light Sleep and N3,REM and Wake into Not Light Sleep. Only works when sleep stages set to 2',dest='light_sleep_vs_all_bool')
    parser.add_argument('-dset', '--data_set',type=str,default='',choices = ['sfdl'],help='Which dataset to use. \nsfdl = The SoundFocus dataset loader that contain ECG and PPGs ',dest='data_set') 
    parser.add_argument('-use_ptm', '--use_pretrained_model',type=str,default="",help='Path to the pretrained model. If it is not set it will not use a pretrained model',dest='use_pretrained_model')   
    parser.add_argument('-filter', '--filter_bool',type=str2bool,default=False,help='Filter the PPG signal before doing any other processing',dest='filter_bool')
    parser.add_argument('-ft', '--fine_tune_epochs',type=int,default=0,help='The amount of sleep epochs used for fine tuning',dest='n_epochs_for_finetune')
    parser.add_argument('-ft_train', '--fine_tune_epochs_training',type=int,default=0,help='The duration of finetuning in training epochs',dest='fine_tune_epochs')
    parser.add_argument('-freeze', '--freeze_layers',type=str2bool,default=False,help='Freeze layers for fine tuning',dest='freeze_layers')
   
    args = parser.parse_args()
    
    # args.number_of_sleep_stages=2
    # args.folder_name_prefix = 'kfoldtest_pretrained'
    # args.model_type = 'ECGSleepNetAdaptable'#'ECGSleepNet' #fcn, dnt
    # args.resample_frequency = 64#200
    # args.resample_signal = True
    # args.light_sleep_vs_all_bool = False
    # args.use_scheduler = False
    # args.window_size = 270
    # args.n_epochs = 2
    # args.data_type = 'rawecg'
    # args.use_weighted_loss = True
    # #args.filter_bool = True

    
    # args.use_pretrained_model =r'C:\temp\Organized\64 Hz\64HzAllsubjCorrectECGSleepNetAdaptable_full_dataset_True_datatype_rawecg_augment_data_False_weighted_loss_True_num_of_sleep_stage_5_2024-10-01_18-49\ECGSleepNetAdaptable_final_model.pt'

    #args.resume_training = True   
    # args.number_of_sleep_stages=5
    # args.n_epochs = 10
    # args.USING_CLAAUDIA = False
    # args.data_type = 'rawecg'
    # args.model_type = 'dnet'#'ECGSleepNet' #fcn, dnt
    # #args.shuffle_bool =False# True
    # #args.augment = False
    # args.use_weighted_loss = False
    # #args.use_weighted_sampler = False
    # args.learning_rate = 0.001
    # args.optimizer = 'rmsprop'
    # args.weight_decay = 1e-6#0.000001
    # #args.number_of_layers = 1
    # #args.shuffle_bool=True
    # args.normalize_type = 'paper'
    # args.resample_frequency = 64#200
    # args.resample_signal = True
    # args.light_sleep_vs_all_bool = False
    # args.use_scheduler = False

    #args.window_size = 270
    
    
    


    CONTINUE_TRAINING = args.resume_training
    
    if CONTINUE_TRAINING == True:
        resume_training_folder_name = args.resume_folder_name
        del args
        parser2 = argparse.ArgumentParser()
        args, unknown = parser2.parse_known_args()
        with open(os.path.join(resume_training_folder_name,'commandline_args.txt'), 'r') as f:
            args.__dict__ = json.load(f)
            args.patience = 0
    
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
    use_pretrained_model = args.use_pretrained_model
    filter_bool = args.filter_bool
    n_epochs_for_finetune = args.n_epochs_for_finetune
    fine_tune_epochs = args.fine_tune_epochs
    freeze_bool = args.freeze_layers

    #print(window_size)
    #exit

    if NUM_CLASSES == 2:
        AUGMENT_DATA = False
    ##############################################################################################################
    
    
    print("Current directory: ",current_dir)   
       
    #experiment_folder = experiment_folder_path(subfolder="resnet1d"+"_num_of_sleep_stage_"+str(NUM_CLASSES))

    # Loads the files used for training and validating the model. If USING_CLAAUDIA is true it uses the full dataset, otherwise it uses a smaller subsample
    ##############################################################################################################
    

    soundfocus_root_folder = r'./aligned_sleep_data_set'
    
    
    paired_paths=getSFDLECGPPGPathPairs(soundfocus_root_folder)   
    #paired_paths = [x for x in paired_paths if x[0] not in ['13']]
    #paired_paths = paired_paths[0:3]
    
    dataset_list = []
    for subject,ecg_path,ppg_path in tqdm(paired_paths,desc='Loading ECG and PPG dataset',total=len(paired_paths)):
        #print(ecg_path,"\n",ppg_path)
        #print(f"Subject: {subject}, ECG File: {ecg_path}, PPG File: {ppg_path}")
        one_subject = SFDL(file_path=(ecg_path,ppg_path),signal_source='both',mask = False,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,light_sleep_vs_all_bool=light_sleep_vs_all_bool,filter_bool = filter_bool)
        if data_type == 'rawecg':
            one_subject.getitm_output = 'ecg'
        if data_type == 'rawppg':
            one_subject.getitm_output = 'ppg'            
        dataset_list.append(one_subject)

    k_fold_test_datasets = []
    k_fold_training_dataset = []
    # Split datasets into folds
    for i in range(len(dataset_list)):
        print('Fold: ',i)
        test_dataset = dataset_list[i]
        print("Test set: ",test_dataset.subject_name)
        k_fold_test_datasets.append(test_dataset)
        
        training_dataset = [dataset_list[j] for j in range(len(dataset_list)) if j != i]
        print('Training set:')
        for tr in training_dataset:
            print(tr.subject_name)
        print("\n")
        k_fold_training_dataset.append(training_dataset)
   


    
    
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
        
    if use_pretrained_model != '':
        print('Pretrained model used')
        if torch.cuda.is_available()==False:
            loaded_dict = torch.load(use_pretrained_model,map_location=torch.device('cpu'))
        else:
            loaded_dict = torch.load(use_pretrained_model)
        model_state_dict = loaded_dict["model_state_dict"]
        
    experiment_folder_root = experiment_folder_path(subfolder=folder_name_prefix+
                                               model_name+
                                               "_datatype_"+str(data_type)+
                                               "_weighted_loss_" + str(use_weighted_loss) +
                                               "_num_of_sleep_stage_"+str(NUM_CLASSES))    
    
    training_distribution = pd.DataFrame(getDistributionOfGeneratorLabels(torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(dataset_list), batch_size=batch_size, shuffle=shuffle_bool)),index=[0])

    label_mapping = dataset_list[0].label_dict


    fig = DistributionPlot(training_distribution,title='Training Distribution for \n Recordings = '+str(len(dataset_list)),label_mapping_dict=label_mapping)
    plt.savefig(os.path.join(experiment_folder_root,"training_distribution"+".png"))    
    with open(os.path.join(experiment_folder_root,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)      
    
    k_fold_tracking = {}
    for sub_training_dataset,sub_test_dataset in tqdm(zip(k_fold_training_dataset,k_fold_test_datasets),total=len(k_fold_training_dataset),desc="K fold validation"):
        """
        indices = list(range(len(sub_test_dataset)))
        train_indices = indices[:10]  # First 3 samples
        test_indicies = indices[10:]    # Remaining samples
        extra_train_dataset = Subset(sub_test_dataset, train_indices)
        short_test_dataset = Subset(sub_test_dataset, test_indicies)
        """
        '''
        n_epochs_for_finetune = 10
        from copy import deepcopy
        temp = deepcopy(sub_test_dataset)
        
        temp.labels = temp.labels[:n_epochs_for_finetune]
        temp.processed_ECG_dataset= temp.processed_ECG_dataset[:n_epochs_for_finetune] 
        temp.ecg_labels = temp.ecg_labels[:n_epochs_for_finetune]
        temp.processed_PPG_dataset= temp.processed_PPG_dataset[:n_epochs_for_finetune]   
        temp.ppg_labels = temp.ppg_labels[:n_epochs_for_finetune]
        sub_training_dataset.append(temp)

        sub_test_dataset.labels = sub_test_dataset.labels[n_epochs:]
        sub_test_dataset.processed_ECG_dataset= sub_test_dataset.processed_ECG_dataset[n_epochs_for_finetune:] 
        sub_test_dataset.ecg_labels = sub_test_dataset.ecg_labels[n_epochs_for_finetune:]
        sub_test_dataset.processed_PPG_dataset= sub_test_dataset.processed_PPG_dataset[n_epochs_for_finetune:]   
        sub_test_dataset.ppg_labels = sub_test_dataset.ppg_labels[n_epochs_for_finetune:]
        '''

        """
        sub_training_dataset.append(extra_train_dataset)
        for i, dataset in enumerate(sub_training_dataset):
            if isinstance(dataset, Subset):
                # Extract the dataset and re-wrap it to maintain compatibility
                sub_training_dataset[i] = torch.utils.data.ConcatDataset([dataset])     
        
        
        #sub_test_dataset = deepcopy(short_test_dataset)
        """

        
        if use_pretrained_model != '':
            model.load_state_dict(model_state_dict)
        else:
            model.apply(reset_weights)
        print(sub_test_dataset.subject_name)
        tracking = {'loss':None,'scores':None,"last_epoch_score":None}
        experiment_folder = os.path.join(experiment_folder_root,sub_test_dataset.subject_name)
        os.makedirs(experiment_folder,exist_ok = True)
        #exit()

        

        dataset_before_split = torch.utils.data.ConcatDataset(sub_training_dataset)
        
        # Define the split lengths
        train_ratio = 0.8  # 80% training data
        val_ratio = 0.2    # 20% validation data
        
        # Calculate lengths
        total_length = len(dataset_before_split)
        train_length = int(total_length * train_ratio)
        val_length = total_length - train_length  # Ensure the lengths sum up
        
        # Split the dataset
        training_dataset, validation_dataset = random_split(dataset_before_split, [train_length, val_length])        
        # Print sizes to verify
        print(f"Train dataset size: {len(training_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
        
        #training_dataset = dataset_before_split # USED FOR TESTING PURPOSES
        #validation_dataset = dataset_before_split #  USED FOR TESTING PURPOSES
    
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
            
            
        files_dict = {'training':[x.subject_name for x in sub_training_dataset],'testing':[sub_test_dataset.subject_name]}
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=False,sampler=sampler)
        test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=False)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=False)

    
        #Determine which model to use based on model_type and it's parameters
        ##############################################################################################################
    
           
             
    
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
        continue    
        epoch_starting_number = 0
        # Loop over epochs
        pbar = tqdm(range(epoch_starting_number,n_epochs),desc="Epoch loop")
    
        if optimizer_choice == 'rmsprop':
            optimizer, scheduler = get_optimzer_and_scheduler(model,lr=learning_rate,decay=decay)
        elif optimizer_choice=='adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)#, weight_decay=1e-6)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=5,min_lr=1e-6)


        #Create the folder to save the model, epoch and performance data. Also instanciates the loss dict and saves the commandline arguments

        loss_dict = {'training':[],'validation':[],'test':[]}
        subject_metric = {'training':[],'validation':[],'test':[]}
        subject_predictions = {'training':[],'validation':[],'test':[]}
    
        epoch_starting_number=0   
            
        with open(os.path.join(experiment_folder,'training_files.json'), "w") as fp:
            json.dump(files_dict, fp)  # encode dict into JSON               
        
        try:
            label_mapping = training_dataset.datasets[0].label_dict
            train_len =len(training_dataset.datasets)
            test_len =len(test_dataset.datasets)
        except:
            pass
        try:
            label_mapping = training_dataset.cached_dataset.label_dict
            train_len =len(training_dataset.file_paths)
            test_len =len(test_dataset.file_paths)
        except:
            pass
    
    
        

    
    
        
        # Early stopping parameters
        #patience = 27
        #best_loss = float('inf')
        #counter = 0
        
        best_validation_loss = 100
        for epoch in pbar:
            #break # to exit training loop
            print('EPOCH {}:'.format(epoch + 1))
            t0 = time.time()

            #exit()
            print('Training: \n')
            model.train(True)
            avg_loss,training_prediction_batch_dict = run_one_epoch(model, optimizer, None, loss_fn,training_loader,train=True)
            train_scores,train_predictions = gatherBatchDataToSubjectMetrics(training_prediction_batch_dict)
            subject_predictions['training'].append(train_predictions)
            subject_metric['training'].append(train_scores)
            
            model.eval()
        
            # Disable gradient computation and reduce memory consumption.
            print('Validation: \n')
            #avg_vloss = 0
                    
            with torch.no_grad():
                avg_vloss,validation_prediction_batch_dict = run_one_epoch(model, None, scheduler, loss_fn,validation_loader,train=False)
                val_scores,val_predictions = gatherBatchDataToSubjectMetrics(validation_prediction_batch_dict)
                subject_metric['validation'].append(val_scores)
                subject_predictions['validation'].append(val_predictions)            
    
    
    
            # TEST SET
            print('Testing: \n')
            with torch.no_grad():
                avg_test_loss,testing_prediction_batch_dict = run_one_epoch(model, None, scheduler, loss_fn, test_loader,train=False)
                test_scores,test_predictions = gatherBatchDataToSubjectMetrics(testing_prediction_batch_dict)
                subject_metric['test'].append(test_scores)
                subject_predictions['test'].append(test_predictions)
                
                #subject_metric['validation'].append(test_scores) # Fake data to get the code to run without changing the validation plotting
                #subject_predictions['validation'].append(test_predictions) # Fake data to get the code to run without changing the validation plotting
    
    
            print('\nTraining time: {:.1f} minutes LOSS train {:.3f} valid {:.3f} test {:.3f}'.format((time.time() - t0)/60, avg_loss, avg_vloss,avg_test_loss))
        
            loss_dict['training'].append(avg_loss)
            loss_dict['validation'].append(avg_vloss)
            loss_dict['test'].append(avg_test_loss)    
    
            
    
                       
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dict,
                        'model_parameters':model_paramaters,
                        'subject_metric':subject_metric,
                        'subject_predictions':subject_predictions},
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
                            'model_parameters':model_paramaters}, os.path.join(experiment_folder,model_name+"_best_model.pt"))        
                
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
        
        tracking['loss'] = loss_dict
        with open(os.path.join(experiment_folder,'loss.json'), "w") as fp:
            json.dump(loss_dict, fp)  #         
            
            
        #####################################################################################
    
    
        baselines = {}
        baselines['training'] = calculateBaselinePrClass(subject_predictions['training'][-1]['labels'],label_mapping['mapping'])
        baselines['validation'] = calculateBaselinePrClass(subject_predictions['test'][-1]['labels'],label_mapping['mapping']) # Fake data so plot still works.
        baselines['test'] = calculateBaselinePrClass(subject_predictions['test'][-1]['labels'],label_mapping['mapping'])
        with open(os.path.join(experiment_folder,'baseline_metrics.json'), "w") as fp:
            json.dump(baselines, fp)  
        ##################################################################################### PLOTTING
    
        p1 = create_loss_plot(loss_dict)
        plt.savefig(os.path.join(experiment_folder,"loss.png"))
        
    
        
        cm2, d = create_confusion_matrix(sub_test_dataset,model,title='Test set') # DOES NOT WORK
        plt.savefig(os.path.join(experiment_folder,'confusion_matix_test.png'))
        
        cm3, d = create_confusion_matrix(training_dataset,model,title = 'Training set')
        plt.savefig(os.path.join(experiment_folder,'confusion_matix_training.png'))

        
        metric_per_epoch = {'training': getMetricPerEpoch(subject_metric['training']),
                            'test':getMetricPerEpoch(subject_metric['test']),
                            'validation':getMetricPerEpoch(subject_metric['test'])} #Validation is fake data
        tracking['scores'] = metric_per_epoch
        #tracking['last_epoch_scores'] = {'training':metric_per_epoch['training'][-1],
        #                                 'test':metric_per_epoch['test'][-1]}
        
        k_fold_tracking[sub_test_dataset.subject_name] = tracking


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
        
        
        # Experimental option to further finetune a model on the first X epochs of a subject

        """
        # Fine-Tuning Loop
        print("\n\n=== Fine-Tuning on Subject-Specific Wake Epochs ===\n\n")
        
        # Define fine-tuning hyperparameters
        if n_epochs_for_finetune != 0:
            fine_tune_learning_rate = learning_rate#0.0001  # Reduce learning rate for fine-tuning
            fine_tune_batch_size = 5  # Smaller batch size for fine-tuning
            fine_tune_patience = 5  # Early stopping patience for fine-tuning
            best_fine_tune_loss = float('inf')
            fine_tune_counter = 0
            
            # Re-initialize optimizer with smaller learning rate
            fine_tune_optimizer,fine_tune_scheduler = get_optimzer_and_scheduler(model,lr=fine_tune_learning_rate,decay=decay)
            #loss_fn = nn.CrossEntropyLoss() 
            
            from copy import deepcopy
            fine_tune_train_set = deepcopy(sub_test_dataset)
            
            fine_tune_train_set.labels = fine_tune_train_set.labels[:n_epochs_for_finetune]
            fine_tune_train_set.processed_ECG_dataset= fine_tune_train_set.processed_ECG_dataset[:n_epochs_for_finetune] 
            fine_tune_train_set.ecg_labels = fine_tune_train_set.ecg_labels[:n_epochs_for_finetune]
            fine_tune_train_set.processed_PPG_dataset= fine_tune_train_set.processed_PPG_dataset[:n_epochs_for_finetune]   
            fine_tune_train_set.ppg_labels = fine_tune_train_set.ppg_labels[:n_epochs_for_finetune]
            #sub_training_dataset.append(temp)
    
            sub_test_dataset.labels = sub_test_dataset.labels[n_epochs:]
            sub_test_dataset.processed_ECG_dataset= sub_test_dataset.processed_ECG_dataset[n_epochs_for_finetune:] 
            sub_test_dataset.ecg_labels = sub_test_dataset.ecg_labels[n_epochs_for_finetune:]
            sub_test_dataset.processed_PPG_dataset= sub_test_dataset.processed_PPG_dataset[n_epochs_for_finetune:]   
            sub_test_dataset.ppg_labels = sub_test_dataset.ppg_labels[n_epochs_for_finetune:]        
    
            
            # Fine-tuning Dataset Preparation
            fine_tune_loader = torch.utils.data.DataLoader(fine_tune_train_set, batch_size=fine_tune_batch_size, shuffle=True)
            fine_tune_test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
            
            #sub_test_dataset
            #fine_tune_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=fine_tune_batch_size, shuffle=True)
            #fine_tune_test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
            
    
            fine_tune_loss_dict = {'training':[],'validation':[],'test':[]}
            fine_tune_subject_metric = {'training':[],'validation':[],'test':[]}
            fine_tune_predictions_dict = {'training':[],'validation':[],'test':[]}        
            
            
            if freeze_bool == True:
                def freeze_layers(model, freeze_up_to="resblock12"):
                    freezing = True  # Start with freezing enabled
                    for name, child in model.named_children():
                        if name == freeze_up_to:
                            freezing = False  # Stop freezing when we reach the target layer
                        
                        # Freeze/unfreeze layers
                        for param in child.parameters():
                            param.requires_grad = not freezing
                
                # Apply freezing to your model
                freeze_layers(model)
                
                # Print trainable parameters to verify
                for name, param in model.named_parameters():
                    print(f"{name}: Requires Grad = {param.requires_grad}")                
            
            # Fine-Tuning Loop
            for epoch in tqdm(range(fine_tune_epochs), desc="Fine-Tuning Epoch Loop"):
                print(f"Fine-Tuning EPOCH {epoch + 1}:")
                t0 = time.time()
            
                # Training on Fine-Tuning Dataset
                model.train(True)
                fine_tune_loss, fine_tune_prediction_batch_dict = run_one_epoch(
                    model, fine_tune_optimizer, None, loss_fn, fine_tune_loader, train=True
                )
                fine_tune_scores, fine_tune_predictions = gatherBatchDataToSubjectMetrics(fine_tune_prediction_batch_dict)
                
                fine_tune_predictions_dict['training'].append(train_predictions)
                fine_tune_subject_metric['training'].append(train_scores)
                
                model.eval()
            
                # Disable gradient computation and reduce memory consumption.
                print('Validation: \n')
                avg_vloss = 0
        
        
                # TEST SET
                print('Testing: \n')
                with torch.no_grad():
                    fine_tune_avg_test_loss,fine_tune_testing_prediction_batch_dict = run_one_epoch(model, None, None, loss_fn, fine_tune_test_loader,train=False)
                    fine_tune_test_scores,fine_tune_test_predictions = gatherBatchDataToSubjectMetrics(fine_tune_testing_prediction_batch_dict)
                    fine_tune_subject_metric['test'].append(fine_tune_test_scores)
                    fine_tune_predictions_dict['test'].append(fine_tune_test_predictions)
                    
                    fine_tune_subject_metric['validation'].append(fine_tune_test_scores) # Fake data to get the code to run without changing the validation plotting
                    fine_tune_predictions_dict['validation'].append(fine_tune_test_predictions) # Fake data to get the code to run without changing the validation plotting
        
        
                print('\nTraining time: {:.1f} minutes finetune LOSS train {:.3f} valid {:.3f} test {:.3f}'.format((time.time() - t0)/60, fine_tune_loss, fine_tune_avg_test_loss,fine_tune_avg_test_loss))
            
                fine_tune_loss_dict['training'].append(fine_tune_loss)
                fine_tune_loss_dict['validation'].append(0)
                fine_tune_loss_dict['test'].append(fine_tune_avg_test_loss)                
            
            p6 = create_loss_plot(fine_tune_loss_dict)
            plt.savefig(os.path.join(experiment_folder,"fine_tune_loss.png"))
            
        
            
            cm2, d = create_confusion_matrix(sub_test_dataset,model,title='Test set')
            plt.savefig(os.path.join(experiment_folder,'confusion_matix_test_fine_tune.png'))
            
            fine_tune_metric_per_epoch = {'training': getMetricPerEpoch(fine_tune_subject_metric['training']),
                                'test':getMetricPerEpoch(fine_tune_subject_metric['test']),
                                'validation':getMetricPerEpoch(fine_tune_subject_metric['test'])} #Validation is fake data



            with open(os.path.join(experiment_folder,'fine_tune_metric_per_epoch.json'), "w") as fp:
                json.dump(fine_tune_metric_per_epoch, fp)  # 
                
            file = open(os.path.join(experiment_folder,'fine_tune_subject_predictions.pkl'), 'wb')
            pickle.dump(fine_tune_predictions_dict,file )
            file.close()                
            
            with open(os.path.join(experiment_folder,'fine_tune_loss.json'), "w") as fp:
                json.dump(fine_tune_loss_dict, fp)  #         
                
            fig, axs, fine_tune_accuracy_df_output = PlotClassAccuracies(fine_tune_predictions_dict,label_mapping)
            fine_tune_accuracy_df_output.to_pickle(os.path.join(experiment_folder,'fine_tune_accuracy_df.pkl'))
            plt.savefig(os.path.join(experiment_folder,"fine_tune_class_accuracies_epochs.png"))                
            
            fig, axs = MetricPlots(fine_tune_metric_per_epoch,NUM_CLASSES=NUM_CLASSES)
            plt.savefig(os.path.join(experiment_folder,"fine_tune_train_accuracy.png"))
                    
                # Save the fine-tuned model if it improves
                # if fine_tune_loss < best_fine_tune_loss:
                #     best_fine_tune_loss = fine_tune_loss
                #     fine_tune_counter = 0
                #     torch.save(
                #         {
                #             "epoch": epoch,
                #             "model_state_dict": model.state_dict(),
                #             "optimizer_state_dict": fine_tune_optimizer.state_dict(),
                #             "fine_tune_loss": fine_tune_loss,
                #             "fine_tune_scores": fine_tune_scores,
                #         },
                #         os.path.join(experiment_folder, model_name + "_fine_tuned_best_model.pt"),
                #     )
                #     print(f"Saved fine-tuned model at epoch {epoch + 1}")
                # else:
                #     fine_tune_counter += 1
            
                # # Early stopping for fine-tuning
                # if fine_tune_counter >= fine_tune_patience:
                #     print("Early stopping for fine-tuning!")
                #     break
            
            #print("Fine-tuning complete. Best fine-tune loss:", best_fine_tune_loss)
            
            # Save final fine-tuned model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fine_tune_loss": fine_tune_loss,
                    "fine_tune_scores": fine_tune_scores,
                },
                os.path.join(experiment_folder, model_name + "_fine_tuned_final_model.pt"),
            )    
    
        #create_hypnograms(model = model,dataloader=sub_test_dataset,experiment_folder = os.path.join(experiment_folder,'testing'),NUM_CLASSES=NUM_CLASSES,batch_size=batch_size)
        """
    with open(os.path.join(experiment_folder_root,'k_fold_tracking.json'), "w") as fp:
        json.dump(k_fold_tracking, fp)  #   

    kfold_resuts = {'training':{},'test':{}}
    
    for mode in ['training','test']:
        for fold in k_fold_tracking:
            scores = {}
            for metric,metric_list in k_fold_tracking[fold]['scores'][mode].items():
                scores[metric] = metric_list[-1]
            kfold_resuts[mode][fold] = scores
        
    df_train_scores = pd.DataFrame(kfold_resuts['training']).transpose().reset_index(names='subject')
    df_test_scores = pd.DataFrame(kfold_resuts['test']).transpose().reset_index(names='subject')
    
    
    mean_row_train = df_train_scores.mean(numeric_only=True)
    mean_row_train['subject'] = 'mean'
    
    # Convert the mean row to a DataFrame and append it to the existing DataFrame
    df_train_scores = pd.concat([df_train_scores, pd.DataFrame(mean_row_train).T], ignore_index=True)
    
    
    mean_row_test = df_test_scores.mean(numeric_only=True)
    mean_row_test['subject'] = 'mean'
    
    # Convert the mean row to a DataFrame and append it to the existing DataFrame
    df_test_scores = pd.concat([df_test_scores, pd.DataFrame(mean_row_test).T], ignore_index=True)
    
    df_train_scores.to_csv(os.path.join(experiment_folder_root,'train_scores.csv'))
    df_test_scores.to_csv(os.path.join(experiment_folder_root,'test_scores.csv'))
    
    

