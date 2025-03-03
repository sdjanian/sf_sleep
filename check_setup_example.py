import sys
sys.path
sys.path.append('./code')
sys.path.append('./model_library')


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
from deep_learning_loop_utilities import get_train_valid_test_files_recreate,get_train_valid_test_files_recreate_small,getSubjectPaths, getSubjectPathsContinuedTraining, getSFDLECGPPGPathPairs
from deep_learning_loop_utilities import create_datasets, getCrossEntropyWeights, get_optimzer_and_scheduler
from deep_learning_plots import create_loss_plot, create_confusion_matrix, create_hypnograms,getDistributionOfGeneratorLabels,DistributionPlot, MetricPlots, PlotMetricDistribution, PlotClassAccuracies, createBaselineComparisonPlot
from deep_learning_loop_utilities import calculate_metrics, gatherBatchDataToSubjectMetrics, getDataFrameOfRecordingMetric, getMetricPerEpoch
from deep_learning_loop_utilities import calculateBaselinePrClass,createBaselineAndPerformanceDf


from dataset_densenet import ECGDataSetSingle2
from ecg_mask_loader_simple import ECGPeakMask2
from data_loader_sleep_study_dataset import SFDL
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


    # Loads the files used for training and validating the model. If USING_CLAAUDIA is true it uses the full dataset, otherwise it uses a smaller subsample
    ##############################################################################################################

    example_files = glob.glob('./example_data/*')
    
    training_fileNames = example_files[0:2]
    validation_fileNames = example_files[2:4]
    test_fileNames = example_files[4:6]

    files_dict = {'training':training_fileNames,'validation':validation_fileNames,'testing':test_fileNames}
    #d = ECGDataSetSingle2(training_fileNames[0],shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = 250,resample_bool=resample_data,window_size=window_size,sampling_frequency=256,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    #d = ECGPeakMask2(training_fileNames[0],shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,light_sleep_vs_all_bool=light_sleep_vs_all_bool)

    for key, value in files_dict.items():
        print('Number of',key,'files:',len(value))
    

    sampling_frequency = 200
    NUM_CLASSES = 5
    normalize_data_bool = True
    AUGMENT_DATA = False
    normalize_type='zscore'
    resample_frequency = 64
    resample_data=False
    window_size=270
    light_sleep_vs_all_bool=False
    save_best_model_bool = True

    training_dataset = create_datasets(training_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool, augment_data = AUGMENT_DATA,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    validation_dataset = create_datasets(validation_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency = resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)
    test_dataset = create_datasets(test_fileNames,ECGDataSetSingle2,shuffle_recording=False, number_of_sleep_stage=NUM_CLASSES, normalize=normalize_data_bool,normalize_type=normalize_type,resample_frequency= resample_frequency,resample_bool=resample_data,window_size=window_size,sampling_frequency=sampling_frequency,light_sleep_vs_all_bool=light_sleep_vs_all_bool)

    batch_size = 16
    sampler = None
    shuffle_bool = True
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True,sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_bool,drop_last=True)

    

   
    n_timestep = resample_frequency*window_size # 64*270 = 17280 and 200*270=54000

    model = ECGSleepNetAdaptable(nb_classes = NUM_CLASSES,n_timestep=n_timestep)
    model_name = "ECGSleepNetAdaptable"           
    
    model_paramaters = {"NUM_CLASSES":NUM_CLASSES,
                        "n_timestep":n_timestep}  

    loss_fn = nn.CrossEntropyLoss() 
        

    optimizer, scheduler = get_optimzer_and_scheduler(model,lr=0.001,decay=0.001)

    print('Everything runs correctly"')


        
        
    #Create the folder to save the model, epoch and performance data. Also instanciates the loss dict and saves the commandline arguments
    experiment_folder = experiment_folder_path(subfolder='example'+
                                               model_name)
    loss_dict = {'training':[],'validation':[],'test':[],'control_training':[]}
    subject_metric = {'training':[],'validation':[],'test':[],'control_training':[]}
    subject_predictions = {'training':[],'validation':[],'test':[],'control_training':[]}

    epoch_starting_number=0
    best_validation_loss = 100

    
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
    n_epochs = 3
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
    