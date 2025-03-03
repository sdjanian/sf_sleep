import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os


import sys
sys.path.append('..\data_loader')
sys.path.append('..\ecg_respiration_sleep_staging-master')
sys.path.append('/home/cs.aau.dk/fq73oo/data_loader/ecg_respiration_sleep_staging-master/training_code_still_messy')
sys.path.append('/home/cs.aau.dk/fq73oo/data_loader')

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns



def plot_hypnogram(y_true:list,y_pred:list,recording:str="",save_path:str="",number_of_sleep_stages:int=5,folder_name:str="hypnogram_validation",label_mapping_dict:dict={}):
    """
    Makes two hypnograms fron a list of sleep stage labels. One for the ground truth and one for the predicted. Stores them under a folder that by default is named 'hypnogram_validation'

    Parameters
    ----------
    y_true : list
        DESCRIPTION.
    y_pred : list
        DESCRIPTION.
    recording : str, optional
        DESCRIPTION. The default is "".
    save_path : str, optional
        DESCRIPTION. The default is "".
    number_of_sleep_stages : int, optional
        DESCRIPTION. The default is 5.
    folder_name : str, optional
        DESCRIPTION. The default is "hypnogram_validation".

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    hypno_gram_folder_path = os.path.join(save_path,folder_name)
    os.makedirs(hypno_gram_folder_path,exist_ok=True) 
    """
    if number_of_sleep_stages == 5:
        ytick = [0,1,2,3,4]
        ytick_labels = ["Wake","NREM1","NREM2","NREM3","REM"]
    if number_of_sleep_stages == 4:
        ytick = [0,1,2,3]
        ytick_labels = ["Wake","Light Sleep","Deep Sleep","REM"]
    if number_of_sleep_stages == 3:
        ytick = [0,1,2]
        ytick_labels = ["Wake","NREM","REM"]
    if number_of_sleep_stages == 2:
        ytick = [0,1]
        ytick_labels = ["Wake","Sleep"]
    """
    ytick = label_mapping_dict["ticks"]["tick"]
    ytick_labels=label_mapping_dict["ticks"]["label"]
   
    fig, axs = plt.subplots(2,1,figsize=(16,9),sharex=True)
    fig.suptitle("Hypnogram of recording: "+recording)
    axes = axs.flat
    axes[0].plot(np.arange(0,len(y_true)),y_true,label="Ground Truth",color="b")
    axes[0].set_yticks(ytick,labels=ytick_labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Ground Truth")
    
    axes[1].plot(np.arange(0,len(y_pred)),y_pred,label="Predicted",color="r")
    axes[1].set_yticks(ytick,labels=ytick_labels)
    axes[1].invert_yaxis()
    axes[1].set_title("Predicted")
    plt.xlabel("Sleep Epoch")
    fig.supylabel("Sleep Stage")             
   
    print("\n",hypno_gram_folder_path+"_hypnogram_"+recording+".png")
    plt.savefig(os.path.join(hypno_gram_folder_path,"hypnogram_"+recording+".png"))
    plt.close()
    return fig  


def create_confusion_matrix(data_set,model,title:str='Confusion Matrix'):
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False,drop_last=True) 
    model.eval()
    predicted = []
    ground_truth = []    
    try:
        heat_map_ticks = data_set.datasets[0].label_dict["ticks"]["label"]
    except:
        try: 
            heat_map_ticks = data_set.dataset.datasets[0].label_dict["ticks"]["label"]
        except:
            heat_map_ticks = data_set.label_dict["ticks"]["label"]
        
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(data_loader),desc=title):
            vinputs, gt, vname = vdata
            gt = gt.squeeze()
            if torch.cuda.is_available():   
                gt = gt.to('cuda')
                vinputs = vinputs.to('cuda')
            pred_output = model(vinputs)
            if type(pred_output)==tuple:
                pred_output, H = pred_output
            pred_output = pred_output.squeeze()
            predicted.append(np.argmax(pred_output.cpu().detach().numpy()))
            ground_truth.append(gt.detach().cpu().numpy())
            
    predict_flat = predicted#np.concatenate(predicted)
    ground_truth_flat =ground_truth# np.concatenate(ground_truth)
    #print(predict_flat)

    cm = confusion_matrix(ground_truth_flat,predict_flat,normalize='true')
    plt.figure()
    #s = sns.heatmap(cm,annot=True,fmt='g')
    s = sns.heatmap(cm,annot=True,xticklabels=heat_map_ticks,yticklabels=heat_map_ticks)
    s.set(xlabel='Predicted', ylabel='Actual') # source https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    plt.title(title)
    #plt.show    
    #plt.savefig(os.path.join(save_folder,'test_confusion_matrix.png'))    
    return s, {'predicted':predict_flat,'ground_truth':ground_truth_flat}

def create_loss_plot(loss_dict):
    plt.figure()
    max_lim = np.max(loss_dict["training"])+2
    plt.plot(loss_dict["training"],'-o')
    plt.plot(loss_dict["validation"],'-o')
    plt.plot(loss_dict["test"],'-o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(['Train','Valid','Test'])    
    ylims = plt.gca().get_ylim()
    if ylims[1] > max_lim:
        plt.ylim(0, max_lim)      
    return plt.gca()

def create_hypnograms(model, dataloader,experiment_folder,NUM_CLASSES,batch_size=6):    
    for dataset in tqdm(dataloader.datasets,total=len(dataloader.datasets),desc="creating hypnograms",leave=True):
        """
        if USE_FEATURES == False:
            subject_dataset = ECGDataSetSingle(validation_file,shuffle_recording=False,number_of_sleep_stage=NUM_CLASSES,augment_data=False)
        elif USE_IBI == True:
            subject_dataset.append(HRVFeaturesDataset(validation_file,number_of_sleep_stage=NUM_CLASSES,hz=256,window_length_in_min=5,get_ibi=True,resample_ibi=True,normalize_ibi=True)) 
        else:
            subject_dataset = HRVFeaturesDataset(validation_file,number_of_sleep_stage=NUM_CLASSES,window_length_in_min=5)
        """    
        subject_generator = torch.utils.data.DataLoader(dataset,batch_size=batch_size,drop_last=False) 
        y_pred = []
        y_true = []
        
        for local_batch, local_labels, file_name in tqdm(subject_generator,desc="current hypnogram",leave=False,position=0):        
            if torch.cuda.is_available():   
                #inputs, labels = inputs.to(device), labels.to(device)   
                local_batch, local_labels = local_batch.to('cuda'), local_labels.to('cuda')                  
            subject_name = file_name[0].split(".")[0]
            local_labels = local_labels.squeeze()    
            outputs = model(local_batch)
            if type(outputs) == tuple:
                outputs, _h = outputs # unfold the tuple to contain the prediction and the last layer before          
            #val_forward_pass_outputs = model(val_local_batch)
            _, predicted = outputs.max(1)
            
            #validation_labels_flat = [item for sublist in val_local_labels.cpu().detach().numpy().tolist() for item in sublist]
            y_pred.append(predicted.cpu().detach().numpy().tolist())
            y_true.append(local_labels.cpu().detach().numpy().tolist())

        # Sometimes the last batch is an integer instead of a list. This ensures that it a list
        def check_if_all_is_list(lst):
            lst2 = [x if type(x) is list else [x] for x in lst]
            return lst2
        y_pred = check_if_all_is_list(y_pred)
        y_true = check_if_all_is_list(y_true)
       
        y_pred = [item for sublist in y_pred for item in sublist]
        y_true = [item for sublist in y_true for item in sublist]
        
        try:
            label_mapping_dict = dataset.datasets[0].label_dict
        except:
            label_mapping_dict = dataset.label_dict        
        ax = plot_hypnogram(y_true,y_pred,recording=subject_name,save_path=experiment_folder,number_of_sleep_stages=NUM_CLASSES,label_mapping_dict=dataset.label_dict)  
    return None

def getDistributionOfGeneratorLabels(generator):
    
    augment_label_list = []
    for batch, labels, file_name in tqdm(generator,total=len(generator)):
        
        #if torch.cuda.is_available():   
        #    batch, labels = batch.to('cuda'), labels.to('cuda')  
        
        #batch = batch.squeeze(0)
        labels = labels.squeeze()
        augment_label_list.append(labels.cpu().detach().numpy())
    
    
    augment_unique, augment_counts = np.unique(np.concatenate(augment_label_list), return_counts=True)  
    print('distribution: ', augment_unique, augment_counts)
    d = dict(zip(augment_unique, augment_counts))
    print(d)
    return d
def DistributionPlot(distribution, title:str='training',label_mapping_dict={}):
    fig = plt.figure()
    distribution_melted = distribution.melt(value_vars=distribution.columns, var_name='Bar', value_name='Height')
    sns.barplot(x='Bar', y='Height',data = distribution_melted)
    #plt.bar(distribution)
    if title == 'training':
        plt.title("Training Distribution")
    elif title=='validation':
        plt.title("Validation Distribution")
    elif title=='test':
        plt.title("Test Distribution")
    else:
        plt.title(title)
        
    plt.xlabel("Sleep Stages")
    """
    if NUM_CLASSES == 5:
        tick = [0,1,2,3,4]
        tick_labels = ["Wake","NREM1","NREM2","NREM3","REM"]
    if NUM_CLASSES == 4:
        tick = [0,1,2,3]
        tick_labels = ["Wake","Light Sleep","Deep Sleep","REM"]
    if NUM_CLASSES == 3:
        tick = [0,1,2]
        tick_labels = ["Wake","NREM","REM"]
    if NUM_CLASSES == 2:
        tick = [0,1]
        tick_labels = ["Wake","Sleep"]
    """
    tick = label_mapping_dict["ticks"]["tick"]
    tick_labels=label_mapping_dict["ticks"]["label"]
    plt.gca().set_xticks(tick,labels=tick_labels)
    plt.ylabel("Frequency Count")
    return fig
def MetricPlots(metric_per_epoch,NUM_CLASSES):
    fig, axs = plt.subplots(3,sharey=True,figsize=(12,9))
    axs[0].plot(metric_per_epoch["training"]["accuracy"],'-o', label = "Train")
    axs[0].plot(metric_per_epoch["validation"]["accuracy"],'-o',label = "Validation")
    axs[0].plot(metric_per_epoch["test"]["accuracy"],'-o',label = "Test")
    axs[0].set_title('Accuracy')

    axs[1].plot(metric_per_epoch["training"]["f1_score"],'-o', label = "Train")
    axs[1].plot(metric_per_epoch["validation"]["f1_score"],'-o',label = "Validation")
    axs[1].plot(metric_per_epoch["test"]["f1_score"],'-o',label = "Test")
    axs[1].set_title('F1-score')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_ylabel('Evaluation Metric')

    axs[2].plot(metric_per_epoch["training"]["cohen_kappa"],'-o', label = "Train")
    axs[2].plot(metric_per_epoch["validation"]["cohen_kappa"],'-o',label = "Validation")
    axs[2].plot(metric_per_epoch["test"]["cohen_kappa"],'-o',label = "Test")
    axs[2].set_title('Cohens Kappa')
    plt.ylim([0,1])
    plt.xlabel('Epoch')
    plt.suptitle('Train vs Valid Metrics {} Classes'.format(NUM_CLASSES))  
    plt.tight_layout()
    return fig, axs

def PlotMetricDistribution(subject_metric):
    train_df = pd.DataFrame(subject_metric['training'][-1]).T
    train_df['type'] = 'Training'
    valid_df = pd.DataFrame(subject_metric['validation'][-1]).T
    valid_df['type'] = "Validation"
    test_df = pd.DataFrame(subject_metric['test'][-1]).T
    test_df['type'] = "Test"    
    subject_df = pd.concat([train_df,valid_df,test_df],axis=0)
    subject_df_long = pd.melt(subject_df, id_vars='type',var_name=['Metric'],value_name='Score')
    plt.figure()
    g = sns.violinplot(data=subject_df_long, x="Metric", y="Score", hue="type",inner="stick")
    g.legend_.set_title(None)
    g.set_title('Distribution of metrics')
    return g    

def PlotClassAccuracies(subject_predictions,label_mapping):
    """
    def GetClassAccuracies(__subject_predictions,__label_mapping):
        epoch_list = []
        for idx,df in enumerate(__subject_predictions):
            class_accuracy_list = []
            #print(df)
            for subject in tqdm(df['subject'].unique(),total=len(df['subject'].unique()),leave=False):
                sub_df = df[df['subject']==subject]
                temp_matrix = confusion_matrix(sub_df['labels'], sub_df['predicted'],normalize='true',labels=__label_mapping['ticks']['tick'])
                subject_class_accuracies = temp_matrix.diagonal()
                subject_class_accuracies = np.append(subject,subject_class_accuracies)
                class_accuracy_list.append(subject_class_accuracies)
            class_accuracy_df = pd.DataFrame(np.vstack(class_accuracy_list),columns=["Subject"]+__label_mapping['ticks']['label'])
            class_accuracy_df["Epoch"] = idx
            class_accuracy_df_long = pd.melt(class_accuracy_df, id_vars=["Epoch","Subject"],var_name=['Class'],value_name='Accuracy')
            
            epoch_list.append(class_accuracy_df_long)
        epoch_df = pd.concat(epoch_list,axis=0)
        epoch_df['Accuracy'] = epoch_df['Accuracy'].astype('float64')
        return epoch_df    
    """
    def GetClassAccuracies(__subject_predictions,__label_mapping):
        epoch_list = []
        for idx,df in tqdm(enumerate(__subject_predictions),total =len(__subject_predictions),leave=False):
            cm_diag_df = df.groupby('subject').apply(lambda x: confusion_matrix(x['labels'],x['predicted'],normalize='true',labels=__label_mapping['ticks']['tick']).diagonal())
            cm_diag_np = np.vstack(cm_diag_df.values)
            subj = np.array(cm_diag_df.index.values).reshape(-1,1)
            cm_diag_with_subj_np = np.append(subj,cm_diag_np,axis=1)
            class_accuracy_df = pd.DataFrame(cm_diag_with_subj_np,columns=["Subject"]+label_mapping['ticks']['label'])
            class_accuracy_df["Epoch"] = idx
            class_accuracy_df_long = pd.melt(class_accuracy_df, id_vars=["Epoch","Subject"],var_name=['Class'],value_name='Accuracy')
            
            epoch_list.append(class_accuracy_df_long)
        epoch_df = pd.concat(epoch_list,axis=0)
        epoch_df['Accuracy'] = epoch_df['Accuracy'].astype('float64')
        return epoch_df    

    fig,ax = plt.subplots(3, sharex=True,sharey=True)
    training_df = GetClassAccuracies(subject_predictions['training'],label_mapping)
    validation_df = GetClassAccuracies(subject_predictions['validation'],label_mapping)
    test_df = GetClassAccuracies(subject_predictions['test'],label_mapping)
    sns.lineplot(data=training_df, x="Epoch", y="Accuracy", hue="Class",ax=ax[0])
    sns.lineplot(data=validation_df, x="Epoch", y="Accuracy", hue="Class",ax=ax[1])
    sns.lineplot(data=test_df, x="Epoch", y="Accuracy", hue="Class",ax=ax[2])
    ax[0].set_title('Train')
    ax[1].set_title('Validation')
    ax[2].set_title('Test')
    ax[0].set_ylim([0,1])
    ax[0].set_ylabel(None)
    ax[2].set_ylabel(None)
    ax[0].legend().set_visible(False)
    ax[2].legend().set_visible(False)
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    from matplotlib.ticker import MaxNLocator
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))        

    training_df['Type'] = 'train'
    validation_df['Type'] = 'validation'
    test_df['Type'] = 'test'
    output_df = pd.concat([training_df,validation_df,test_df])
    return fig, ax, output_df

def createBaselineComparisonPlot(df_long):
    unique_metrics = df_long['metric'].unique()
    
    # Create subplots
    fig, axes = plt.subplots(nrows=len(unique_metrics), ncols=1, figsize=(10, 6*len(unique_metrics)))
    
    # Plot each metric
    for i, metric in enumerate(unique_metrics):
        sns.barplot(data=df_long[df_long['metric'] == metric], x="model", y="value", hue="dataset", ax=axes[i])
        axes[i].set_ylabel(metric)
        if i == (len(unique_metrics) - 2):  # Middle subplot
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))    
        else:
            axes[i].get_legend().remove()
        axes[i].set_xlabel("")    
        
    plt.suptitle('Metric of model vs baseline assumption')
    plt.tight_layout()
    return fig, axes    