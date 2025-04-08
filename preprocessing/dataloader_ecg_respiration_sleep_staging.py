import sys
sys.path
sys.path.append('./code')
sys.path.append('./model_library')
sys.path.append('./preprocessing')
import torch
import pandas as pd
import scipy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import os
import glob

import time
from tqdm import tqdm

from sklearn.utils import shuffle
from segment_signal import *
from mymodel import *
import matplotlib.pyplot as plt
import pickle
from dataloader_utilities import getLabelMappingsFromPaper, CollapseSleepStagesFromPaper



import random
random.seed(42)


class ECGPeakMask(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_path:str, shuffle_recording = False, number_of_sleep_stage:int = 5,augment_data:bool = False,normalize:bool=True,window_length_in_min:int=5, hz = 256,get_ibi:bool=False,resample_ibi = False, ibi_frequency = 4, normalize_ibi = False):
        self.file_path = file_path
        self.subject_name = file_path.split(os.sep)[-1]
        self.shuffle_recording = shuffle_recording
        self.number_of_sleep_stage = number_of_sleep_stage
        self.augment_data = augment_data
        self.normalize_bool = normalize
        self.normalize_ibi = normalize_ibi
        self.ibi_frequency = ibi_frequency
        self.resample_ibi = resample_ibi
        self.hz = hz
        self.window_length_in_min = window_length_in_min
        self.get_ibi = get_ibi
        self.feature_df = None
        self.data = None
        self.label_dict = self.__getLabelDict()
        self.processed_ECG_dataset,self.labels = self.__prepareECGData()
    def __len__(self):
        #print("number_of_index:", self.processed_ECG_dataset.shape[0])
        return self.processed_ECG_dataset.shape[0]

    def __getitem__(self,idx):
        x = self.processed_ECG_dataset[idx]
        y = self.labels[idx]
        return x, y, self.subject_name
        
    def __prepareECGData(self):
        #return [self.__createXandY(path) for path in self.filenames]
        x, y = self.__createXandY(self.file_path)

        #if self.get_ibi == False:
        x = np.array(x,dtype=np.float64)
        y = np.array(y,dtype=np.float64)
        #row, col = x.shape
        x = torch.from_numpy(x).float()
        #x = torch.reshape(x,(row,1,col))
        y = torch.from_numpy(y).float()
        y = y.type(torch.LongTensor)        
        
        return x, y


        
    def __createXandY(self,path):
        # Read file
        data = pd.read_pickle(path)
        data = self.__converSleepStage4To3(data)
        self.data = data
        x = data.drop('labels', axis=1).values
        print("initial size",x.shape)
  
        y = data['labels'].values
        y = self.__collapseSleepStageOld(y)
        y = self.__new_mapping(y)
        y = self.__collapseSleepStage(y)
        
        y_temp = y.tolist()
        #y1 = y2*[30]
        new_list = [[_x]*len(x[0]) for _x in y_temp]
        flat_list = [item for sublist in new_list for item in sublist]
        labels = np.array(flat_list)
        #x = df.values
        x = x.flatten()
        print("flat shape",x.shape)
        window_time = 4.5*60 # [s]
        step_time = 30 # [s]
        notch_freq = [50, 60]  # [Hz]
        bandpass_freq = [None, 4.]  # [Hz]
        n_gpu = 1
        Fs = 256    
        #stages = None
        x = x.reshape(1,-1)*1e3
        print("last shape",x.shape)
        #exit()
        ecg_segs, labels , _, _ = segment_ecg_signal(x, labels, window_time, step_time, Fs, newFs=200, start_end_remove_window_num=0, n_jobs=-1, amplitude_thres=6000)
            
        # Create one-hot encoding for label
        integer_encoded = labels.reshape(len(labels), 1)         
        return ecg_segs, integer_encoded

    
    def __collapseSleepStageOld(self,y):
        y[y==5] = 4
        return y    
    
    def __converSleepStage4To3(self, df):
        df["labels"][df["labels"]==4] = 3
        return df
    
    def __new_mapping(self,y):
            # 0 = Wake  ==> 5
            # 1 = Stage 1 Sleep ==> 3
            # 2 = Stage 2 Sleep ==> 2
            # 3 = Stage 3 Sleep ==> 1
            # 4 = Stage 4 Sleep ==> 1
            # 5 = REM Sleep ==> 4            
        
        # Define the original NumPy array

        
        # Define the mapping dictionary
        mapping_dict = {0:5, 1:3, 2:2, 3:1}
        
        # Create a function that maps the values using the dictionary
        mapping_function = np.vectorize(lambda x: mapping_dict.get(x, x))
        
        # Apply the mapping function to the original array
        mapped_array = mapping_function(y)
        
        # Output the mapped array
        print(mapped_array)
                    
        return mapped_array
    
    def __getLabelDict(self):
        #tick_name = "ticks"
        #dict_name = ""
        label_dict = getLabelMappingsFromPaper(number_of_sleep_stage = self.number_of_sleep_stage)
        return label_dict     
    
    def __collapseSleepStage(self,y):
        y = y-1
        y = CollapseSleepStagesFromPaper(y, self.number_of_sleep_stage)
        return y      

class ECGPeakMask2(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_path:str, shuffle_recording = False, number_of_sleep_stage:int = 5,augment_data:bool = False,normalize:bool=True,window_length_in_min:int=5, hz = 256,get_ibi:bool=False,resample_ibi = False, ibi_frequency = 4, normalize_ibi = False):
        self.file_path = file_path
        self.subject_name = file_path.split(os.sep)[-1]
        self.shuffle_recording = shuffle_recording
        self.number_of_sleep_stage = number_of_sleep_stage
        self.processed_ECG_dataset,self.labels = self.__prepareECGData()
    def __len__(self):
        #print("number_of_index:", self.processed_ECG_dataset.shape[0])
        return self.processed_ECG_dataset.shape[0]

    def __getitem__(self,idx):
        x = self.processed_ECG_dataset[idx]
        y = self.labels[idx]
        return x, y, self.subject_name
        
    def __prepareECGData(self):
        #return [self.__createXandY(path) for path in self.filenames]
        with open(self.file_path, 'rb') as handle:
            d = pickle.load(handle) 
        x = d["X"]
        y = d['Y']-1
        
        return x, y

if __name__ == '__main__':


    
    # Generators  
    training_path_mesa = os.path.join("non-augmented","processed_data_train",'mesa',"*")
    training_fileNames_mesa = glob.glob(training_path_mesa)
    ground_truth = []
    predicted = []
    cnn_predicted = []
    #training_fileNames_mesa = [training_fileNames_mesa[0]]
    
    for file in training_fileNames_mesa[0:20]:

        one_recording = ECGPeakMask(file,number_of_sleep_stage=5,hz=256)
        #one_recording = ECGPeakMask2(file,number_of_sleep_stage=5,hz=256)
        #generator = torch.utils.data.DataLoader(one_recording,batch_size=24,drop_last=True)  
        #for batch_ndx, (local_batch, local_labels, train_file_name) in enumerate(tqdm(generator,position=0,leave=True)):    
        #    print(batch_ndx)
        
        #df = one_recording.data
        X = one_recording.processed_ECG_dataset
        y = one_recording.labels
        data_source = 'ECG'
        '''        
        trained_model_paths = {
                'ECG':'F:\OneDrive - Aalborg Universitet\Dokumenter\PhD\ecg_respiration_sleep_staging-master\models\CNN_ECG_fold1.pth',
                'ABD':'models/CNN_ABD_fold1.pth',
                'CHEST':'models/CNN_CHEST_fold1.pth' }
        # load ECG model
        ecg_cnn_model = ECGSleepNet()
        ecg_cnn_model.load_state_dict(th.load(trained_model_paths['ECG']))
        n_gpu = 1
        if n_gpu>0:
            ecg_cnn_model = ecg_cnn_model.cuda()
            if n_gpu>1:
                ecg_cnn_model = nn.DataParallel(ecg_cnn_model, device_ids=list(range(n_gpu)))
        ecg_cnn_model.eval()
        ecg_rnn_model = SleepNet_RNN(1280, 5, 20, 2, dropout=0, bidirectional=True)
        ecg_rnn_model.load_state_dict(th.load('F:\OneDrive - Aalborg Universitet\Dokumenter\PhD\ecg_respiration_sleep_staging-master\models\LSTM_%s_fold1.pth'%data_source))
        if n_gpu>0:
            ecg_rnn_model = ecg_rnn_model.cuda()
            if n_gpu>1:
                ecg_rnn_model = nn.DataParallel(ecg_rnn_model, device_ids=list(range(n_gpu)))
        ecg_rnn_model.eval()
        
        # feed to ECG model
        #X = np_to_var(ecg_segs.astype('float32'))
        if n_gpu>0:
            X = X.cuda()
        with th.no_grad():
            ids = np.array_split(np.arange(len(X)), 50)
            H = []
            ycnn = []
            for id_ in ids:
                _, H_ = ecg_cnn_model(X[id_])
                ycnn.append(np.argmax(_.cpu().data.numpy(), axis=1)+1)
                
                H.append(H_)
            H = th.cat(H, dim=0)
            H = H.reshape(1, H.shape[0], -1)
            yp, _ = ecg_rnn_model(H)
        yp = yp[0].cpu().data.numpy()
        yp = np.argmax(yp, axis=1)+1  
        ground_truth.append(y.cpu().data.numpy())
        predicted.append(yp)
        ycnn_flat = np.array([item for sublist in ycnn for item in sublist])
        cnn_predicted.append(ycnn_flat)
        
        plt.figure()
        plt.plot(y.cpu().data.numpy(),label="Ground Truth")
        plt.plot(yp,label ='cnnlstm predicted')
        plt.legend()
        plt.title(one_recording.subject_name)
        plt.savefig(one_recording.subject_name+"cnnlstm.png")
        
        plt.figure()
        plt.plot(y.cpu().data.numpy(),label="Ground Truth")
        plt.plot(ycnn_flat,label ='cnn predicted')
        plt.legend()
        plt.title(one_recording.subject_name)
        plt.savefig(one_recording.subject_name+"cnn.png")        
    
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    
    acc = []
    f1 = []
    kappa = []
    for pre,gt in zip(predicted,ground_truth):
        acc.append(accuracy_score(pre,gt))
        f1.append(f1_score(pre,gt,average='micro'))
        kappa.append(cohen_kappa_score(pre,gt))
    print("CNN+LSTM","Mean acc: ",np.mean(acc),"\nmean F1: ",np.mean(f1),"\nmean Kappa: ", np.mean(kappa))  
    
    for pre,gt in zip(cnn_predicted,ground_truth):
        acc.append(accuracy_score(pre,gt))
        f1.append(f1_score(pre,gt,average='micro'))
        kappa.append(cohen_kappa_score(pre,gt))
    print("CNN","Mean acc: ",np.mean(acc),"\nmean F1: ",np.mean(f1),"\nmean Kappa: ", np.mean(kappa))       

    predicted_flat = [item for sublist in predicted for item in sublist]
    ground_truth_flat = [item for sublist in ground_truth for item in sublist]
    cnn_predicted_flat = [item for sublist in cnn_predicted for item in sublist]
    
    NUM_CLASSES = 5
    cm = confusion_matrix(ground_truth_flat,predicted_flat,normalize='true')
    plt.figure()
    #s = sns.heatmap(cm,annot=True,fmt='g')
    heat_map_ticks = ["NREM3","NREM2","NREM1","REM","Wake"]
    s = sns.heatmap(cm,annot=True,xticklabels=heat_map_ticks,yticklabels=heat_map_ticks)
    #s = sns.heatmap(cm,annot=True)
    s.set(xlabel='Predicted CNN+LSTM', ylabel='Actual') # source https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    plt.show    
    plt.savefig("cnnlstm_confusion_matrix.png")
    
    cm = confusion_matrix(cnn_predicted_flat,predicted_flat,normalize='true')
    plt.figure()
    #s = sns.heatmap(cm,annot=True,fmt='g')
    heat_map_ticks = ["NREM3","NREM2","NREM1","REM","Wake"]
    s = sns.heatmap(cm,annot=True,xticklabels=heat_map_ticks,yticklabels=heat_map_ticks)
    #s = sns.heatmap(cm,annot=True)
    s.set(xlabel='Predicted CNN', ylabel='Actual') # source https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    plt.show    
    plt.savefig("cnn_confusion_matrix.png")    
    # 0 = Wake  ==> 5
    # 1 = Stage 1 Sleep ==> 3
    # 2 = Stage 2 Sleep ==> 2
    # 3 = Stage 3 Sleep ==> 1
    # 4 = Stage 4 Sleep ==> 1
    # 5 = REM Sleep ==> 4            
    #mapping = {5:'Wake',3:'Sleep Stage 1',2:'Sleep Stage 2',1:'Sleep Stage 3',4:'REM'}
    """
    x = df.drop('labels', axis=1).values
  
    y = df['labels'].values    
    y_temp = y.tolist()
    #y1 = y2*[30]
    new_list = [[_x]*len(x[0]) for _x in y_temp]
    flat_list = [item for sublist in new_list for item in sublist]
    labels = np.array(flat_list)
    #x = df.values
    x = x.flatten()
    window_time = 4.5*60 # [s]
    step_time = 30 # [s]
    notch_freq = [50, 60]  # [Hz]
    bandpass_freq = [None, 4.]  # [Hz]
    n_gpu = 1
    Fs = 256    
    #stages = None
    x = x.reshape(1,-1)*1e3
    ecg_segs, labels , _, _ = segment_ecg_signal(x, labels, window_time, step_time, Fs, newFs=200, start_end_remove_window_num=0, n_jobs=-1, amplitude_thres=6000)
    """
    '''