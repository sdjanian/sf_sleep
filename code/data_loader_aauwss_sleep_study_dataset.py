import torch
import pandas as pd
import scipy
import numpy as np
from scipy import signal
#import scipy
import os
import glob
from sklearn.utils import shuffle
import random
random.seed(42)
from scipy.stats import zscore
from dataset_densenet import ECGDataSetSingle2
from dataloader_utilities import getLabelMappings, LightSleepVsAllSleepStageCollapse 

from segment_signal import segment_ecg_signal
from biosppy.signals import ecg, bvp




class AAUWSSDL(ECGDataSetSingle2):
    def __init__(self, file_path,ecg_file_path:str = None,ppg_file_path:str=None, shuffle_recording = False, number_of_sleep_stage:int = 5,augment_data:bool = False,normalize:bool=True,resample_bool:bool = False,resample_frequency:int = 0,normalize_type = 'zscore',sampling_frequency=None,window_size = None,drop_initial_wake=False,light_sleep_vs_all_bool:bool = False,signal_source:str='both',mask:bool = True,trim_ends:bool=True,filter_bool:bool = False,invert_ecg_bool:bool = False):
        """
        

        Parameters
        ----------
        file_path : TYPE
            DESCRIPTION.
        ecg_file_path : str, optional
            DESCRIPTION. The default is None.
        ppg_file_path : str, optional
            DESCRIPTION. The default is None.
        shuffle_recording : TYPE, optional
            DESCRIPTION. The default is False.
        number_of_sleep_stage : int, optional
            DESCRIPTION. The default is 5.
        augment_data : bool, optional
            DESCRIPTION. The default is False.
        normalize : bool, optional
            DESCRIPTION. The default is True.
        resample_bool : bool, optional
            DESCRIPTION. The default is False.
        resample_frequency : int, optional
            DESCRIPTION. The default is 0.
        normalize_type : TYPE, optional
            DESCRIPTION. The default is 'zscore'.
        sampling_frequency : TYPE, optional
            DESCRIPTION. The default is None.
        window_size : TYPE, optional
            DESCRIPTION. The default is None.
        drop_initial_wake : TYPE, optional
            DESCRIPTION. The default is False.
        light_sleep_vs_all_bool : bool, optional
            DESCRIPTION. The default is False.
        signal_source : str, optional
            DESCRIPTION. The default is 'both'.
        mask : bool, optional
            DESCRIPTION. The default is True.
        trim_ends : bool, optional
            DESCRIPTION. The default is True.
        filter_bool : bool, optional
            DESCRIPTION. The default is False.
        invert_ecg_bool : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if type(file_path)==tuple and signal_source=='both':
            self.ecg_file_path = file_path[0]
            self.ppg_file_path = file_path[1]
            self.subject_name = self.ecg_file_path.split(os.sep)[-1].split('_ecg')[0]

        elif signal_source == 'ecg':
            self.ecg_file_path = file_path
            self.file_path = self.ecg_file_path
            self.subject_name = self.ecg_file_path.split(os.sep)[-1].split(".")[0]

        elif signal_source == 'ppg':
            self.ppg_file_path = file_path
            self.subject_name = self.ppg_file_path.split(os.sep)[-1].split(".")[0]



            
        #self.ecg_file_path = ecg_file_path
        #self.file_path = self.ecg_file_path
        #self.ppg_file_path = ppg_file_path

        #self.subject_name = ecg_file_path.split(os.sep)[-1]
        self.shuffle_recording = shuffle_recording
        self.number_of_sleep_stage = number_of_sleep_stage
        self.augment_data = augment_data
        self.normalize_bool = normalize
        self.resample_bool = resample_bool
        self.resample_frequency = resample_frequency
        self.sampling_frequency = sampling_frequency
        self.normalize_type = normalize_type
        self.light_sleep_vs_all_bool = light_sleep_vs_all_bool
        self.filter_bool = filter_bool
        self.raw_df = None
        self.drop_initial_wake_bool = drop_initial_wake
        self.window_size = window_size
        self.label_dict = self._getLabelDict()
        self.signal_source = signal_source
        self.getitm_output = self.signal_source
        self.mask = mask
        self.trim_ends_bool = trim_ends
        self.invert_ecg_bool = invert_ecg_bool
        #signal, integer_encoded_label = self._loadPickleData(self.ecg_file_path)
        self._signal_source_preparation_routine()
        self._remove_epochs_without_labels()
        self._trim_end_epochs_to_equal_lengths()
        self._filter_signals()
        self._remove_signal_rows_with_nans()
        if signal_source=='both':
            self.labels = self.ecg_labels
        #self.processed_ECG_dataset,self.labels = self._prepareECGData(signal,integer_encoded_label)
    def __len__(self):
        #print("number_of_index:", self.processed_ECG_dataset.shape[0])
        return self.processed_ECG_dataset.shape[0]

    def __getitem__(self,idx):
        if self.getitm_output == "ppg":
            return self.processed_PPG_dataset[idx], self.ppg_labels[idx], self.subject_name    

            
        if self.getitm_output == "ecg":
            return self.processed_ECG_dataset[idx], self.ecg_labels[idx], self.subject_name    

        if self.getitm_output == "both":
            if self.ecg_labels[idx] != self.ppg_labels[idx]:
                raise Exception(f"ecg and ppg labels do not match. ECG: {self.ecg_labels[idx]} PPG: {self.ppg_labels[idx]}")
            else:
                return (self.processed_ECG_dataset[idx],self.processed_PPG_dataset[idx]), self.ppg_labels[idx], self.subject_name    

                
    def _remove_epochs_without_labels(self):
        # Looks for labels that are not in the label dictionary and drops them from the the label vector and the signal data. This is to remove the "Missing" labels where parts of the dataset is unlabelled
        correct_labels = torch.tensor(list(self.label_dict['mapping'].values()))
        
        
        if self.signal_source == 'ecg' or self.signal_source == 'both':
            ecg_mask = torch.isin(self.ecg_labels,correct_labels,invert=True)
            self.ecg_labels = self.ecg_labels[~ecg_mask]
            self.processed_ECG_dataset = self.processed_ECG_dataset[~ecg_mask]
            
        if self.signal_source == 'ppg' or self.signal_source == 'both':
            ppg_mask = torch.isin(self.ppg_labels,correct_labels,invert=True)
            self.ppg_labels = self.ppg_labels[~ppg_mask]
            self.processed_PPG_dataset = self.processed_PPG_dataset[~ppg_mask]
            
    def _remove_signal_rows_with_nans(self):
        # When creating windows it will use the nans from the parts of the signal that is missing. Since there is no actual signal these rows have to be dropped, otherwise the neural network make everything to NaNs.
        # Find the rows with nans
        if self.signal_source == 'ecg' or self.signal_source == 'both':
            nan_in_rows_ecg = torch.isnan(self.processed_ECG_dataset).any(dim=2).squeeze()


        if self.signal_source == 'ppg' or self.signal_source == 'both':
            nan_in_rows_ppg = torch.isnan(self.processed_PPG_dataset).any(dim=2).squeeze()

            
        if self.signal_source == 'both':
            nan_vector = nan_in_rows_ecg+nan_in_rows_ppg
        elif self.signal_source == 'ecg':
            nan_vector = nan_in_rows_ecg
        elif self.signal_source == 'ppg':
            nan_vector = nan_in_rows_ppg
        
        nan_count = nan_vector.sum()
        if nan_count != 0:
            print('Found ',nan_count,'rows with NaNs to be removed')       
            # Remove rows with nans
            if self.signal_source == 'ecg' or self.signal_source == 'both':
                self.ecg_labels = self.ecg_labels[~nan_vector.squeeze()]
                self.processed_ECG_dataset = self.processed_ECG_dataset[~nan_vector]
                
            if self.signal_source == 'ppg' or self.signal_source == 'both':
                self.ppg_labels = self.ppg_labels[~nan_vector]
                self.processed_PPG_dataset = self.processed_PPG_dataset[~nan_vector]        
            

    def _trim_end_epochs_to_equal_lengths(self):
        if self.trim_ends_bool==True and self.signal_source=='both':
            ppg_length = len(self.ppg_labels)
            ecg_length = len(self.ecg_labels)
            if ecg_length != ppg_length:
                print('PPG and ECG of unequal epochs. Trimming the end of file: ',str(self.subject_name))
                cutt_off = min(ppg_length,ecg_length)
                self.ppg_labels = self.ppg_labels[0:cutt_off]
                self.ecg_labels = self.ecg_labels[0:cutt_off]
                self.processed_ECG_dataset = self.processed_ECG_dataset[0:cutt_off]
                self.processed_PPG_dataset = self.processed_PPG_dataset[0:cutt_off]


    def _loadPickleData(self,path):
        # Read file
        data = pd.read_pickle(path)


        #self.raw_df = data

        x = data.iloc[:,:-3].values
        
        y = data.iloc[:,-3].values
        
        y = self._mapStringLabelsToNumericLabels(y)
        
        if self.light_sleep_vs_all_bool==False:
            y = self._collapseSleepStage(y)
        if self.light_sleep_vs_all_bool==True:
            y = LightSleepVsAllSleepStageCollapse(y)        
        
        epoch_number =data.iloc[:,-2].values

        # Create one-hot encoding for label
        #integer_encoded = y.reshape(len(y), 1)         
        #return x, integer_encoded
        return x,y
    
    def _mapStringLabelsToNumericLabels(self,y):
        data_dict = getLabelMappings(5)# {'ticks': {'tick': [0, 1, 2, 3, 4], 'label': ['Wake', 'N1', 'N2', 'N3', 'REM']}, 'mapping': {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}}
        
        data_dict = {'ticks': {'tick': [0, 1, 2, 3, 4], 'label': ['Wake', 'N1', 'N2', 'N3', 'REM']}, 'mapping': {'Wake': 0.0, 'N1': 1.0, 'N2': 2.0, 'N3': 3.0, 'REM': 5.0}}
        
        mapped_array = np.vectorize(lambda x: data_dict['mapping'].get(x, -99))(y)
        return mapped_array

    
    def _createMask(self,x, y,Fs = 256):

        #exit()
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
        #Fs = 256    
        #stages = None
        x = x.reshape(1,-1)*1e3
        print("last shape",x.shape)
        #exit()
        ecg_segs, labels , _, _ = segment_ecg_signal(x, labels, window_time, step_time, Fs, newFs=self.resample_frequency, start_end_remove_window_num=0, n_jobs=-1, amplitude_thres=6000)
            
        # Create one-hot encoding for label
        integer_encoded = labels.reshape(len(labels), 1)         
        
        x = torch.from_numpy(ecg_segs).float()
        y = torch.from_numpy(integer_encoded).float()
        y = y.type(torch.LongTensor)        
        return x, y    
    
        
    
    def _signal_source_preparation_routine(self):
        if self.signal_source== 'both':
            ppg_subject =self.ppg_file_path.split(os.sep)[-1].split("ppg")[0]
            ecg_subject =self.ecg_file_path.split(os.sep)[-1].split("ecg")[0]
            if ppg_subject!=ecg_subject:
                raise ValueError(f"PPG and ECG paths do not lead to same subject: \nPPG: {self.ppg_file_path} \nECG: {self.ecg_file_path}")
        
        if self.signal_source == "ppg" or self.signal_source == "both" :
            ppg_signal, ppg_label = self._loadPickleData(self.ppg_file_path)
            self.ppg_samplig_frequency = 64
            self.sampling_frequency = self.ppg_samplig_frequency
            if self.resample_bool == False:
                self.resample_frequency = self.sampling_frequency
                
            if self.mask == False:
                self.processed_PPG_dataset,self.ppg_labels = self._prepareECGData(ppg_signal,ppg_label)
                # if self.filter_bool == True:
                #     _temp_array = []
                #     # This is a hacky solution to ensure that the ECG processing is done on the signal after the window size has been set. It is cast to numpy and then back to torch. It is hacky but it works.
                #     for row in self.processed_PPG_dataset.numpy().squeeze():
                #         bvp_analysis_res = bvp.bvp(signal=row,sampling_rate=self.ppg_samplig_frequency,show=False)
                #         _temp_array.append(bvp_analysis_res['filtered'])
                #     ppg_signal = np.stack(_temp_array)        
                #     row, col = ppg_signal.shape
                #     x = torch.from_numpy(ppg_signal).float()
                #     x = torch.reshape(x,(row,1,col))        
                #     self.processed_PPG_dataset = x


            else:
                ####### THIS DOES NOT WORK CORRECTLY! PPG PEAK DETECTION IS NOT IMPLEMENTED YET
                self.processed_PPG_dataset,self.ppg_labels = self._createMask(ppg_signal,ppg_label,self.ppg_samplig_frequency)
            
        if self.signal_source == "ecg" or self.signal_source == "both" :
            ecg_signal, ecg_label = self._loadPickleData(self.ecg_file_path)
            if self.invert_ecg_bool == True:
                ecg_signal = ecg_signal*-1
            self.hidden_original_signal = ecg_signal
            self.ecg_samplig_frequency = 200
            self.sampling_frequency = self.ecg_samplig_frequency
            if self.resample_bool == False:
                self.resample_frequency = self.sampling_frequency            

            if self.mask == False:
                if self.filter_bool == True:
                    # This ensures that the resampling only takes place after the ecg filtering, as it cannot handle filtering at 64 Hz.
                    temp_resample_frequncy = self.resample_frequency
                    self.resample_frequency = self.ecg_samplig_frequency

                self.processed_ECG_dataset,self.ecg_labels = self._prepareECGData(ecg_signal,ecg_label)
                if self.filter_bool == True:
                    # This ensures that the resampling only takes place after the ecg filtering, as it cannot handle filtering at 64 Hz.
                    self.resample_frequency = temp_resample_frequncy
                # if self.filter_bool == True:
                #     _temp_array = []
                #     i = 0
                #     # This is a hacky solution to ensure that the ECG processing is done on the signal after the window size has been set. It is cast to numpy and then back to torch. It is hacky but it works.
                #     for row in self.processed_ECG_dataset.numpy().squeeze():
                #         print(i)
                #         ecg_analysis_res = ecg.ecg(signal=row,sampling_rate=self.ecg_samplig_frequency,show=False)
                #         _temp_array.append(ecg_analysis_res['filtered'])
                #         i = i+1
                #     ecg_signal = np.stack(_temp_array)        
                #     row, col = ecg_signal.shape
                #     x = torch.from_numpy(ecg_signal).float()
                #     x = torch.reshape(x,(row,1,col))
                #     self.processed_ECG_dataset = x
            else:
                self.processed_ECG_dataset,self.ecg_labels = self._createMask(ecg_signal,ecg_label,self.ecg_samplig_frequency)
                
        if self.signal_source not in ('ecg', 'ppg', 'both'):
            raise ValueError(f"Invalid choice of signal_source: {self.signal_source}. Expected 'ecg', 'ppg' or 'both'")
            
    def _filter_signals(self):

        if self.signal_source == "ppg" or self.signal_source == "both" :
            if self.filter_bool == True:
                if self.resample_bool == True:
                    frequency = self.resample_frequency 
                else:
                    frequency = self.ppg_samplig_frequency
                _temp_ppg_array = []
                # This is a hacky solution to ensure that the PPG processing is done on the signal after the window size has been set. It is cast to numpy and then back to torch. It is hacky but it works.
                for ppg_row in self.processed_PPG_dataset.numpy().squeeze():
                    bvp_analysis_res = bvp.bvp(signal=ppg_row,sampling_rate=frequency,show=False)
                    _temp_ppg_array.append(bvp_analysis_res['filtered'])
                ppg_signal = np.stack(_temp_ppg_array)        
                row, col = ppg_signal.shape
                x = torch.from_numpy(ppg_signal).float()
                x = torch.reshape(x,(row,1,col))        
                self.processed_PPG_dataset = x

            
        if self.signal_source == "ecg" or self.signal_source == "both" :
            if self.filter_bool == True:

                try:
                    i = 0
                    # This is a hacky solution to ensure that the ECG processing is done on the signal after the window size has been set. It is cast to numpy and then back to torch. It is hacky but it works.
                    _temp_ecg_array = []

                    for ecg_row in self.processed_ECG_dataset.numpy().squeeze():
                        #print(i)
                        ecg_analysis_res = ecg.ecg(signal=ecg_row,sampling_rate=self.ecg_samplig_frequency,show=False)
                        _temp_ecg_signal =ecg_analysis_res['filtered']
                        # Hacky solution to resample signal after ECG filtering as it cannot filter with 64 hz frequency
                        if self.resample_bool == True:
                            if self.window_size != None:
                                number_of_new_samps = self.resample_frequency*self.window_size
                            else:
                                number_of_new_samps = self.resample_frequency*30
                            _temp_ecg_signal = signal.resample(_temp_ecg_signal, number_of_new_samps)

                        _temp_ecg_array.append(_temp_ecg_signal)
                        i = i+1
                    ecg_signal = np.stack(_temp_ecg_array)        
                    row, col = ecg_signal.shape
                    ecg_x = torch.from_numpy(ecg_signal).float()
                    ecg_x = torch.reshape(ecg_x,(row,1,col))
                    self.processed_ECG_dataset = ecg_x
                except:
                    print("Error when filtering ECG signal. Subject: "+str(self.subject_name)+" row: "+str(i)) 


if __name__ == '__main__':
    
    #exit
    # Generators
    aauwss_root_folder = r".\aligned_sleep_data_set"
    ecg_folder = os.path.join(aauwss_root_folder,'ecg',"*")
    ppg_folder = os.path.join(aauwss_root_folder,'ppg',"*")


    ecg_paths = glob.glob(ecg_folder)
    ppg_paths = glob.glob(ppg_folder)

    #test_peaks = AAUWSSDL(file_path=ecg_paths[-3],augment_data=False,number_of_sleep_stage=2,normalize_type='zscore',normalize=True,sampling_frequency=None,drop_initial_wake=False,signal_source='ecg',mask = False,light_sleep_vs_all_bool=True)
    test_peaks_raw = AAUWSSDL(file_path=(ecg_paths[-1],ppg_paths[-1]),augment_data=False,number_of_sleep_stage=5,normalize_type='zscore',normalize=False,sampling_frequency=None,resample_bool = True,resample_frequency=64,drop_initial_wake=False,signal_source='both',mask = False,light_sleep_vs_all_bool=False,window_size=270)
    #tp = test_peaks.processed_ECG_dataset


    tpr = np.squeeze(test_peaks_raw.processed_ECG_dataset.numpy())
    tpppg = test_peaks_raw.processed_PPG_dataset
    rows_with_nan = np.where(np.isnan(tpr).any(axis=1))[0]

    print(rows_with_nan)    
    
    abc = test_peaks_raw.processed_ECG_dataset
    abc2 = test_peaks_raw.processed_PPG_dataset
    abc3 =test_peaks_raw.labels

    

    
    
    
