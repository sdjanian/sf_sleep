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
from dataloader_utilities import getLabelMappings, LightSleepVsAllSleepStageCollapse
from scipy.stats import zscore


class ECGDataSetSingle2(torch.utils.data.Dataset):
    
    def __init__(self, file_path:str, shuffle_recording = False, number_of_sleep_stage:int = 5,augment_data:bool = False,normalize:bool=True,resample_bool:bool = True,resample_frequency:int = 64,normalize_type = 'zscore',sampling_frequency=None,window_size = None,drop_initial_wake=False,light_sleep_vs_all_bool:bool = False):
        """
        Loads a single pickled pandas dataframe that contains ECG signal split up into 30 second segments with a corresponding label. When loading each row is a 30 second segment and each column is a single sample.
        This dataframe is then preprocessed to be used for a pytorch neural network. 

        Parameters
        ----------
        file_path : str
            Path for the .pkl files for an ECG with sleep labels.
        shuffle_recording : TYPE, optional
            Not implemented. The default is False.
        number_of_sleep_stage : int, optional
            How many sleep stages to be used for classification. If other than 5 chosen, some will be collapsed. The default is 5.
        augment_data : bool, optional
            Augmentation creates more sleep epochs of the sleep stages with fewer samples. It tries to create an equal distribution if possible. The default is False.
        normalize : bool, optional
            Normalizing the ECG signal. Normalization occurs on epoch level. The default is True.

        Returns
        -------
        None.

        """
        
        self.file_path = file_path
        self.subject_name = file_path.split(os.sep)[-1]
        self.shuffle_recording = shuffle_recording
        self.number_of_sleep_stage = number_of_sleep_stage
        self.augment_data = augment_data
        self.normalize_bool = normalize
        self.resample_bool = resample_bool
        self.resample_frequency = resample_frequency
        self.sampling_frequency = sampling_frequency
        self.normalize_type = normalize_type
        self.light_sleep_vs_all_bool = light_sleep_vs_all_bool
        self.raw_df = None
        self.drop_initial_wake_bool = drop_initial_wake
        self.window_size = window_size
        self.label_dict = self._getLabelDict()
        signal,inter_encoded_labels = self._loadPickleData(self.file_path)
        self.processed_ECG_dataset,self.labels = self._prepareECGData(signal,inter_encoded_labels)        
        #self.processed_ECG_dataset,self.labels = self._prepareECGData()
    def __len__(self):
        #print("number_of_index:", self.processed_ECG_dataset.shape[0])
        return self.processed_ECG_dataset.shape[0]

    def __getitem__(self,idx):
        x = self.processed_ECG_dataset[idx]
        y = self.labels[idx]
        
     
        
        return x, y, self.subject_name
        
    def _prepareECGData(self,signal,inter_encoded_labels):
        """
        Load the ECG data and transform into tensors in the correct shape for a neural network.

        Returns
        -------
        x : torch.LongTensor
            The ECG signal where each row is an epoch.
        y : torch.LongTensor
            The sleep stage labels.

        """
        #return [self._createXandY(path) for path in self.filenames]
        #x, y = self._loadPickleData(self.file_path)
        #x = np.array(x)
        #y = np.array(y)
        x = np.array(signal)
        y = np.array(inter_encoded_labels)        

        if self.window_size != None:
            y_temp = y.tolist()
            #y1 = y2*[30]
            new_list = [[_x]*len(x[0]) for _x in y_temp]
            flat_list = [item for sublist in new_list for item in sublist]
            labels = np.array(flat_list)
            #x = df.values
            x = x.flatten()
            #print("flat shape",x.shape)
            #window_time = 4.5*60 # [s]
            #step_time = 30 # [s]
            #Fs = 256
            #stages = None
            x = x.reshape(1,-1)#*1e3 # This is a weird line of code from the imported github repo. I do not know why they multiply with 1000
            x,y = self._create_windows(x,labels, Fs=self.sampling_frequency,newFs=self.resample_frequency,window_time = self.window_size)
            x = x.squeeze()
            self.sampling_frequency = self.resample_frequency

        if (self.resample_bool == True) and (self.window_size==None):
            x = self._resample(x)
            self.sampling_frequency = self.resample_frequency

        if self.normalize_bool == True:
            if self.normalize_type == 'zscore':
                x = self._normalize(x)
            elif self.normalize_type == 'paper':
                x = self._normalize_all_rows_from_paper(x)
            else:
                raise Exception(self.normalize_type, 'Not a valid normalization chosen')

        if self.augment_data == True:
            data = pd.DataFrame(x)
            data['labels'] = y
            data = self._augmentDataset(data)
            x = data.iloc[:,:-1].values
            y = data.iloc[:,-1].values


        row, col = x.shape
        x = torch.from_numpy(x).float()
        x = torch.reshape(x,(row,1,col))
        y = torch.from_numpy(y).float()
        y = y.type(torch.LongTensor)        
        
        return x, y

    def _loadPickleData(self,path):
        # Read file
        data = pd.read_pickle(path)
        data = self._converSleepStage4To3(data) # Remenant of using R&K sleep stage in MESA and SHHS instead of AASM

        if self.drop_initial_wake_bool == True:
            data = self.__drop_initial_wake(data)

        #self.raw_df = data

        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        if self.light_sleep_vs_all_bool==False:
            y = self._collapseSleepStage(y)
        if self.light_sleep_vs_all_bool==True:
            y = LightSleepVsAllSleepStageCollapse(y)
        # Create one-hot encoding for label
        integer_encoded = y.reshape(len(y), 1)         
        return x, integer_encoded

    def _createXandY(self,path):
        """
        Loads the ECG pickle file. Then it converts from 5 sleep stages to the fewer stages, if chosen. Performs augmentation and shuffles if the dataset if chosen.

        Parameters
        ----------
        path : str
            Path to pickle file.

        Returns
        -------
        x : numpy.array
            The ECG signal. Rows are epochs.
        integer_encoded : numpy.array
            The sleep stage labels.

        """
        # Read file
        data = pd.read_pickle(path)
        data = self._converSleepStage4To3(data)
        data['labels'] = self._collapseSleepStage(data['labels'])

        
        if self.augment_data == True:
            data = self._augmentDataset(data)

        if self.shuffle_recording == True:
            data = shuffle(data)
        #data = pd.read_csv(path)
        
        #Split file into signal and label
        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        # Create one-hot encoding for label
        integer_encoded = y.reshape(len(y), 1)         
        return x, integer_encoded

    
    def _collapseSleepStage(self,y):
        """
        Groups sleep stages according to the AASM guidelines depending on 2, 3, 4 and 5 sleep stages. If the input file was scorede according to R&K it is converted AASM.

        Parameters
        ----------
        y : numpy.array
            Array of sleep stages to collapse.

        Returns
        -------
        y : numpy.array
            Collapsed sleep stages.

        """
        
        if self.number_of_sleep_stage == 2:
            y[y!=0] = 1 # Group all sleep into one class
        if self.number_of_sleep_stage == 3:
            # Group into Wake, NREM, REM
            y[y==2] = 1
            y[y==3] = 1
            y[y==5] = 2
        if self.number_of_sleep_stage == 4:
            # Group into Wake, Light, Deep, REM
            y[y==2] = 1
            y[y==3] = 2
            y[y==5] = 3            
 
        if self.number_of_sleep_stage == 5:
            # Group into Wake, NREM, REM
            y[y==5] = 4
        return y    
    
    def _converSleepStage4To3(self, df):
        df["labels"][df["labels"]==4] = 3
        return df
    def _getLabelDict(self):
        #tick_name = "ticks"
        #dict_name = ""
        label_dict = getLabelMappings(number_of_sleep_stage = self.number_of_sleep_stage,light_sleep_vs_all_bool=self.light_sleep_vs_all_bool) 
        return label_dict        

    def __drop_initial_wake(self,df,detect_from = 0):
        #detect_from = 0
        
        # Detect the index of the first change from detect_from to any other value
        index_of_change = df[(df['labels'] != detect_from) & (df['labels'].shift(1) == detect_from)].index[0]
        sub_df = df.iloc[index_of_change:]
        sub_df = sub_df.reset_index(drop=True)
        return sub_df

    def _normalize(self,x):        
        """
        Normalizing a 30 second ECG segment by (x-mean)/(max-min)

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        norm : TYPE
            DESCRIPTION.

        """
        #norm = (x-np.mean(x,axis=1)[:,np.newaxis])/(np.max(x,axis=1)[:,np.newaxis]-np.min(x,axis=1)[:,np.newaxis])
        #norm =zscore(x,0)
        norm =zscore(x,0,nan_policy='omit')
        return norm
    '''
    def _normalize_from_paper(self,x):
        """
        Normalizing a 30 second segment by splitting it into 1 second segment and normlizing each 1 second segment to zero mean and unit variance.
        Based on the approach in "Performance of a Convolutional Neural Network Derived from PPG Signal in Classifying Sleep Stages" by Habib et al. 2022

        Parameters
        ----------
        x : TYPE
            30 second ECG segment.

        Returns
        -------
        x : TYPE
            Normalized 30 second ECG segment.

        """
        """
        epoch_length_in_samples = x.shape[1]
        one_second_interval_length = int(epoch_length_in_samples/30)
        
        for i in np.arange(0,30):
            start_of_interval = i*one_second_interval_length
            end_of_interval = (i+1)*one_second_interval_length
            x[:,start_of_interval:end_of_interval] =  (x[:,start_of_interval:end_of_interval]-np.mean(x[:,start_of_interval:end_of_interval]))/np.std(x[:,start_of_interval:end_of_interval])
        
        #norm = (x-np.mean(x,axis=1)[:,np.newaxis])/(np.max(x,axis=1)[:,np.newaxis]-np.min(x,axis=1)[:,np.newaxis])
        return x      
        """
        #print('signal before: ',x)
        #print('signal before: ',x.shape)

            #print (row.shape)
        row_list = []
        for row in x:
            #print(row.shape)
            #print(type(row[0*64:(0+1)*64]))
            seg_z_ = []
            for i_sec_ in range(0, 30):
                sec_seg_ = row[i_sec_*64:(i_sec_+1)*64]
                seg_z_.extend(zscore(sec_seg_))
            #print('\nsignal after: ', seg_z_)
            row_list.append(seg_z_)
        seg_z_X = np.vstack(np.array(row_list))


        return seg_z_X
    '''
    def _normalize_from_paper(self,x):

        #print(row.shape)
        #print(type(row[0*64:(0+1)*64]))
        seg_z_ = []
        if self.window_size == None:
            epoch_size = 30
        else:
            epoch_size = self.window_size
            #print("epoch size:",epoch_size)

        for i_sec_ in range(0, epoch_size):
            sec_seg_ = x[i_sec_*self.sampling_frequency:(i_sec_+1)*self.sampling_frequency]
            seg_z_.extend(zscore(sec_seg_))
            #print(i_sec_)
        #print('\nsignal after: ', seg_z_)
        return seg_z_
    def _normalize_all_rows_from_paper(self,x):
        norm_x = []
        for row in x:
            norm_x.append(self._normalize_from_paper(row))
        norm_x = np.vstack(norm_x)
        return norm_x
    def _resample(self,x):
        """
        Resamples the signal to 64 Hz
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        new_np_array : TYPE
            DESCRIPTION.

        """
        new_arr = []
        new_Hz = self.resample_frequency
        secs = 30
        number_of_new_samps = secs*new_Hz
    
        for row in x:
            new_arr.append(signal.resample(row, number_of_new_samps))
        
        new_np_array = np.vstack(new_arr)
        return new_np_array

    #def _createAugmentedSignal(self,s0,s1,label,sampling_frequency:int=256,step_in_seconds:int=2,start_offset_in_seconds:int = 5,stop_offset_in_seconds:int = 4 ) -> np.ndarray:
    def _createAugmentedSignal(self,cur_seg,prev_seg,cur_label, i_prev_seg_start_sec=20, step_sec:int=1, i_prev_seg_stop_sec:int=1,sampling_frequency:int=256, seg_sec = 30):
        """
        Rewrite of the function seg_augment from https://github.com/deakin-deep-dreamer/sleep_stage_ppg/blob/main/datasource.py
    
        Parameters
        ----------
        cur_seg : TYPE
            DESCRIPTION.
        prev_seg : TYPE
            DESCRIPTION.
        cur_label : TYPE
            DESCRIPTION.
        i_prev_seg_start_sec : TYPE, optional
            DESCRIPTION. The default is 20.
        step_sec : TYPE, optional
            DESCRIPTION. The default is 1.
        i_prev_seg_stop_sec : TYPE, optional
            DESCRIPTION. The default is 1.
        hz : TYPE, optional
            DESCRIPTION. The default is 64.
        seg_sec : TYPE, optional
            DESCRIPTION. The default is 30.
    
        Returns
        -------
        None.
    
        """
        #prev_seg = self.segments[-2]
        #cur_seg = self.segments[-1]
        #cur_label = self.seg_labels[-1]
        #seg_sec = 30
        #hz = 64
        i_cur_seg_stop_sec = seg_sec - \
            (seg_sec - i_prev_seg_start_sec)
        count_new_seg = 0
    
        augmented_segments = []
        augmented_segments_labels = []
        for i_prev_start_sec in range(i_prev_seg_start_sec, i_prev_seg_stop_sec, step_sec):
            #print(i_prev_start_sec)
            count_new_seg += 1
            r"Form new seg = 0.5*prev_seg + 0.5*cur_seg"
            new_seg = []
            new_seg = np.concatenate(
                (
                    prev_seg[i_prev_start_sec*sampling_frequency:],
                    cur_seg[:i_cur_seg_stop_sec*sampling_frequency]
                ), axis=0
            )
            augmented_segments.append(new_seg)
            augmented_segments_labels.append(cur_label)
            #assert new_seg.shape[0] == self.seg_sz
    
            r"Right shift prev and cur segment index by sampling hz."
            i_cur_seg_stop_sec += step_sec
    
            r"Add to segment and label global list."
            #self.segments.insert(-2, new_seg)
            #self.seg_labels.insert(-2, cur_label)
    
            #r"store segment index to record-wise store."
            #self.record_wise_segments[self.record_names[-1]].append(
            #    len(self.segments)-1
            #)
        return augmented_segments,augmented_segments_labels

    def _augment_ignore_classes(self):
        print("sleep stages: ",self.number_of_sleep_stage )
        if self.number_of_sleep_stage == 5:
            ignore_list = [ self.label_dict['mapping']['Wake'], self.label_dict['mapping']['N2']]
        elif self.number_of_sleep_stage == 4:
            ignore_list = [ self.label_dict['mapping']['Wake'], self.label_dict['mapping']['Light']]
        elif self.number_of_sleep_stage == 3:
            ignore_list = [ self.label_dict['mapping']['Wake'], self.label_dict['mapping']['NREM']]
        else:
            print("No classes to ignore")
            ignore_list = []
        return ignore_list
    def _augmentDataset(self,df):
        """
        Runs self._createAugmentedSignal through the dataframe and appends the newly created epochs to the bottom of the dataframe

        Parameters
        ----------
        df : TYPE
            DataFrame containin epochs in each row with columns for each sample. Last column ("labels) are the sleep stage labels.

        Returns
        -------
        df : TYPE
            DataFrame in same format as input with the augmented epochs appended at the bottom.

        """
        augmented_signals = []
        hz = int(len(df.iloc[0][0:-1])/30)

        print(np.unique(df['labels'],return_counts=True))
        classes_to_ignore_augmentation = self._augment_ignore_classes()
        print(classes_to_ignore_augmentation)

        for index, row in df.iterrows():
            #print(row)
            if (index>=1) and (index<len(df)-1):
            #if df["labels"].iloc[index]==df["labels"].iloc[index+1]:
                #print(row["labels"])
                cur_seg = df.iloc[index][0:-1].values
                prev_seg = df.iloc[index-1][0:-1].values
                #cur_seg,prev_seg
                current_label = df['labels'].iloc[index]
                previous_label = df['labels'].iloc[index-1]
                #print(current_label)
                if current_label == previous_label:
                    #print(self.label_dict)
                    if current_label in classes_to_ignore_augmentation:
                        None
                    else:
                        aug_segs, aug_labels = self._createAugmentedSignal(cur_seg=cur_seg,
                                                                              prev_seg = prev_seg,
                                                                              cur_label=current_label,
                                                                              i_prev_seg_start_sec=5, step_sec=2,i_prev_seg_stop_sec=30,
                                                                              sampling_frequency = hz, seg_sec = 30)
                        #print('number of augs', len(aug_labels))
                        aug_segs = np.vstack(aug_segs)
                        #print("Augt size",aug_segs.shape)
                        #print("Label size",np.array(aug_labels).squeeze().shape)
                        #together = np.append(aug_segs,np.array(aug_labels),axis=1)
                        together = np.c_[aug_segs,np.array(aug_labels)]
                        #print("together",together.shape)
                        augmented_signals.append(together)

                else:
                    r"Prev segment label different, start taking from 75%."
                    aug_segs, aug_labels = self._createAugmentedSignal(cur_seg=cur_seg,
                                                                          prev_seg = prev_seg,
                                                                          cur_label=current_label,
                                                                          i_prev_seg_start_sec=23, step_sec=2,i_prev_seg_stop_sec=30,
                                                                          sampling_frequency = hz, seg_sec = 30)
                    aug_segs = np.vstack(aug_segs)
                    #print("Augt size",aug_segs.shape())
                    #print("Label size",aug_labels.shape())
                    together = np.c_[aug_segs,np.array(aug_labels)]
                    augmented_signals.append(together)

        """
                    augmented_signals.append(self._createAugmentedSignal(cur_seg=cur_seg,
                                                                          prev_seg = prev_seg,
                                                                          cur_label=label,
                                                                          i_prev_seg_start_sec=5, step_sec=2,i_prev_seg_stop_sec=30,
                                                                          sampling_frequency = hz, seg_sec = 30))
        """

        try:
            print("Finished making augments")
            print("Total number of augments", len(augmented_signals),type(augmented_signals),augmented_signals[1].shape)
            augmented_signals_stacked = np.vstack(augmented_signals)
            print("Finished making Stacking")

            temp_df = pd.DataFrame(augmented_signals_stacked, columns=list(df))
            augmented_df = pd.concat([df,temp_df],axis=0) 
            print("Augmented dataset")      
        except Exception as e:
            print("\nWas not able to augment recording: ",self.file_path)
            print(e)
            return df
        #augmented_signals_stacked = np.vstack(augmented_signals)
        #print("Finished making Stacking")

        #temp_df = pd.DataFrame(augmented_signals_stacked, columns=list(df))
        #augmented_df = pd.concat([df,temp_df],axis=0)
        return augmented_df

    def _create_windows(self,signal, labels, window_time:int = 270, step_time:int = 30, Fs = None, newFs=None, start_end_remove_window_num=1, n_jobs=-1):

        from segment_signal import fast_resample
        #newFs = 200
        #Fs = 256
        #window_time = 4.5*60 # [s]
        #step_time = 30 # [s]
        #print('window_time -should be 270',window_time)
        #print('Fs - should be 256',Fs)
        #print('newFs - should be 200',newFs)
        #print('step_time -should be 30',step_time)
        start_end_remove_window_num = 0
        if newFs is not None and Fs!=newFs:
            signal = fast_resample(signal, newFs*1./Fs)
            if labels is not None:
                labels = np.repeat(labels[::int(30*Fs)], int(30*newFs))
                minlen = min(signal.shape[-1], len(labels))
                signal = signal[...,:minlen]
                labels = labels[:minlen]
            Fs = newFs
        
        window_size = int(round(window_time*Fs))
        #print("window_size",window_size)
        step_size = int(round(step_time*Fs))
        #print('step_size',step_size)

        start_ids = np.arange(0, signal.shape[1]-window_size+1, step_size)
        if labels is not None:
            labels = labels[start_ids+window_size//2]
            if start_end_remove_window_num>0:
                start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
                labels = labels[start_end_remove_window_num:-start_end_remove_window_num]
            assert len(start_ids)==len(labels)

        assert signal.shape[0]==1
        signal_segs = signal[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
        #print('final shape',signal_segs[0].shape)
        if labels is not None:
            return signal_segs, labels
        else:
            return signal_segs

if __name__ == '__main__':
    exit
    #exit
    # Generators
    dataset_name = "mesa"
    training_path_mesa = os.path.join("non-augmented","processed_data_train",dataset_name,"*")
    testing_path = os.path.join("non-augmented","processed_data_test",dataset_name,"*")
    training_path_shhs1 = os.path.join("non-augmented_shhs1","processed_data_test","shhs1","*")

    training_fileNames_mesa = glob.glob(training_path_mesa)
    training_fileNames_shhs1 = glob.glob(training_path_shhs1)
    training_fileNames = training_fileNames_mesa+training_fileNames_shhs1

    d200 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize_type='zscore',normalize=True,sampling_frequency=256,resample_frequency=64,drop_initial_wake=True)
    x = d200.processed_ECG_dataset
    print(x[0].shape)
    plt.figure()
    plt.plot(x[0].squeeze())
    #d200 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize_type='paper',normalize=True,sampling_frequency=256,resample_frequency=64)



    exit()
    #d2 = ECGDataSetSingle(training_fileNames_mesa[0],augment_data=True,number_of_sleep_stage=4)
    d3 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize_type='paper',normalize=True)
    d4 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize = False)
    d5 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize_type='zscore',normalize=True)
    d6 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=True,number_of_sleep_stage=4,normalize_type='zscore',normalize=True)

    #d = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=True,number_of_sleep_stage=4)

    print(np.unique(d3.labels.cpu().detach().numpy(),return_counts=True))
    print(np.unique(d6.labels.cpu().detach().numpy(),return_counts=True))
    #l = d3.labels.cpu().detach().numpy()
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(d3.__getitem__(100)[0].cpu().detach().numpy().squeeze())
    plt.title('Paper')
    plt.figure()
    plt.plot(d4.__getitem__(100)[0].cpu().detach().numpy().squeeze())
    plt.title('Non normalized')
    plt.figure()
    plt.plot(d5.__getitem__(100)[0].cpu().detach().numpy().squeeze())
    plt.title('zscore')

    plt.figure()
    plt.plot(d6.__getitem__(2000)[0].cpu().detach().numpy().squeeze())
    plt.title('Aug paper')

    s =d4.__getitem__(100)[0].cpu().detach().numpy().squeeze()
    all_s = d4.processed_ECG_dataset.cpu().detach().numpy().squeeze()
    def _normalize_from_paper(x):

        #print(row.shape)
        #print(type(row[0*64:(0+1)*64]))
        seg_z_ = []
        for i_sec_ in range(0, 30):
            sec_seg_ = x[i_sec_*64:(i_sec_+1)*64]
            seg_z_.extend(zscore(sec_seg_))
        #print('\nsignal after: ', seg_z_)
        return seg_z_
    norm_s =_normalize_from_paper(s)
    p = []
    for w in all_s:
        p.append(_normalize_from_paper(w))
    d = np.vstack(p)
    plt.figure()
    plt.plot(d[100,:])


    d200 = ECGDataSetSingle2(training_fileNames_mesa[0],augment_data=False,number_of_sleep_stage=4,normalize_type='zscore',normalize=True,resample_bool=False)

    x = d200.processed_ECG_dataset
    from dataloader_ecg_respiration_sleep_staging import ECGPeakMask
    #b = one_recording = ECGPeakMask(training_fileNames_mesa[0],number_of_sleep_stage=5,hz=256)
    raw_df =d200.raw_df
    index_of_change = raw_df[(raw_df['labels'] != 0) & (raw_df['labels'].shift(1) == 0)].index[0]
    sub_df = raw_df['labels'].iloc[0:180]

    def drop_initial_wake(df,detect_from = 0):
        #detect_from = 0
        
        # Detect the index of the first change from detect_from to any other value
        index_of_change = df[(df['labels'] != detect_from) & (df['labels'].shift(1) == detect_from)].index[0]
        sub_df = df.iloc[index_of_change:]
        return sub_df

    t = drop_initial_wake(raw_df)




    """
    training_set2 = ECGDataSet2(training_fileNames,shuffle_recording=True,number_of_sleep_stage=5, augment_data=True,internal_batch_size=16)
    training_set = ECGDataSet(training_fileNames,shuffle_recording=True,number_of_sleep_stage=5, augment_data=True)
    sampler = GarbageSampler(training_set,batch_size=16)
    training_generator = torch.utils.data.DataLoader(training_set,batch_size=1,sampler=sampler) 
    
    validation_fileNames = glob.glob(testing_path)
    validation_set = ECGDataSet(validation_fileNames,shuffle_recording=True,number_of_sleep_stage=3)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=1)
     # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   
    
    training_set_alpha = ECGDataSet(training_fileNames[0:2],shuffle_recording=True,number_of_sleep_stage=5, augment_data=True)
    training_set_beta = ECGDataSet(validation_fileNames,shuffle_recording=True,number_of_sleep_stage=5, augment_data=True)
    d_T = [training_set_alpha,training_set_beta]

    dataset = torch.utils.data.ConcatDataset(d_T)
    training_generator_conc = torch.utils.data.DataLoader(dataset,batch_size=1) 
    """


    """
    len_dict = {}
    
    for file in training_fileNames:
        df = pd.read_pickle(file)
        file_name = file.split(os.sep)[-1].split(".")[0]
        len_dict[file_name] = df.shape[0]
        
    internal_batch_size = 16
    index_list = []
    for key, value in len_dict.items():
        print(key,value)
        for i in range(0,int(np.floor(value/internal_batch_size))):
            index_list.append((key,i))
            
    """
    """
    print("Number of recs in dataset 1: ",len(training_generator_conc.dataset.datasets[0].filenames),"\n",training_generator_conc.dataset.datasets[0].filenames,)
    print("Number of recs in dataset 2: ",len(training_generator_conc.dataset.datasets[1].filenames),"\n",training_generator_conc.dataset.datasets[1].filenames,)

    dataset_list = []
    for file in training_fileNames:
        dataset_list.append(ECGDataSetSingle(file,shuffle_recording = False, number_of_sleep_stage = 5,augment_data=True,normalize=True))
    #training_set_single = ECGDataSetSingle(training_fileNames[0],shuffle_recording = False, number_of_sleep_stage = 5,augment_data=True,normalize=True)
    datasets = torch.utils.data.ConcatDataset(dataset_list)
    training_generator_single = torch.utils.data.DataLoader(datasets,batch_size=16) 
    
    tttt,zzzz = torch.utils.data.random_split(datasets, (0.8,0.2))
    training_generator_single = torch.utils.data.DataLoader(tttt,batch_size=25) 


    signal = []
    pbar = tqdm(range(1))
    ii = 0
    f_names = []
    shapes = []
    for epoch in pbar:
        print('EPOCH {}:'.format(epoch + 1))
        for local_batch, local_labels, train_file_name in training_generator_single:
            print(local_batch)
            print(local_labels)
            print(train_file_name)
            f_names.append(train_file_name)            
            shapes.append(local_batch.shape[0])

            # Transfer to GPU
            """
"""
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch = local_batch.squeeze(0)
            local_labels = local_labels.squeeze()
            s = local_batch.cpu().detach().numpy()
            signal.append(s)

            
   
 
 
        for i, (val_local_batch, val_local_labels, val_file_name) in enumerate(validation_generator):
            val_local_batch, val_local_labels = val_local_batch.to(device), val_local_labels.to(device)
            #tt = local_batch.squeeze()
            #tttt = torch.permute(tt,(40,1,1920))
            #tttt = torch.reshape(local_batch,(40,1,1920))
            
            val_local_batch = val_local_batch.squeeze(0)
            val_local_labels = val_local_labels.squeeze()

    t = local_labels.cpu().detach().numpy()
    unique, counts = np.unique(t, return_counts=True)
"""
    #b = signal[0].squeeze().ravel()
    #plt.figure()
    #plt.plot(b)
