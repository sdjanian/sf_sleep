import mne
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
import copy
import itertools
import os
import sys
import glob
from tqdm import tqdm
import datetime
import csv
import pytz
import matplotlib.pyplot as plt 



class AllignECGandPPG:


    def __init__(self,dataset = "", root_folder:str = "sample_dataset", output_folder:str = "processed_data",pre_root:str = os.getcwd()):

        self.dataset_name = "mesa"
        self.ecg_sampling_frequency = None
        self.ecg_col_name = "EKG"
        self.epoch_length_in_sec = 30
        self.samples_in_ecg_epoch = None#self.ecg_sampling_frequency * self.epoch_length_in_sec
        
    def ECGRelativeTimeToAbsoluteTime(self,ecg_df:pd.DataFrame,ecg_start_time_stamp):
        relative_timedelta = [datetime.timedelta(seconds=ts) for ts in ecg_df['time'].values]
        absolute_timestamps = [ecg_start_time_stamp + delta for delta in relative_timedelta]
        ecg_df['time'] = absolute_timestamps
        ecg_df['time'] = pd.to_datetime( ecg_df['time'], utc=True)    
        return ecg_df
     
    def TrimECGSignalToAnnotations(self,ecg_df:pd.DataFrame,initial_annotation_time_stamp = None,last_annotation_time_stamp=None):
        # Go from relative time
        ecg_df_trimmed = ecg_df      
        if initial_annotation_time_stamp != None:
            ecg_df_trimmed = ecg_df_trimmed[ecg_df_trimmed['time']>=initial_annotation_time_stamp]
            
        if initial_annotation_time_stamp != None:
            ecg_df_trimmed = ecg_df_trimmed[ecg_df_trimmed['time']<=last_annotation_time_stamp]            
        ecg_df_trimmed = ecg_df_trimmed.reset_index()    
        #ecg_df_trimmed = None
        return ecg_df_trimmed
    def __generateListOfEpochNumbersForEachSample(self,length_of_vector,sampling_frequency):
    # Create a vector of epoch numbers. Epoch number starts at 1
        samples_in_epoch = sampling_frequency * self.epoch_length_in_sec
        epoch_list = []
        for i in range(1,int(np.floor(length_of_vector/samples_in_epoch))+1):
            epoch_list.append([i]*samples_in_epoch)     
        flat = list(itertools.chain.from_iterable(epoch_list))    
        return flat
    def AssignSleepStagesToEpochs(self,ecg,annotation_df):
        merged_df = ecg.merge(annotation_df[['Epoch Number', 'Sleep Stage']],left_on='Epochs', right_on='Epoch Number', how='left')
        merged_df.drop(columns=['Epoch Number'], inplace=True)
        return merged_df
    
    def __createEpochDataFrameWithAnnotations(self,df):
    
        def select_columns_func(x):
            signal = x[self.current_signal_col].values
            labels = x["Sleep Stage"].iloc[0]
            epoch = x["Epochs"].iloc[0]

            time = x["time"].values
            return{"signal":signal,"labels":labels,"time":time,'epoch':epoch}
        
        temp = df.groupby(["Epochs"]).apply(select_columns_func)
        epoch_df = pd.DataFrame(list(temp.values))    
        return epoch_df    
    
    def _loadPPGcsv(self,ppg_path):
        empatica_data = pd.read_csv(ppg_path,skiprows=1)        
        empatica_data.columns = ["BVP"]        
        with open(ppg_path, newline='') as f:
          reader = csv.reader(f)
          empatica_start_time_unix = int(float(next(reader)[0]))  # gets the first line
          empatica_sampling_freq_ =  int(float(next(reader)[0]))     
          
        # Define the target timezone (GMT+0200)
        gmt_1_subjects = ['subject_01','subject_02','subject_03']
        if any(subject in ppg_path for subject in gmt_1_subjects):
            # Do something if the condition is met
            print("Time zone set to UTC 1")
            target_timezone = pytz.timezone('Etc/GMT-1')  # Note: GMT+1 is represented as GMT-1 in pytz     

        else:
            # Do something else if no match is found
            print("Time zone set to UTC 2")
            target_timezone = pytz.timezone('Etc/GMT-2')  # Note: GMT+2 is represented as GMT-2 in pytz     
        
        
        empatica_time_stamps = self.generateUnixTimestampsForPPGSignal(empatica_start_time_unix,empatica_sampling_freq_,len_of_signal=len(empatica_data),target_timezone=target_timezone)
        #empatica_time_stamps = empatica_time_stamps + empatica_start_time_unix
        empatica_data["time"] = empatica_time_stamps
        empatica_data['time'] = empatica_data['time'].dt.tz_localize(None)
        empatica_data['time'] = empatica_data['time'].dt.tz_localize('UTC')
        #empatica_data['time'] = pd.to_datetime(empatica_data['time'],unit='s', utc=True)
        return empatica_data, empatica_sampling_freq_

    def generateUnixTimestampsForPPGSignal(self,unix_start_time,sampling_frequency,len_of_signal,target_timezone):
        #interval = 1000/sampling_frequency/1000
        #ls = np.arange(0, interval * len_of_signal , interval)    
        #return ls    
        #sampling_rate = 64  # 64 samples per second
        sampling_interval = 1 / sampling_frequency  # Interval between samples in seconds
        
        #start_time = pd.to_datetime(1680724640, unit='s', utc=True)
        # Convert the initial Unix timestamp to a datetime object in UTC
        start_time_utc = datetime.datetime.fromtimestamp(unix_start_time, tz=pytz.utc)
        

        
        # Convert the start time to the target timezone
        start_time_gmt_plus_2 = start_time_utc.astimezone(target_timezone)
        
        # Generate a list of timestamps
        timestamps = [start_time_gmt_plus_2 + datetime.timedelta(seconds=i * sampling_interval) for i in range(len_of_signal)]

        return timestamps
    
    def _LoadECGedf(self,edf_path):
            if 'subject_12' in edf_path:
                signal_df,sampling_frequency = self._mergeSubject12(edf_path)
            elif 'subject_13' in edf_path:
                signal_df,sampling_frequency = self._mergeSubject13(edf_path)
            else:   
                edf_meta = mne.io.read_raw_edf(edf_path,verbose=0,include=["time",'ECG'],preload=False)
                sampling_frequency = int(edf_meta.info['sfreq'])
                initial_timestamp = edf_meta.info['meas_date']
                ecg_df = edf_meta.to_data_frame()
                signal_df = self.ECGRelativeTimeToAbsoluteTime(ecg_df,initial_timestamp)
    
            return signal_df,sampling_frequency
        
    def _mergeSubject12(self,subject_12_path):
        subject_12_1_path=subject_12_path+'_1.edf'
        subject_12_2_path=subject_12_path+'_2.edf'
        
               # edf_meta = mne.io.read_raw_edf(edf_path,verbose=0,include=["time",'ECG'],preload=False)
    
        subject_12_1 = mne.io.read_raw_edf(subject_12_1_path,verbose=0,include=["time",'ECG'],preload=False)# mne.io.read_raw_edf(subject_12_1_path,preload=True)
        subject_12_2 = mne.io.read_raw_edf(subject_12_2_path,verbose=0,include=["time",'ECG'],preload=False)
        subject_12_1_initial_timestamp = subject_12_1.info['meas_date']
        subject_12_2_initial_timestamp = subject_12_2.info['meas_date']
        sampling_frequency = int(subject_12_1.info['sfreq'])

    
        subject_12_1_df = subject_12_1.to_data_frame()
        subject_12_2_df = subject_12_2.to_data_frame()
        
        subject_12_1_df=self.ECGRelativeTimeToAbsoluteTime(subject_12_1_df, subject_12_1_initial_timestamp)
        subject_12_2_df=self.ECGRelativeTimeToAbsoluteTime(subject_12_2_df, subject_12_2_initial_timestamp)
    
    
        time_series = pd.date_range(start=subject_12_1_df['time'].iloc[-1], end=subject_12_2_df['time'].iloc[0], freq='5L',inclusive='neither')  # 'L' stands for milliseconds
        inbetween_df = pd.DataFrame({'time':time_series,'ECG':np.nan})
        merged_df =pd.concat([subject_12_1_df,inbetween_df,subject_12_2_df])
        return merged_df,sampling_frequency
    
    def _mergeSubject13(self,subject_13_path):
        subject_13_1_path=subject_13_path+'_1.edf'
        subject_13_2_path=subject_13_path+'_2.edf'
        subject_13_3_path=subject_13_path+'_3.edf'

            
        subject_13_1 = mne.io.read_raw_edf(subject_13_1_path,verbose=0,include=["time",'ECG'],preload=False)# mne.io.read_raw_edf(subject_13_1_path,preload=True)
        subject_13_2 = mne.io.read_raw_edf(subject_13_2_path,verbose=0,include=["time",'ECG'],preload=False)
        subject_13_3 = mne.io.read_raw_edf(subject_13_3_path,verbose=0,include=["time",'ECG'],preload=False)

        subject_13_1_initial_timestamp = subject_13_1.info['meas_date']
        subject_13_2_initial_timestamp = subject_13_2.info['meas_date']
        subject_13_3_initial_timestamp = subject_13_3.info['meas_date']
        sampling_frequency = int(subject_13_1.info['sfreq'])

    
        subject_13_1_df = subject_13_1.to_data_frame()
        subject_13_2_df = subject_13_2.to_data_frame()
        subject_13_3_df = subject_13_3.to_data_frame()

        
        subject_13_1_df=self.ECGRelativeTimeToAbsoluteTime(subject_13_1_df, subject_13_1_initial_timestamp)
        subject_13_2_df=self.ECGRelativeTimeToAbsoluteTime(subject_13_2_df, subject_13_2_initial_timestamp)
        subject_13_3_df=self.ECGRelativeTimeToAbsoluteTime(subject_13_3_df, subject_13_3_initial_timestamp)

    
    
        time_series_1 = pd.date_range(start=subject_13_1_df['time'].iloc[-1], end=subject_13_2_df['time'].iloc[0], freq='5L',inclusive='neither')  # 'L' stands for milliseconds
        inbetween_df_1 = pd.DataFrame({'time':time_series_1,'ECG':np.nan})
        
        time_series_2 = pd.date_range(start=subject_13_2_df['time'].iloc[-1], end=subject_13_3_df['time'].iloc[0], freq='5L',inclusive='neither')  # 'L' stands for milliseconds
        inbetween_df_2 = pd.DataFrame({'time':time_series_2,'ECG':np.nan})
        #merged_df =pd.concat([subject_13_1_df,inbetween_df_1,subject_13_2_df,inbetween_df_2,subject_13_2_df])
        merged_df =pd.concat([subject_13_1_df,inbetween_df_1,subject_13_2_df,inbetween_df_2,subject_13_3_df])

        return merged_df, sampling_frequency
            
    def AllignOneSubject(self, edf_path:str='',ppg_path:str='',annotation_path:str='',ecg_or_ppg:str = 'ecg'):
        
        if ecg_or_ppg == 'ecg':
            self.current_signal_col = 'ECG'
            signal_df,sampling_frequency = self._LoadECGedf(edf_path)

            
            '''
            edf_meta = mne.io.read_raw_edf(edf_path,verbose=0,include=["time",'ECG'],preload=False)
            sampling_frequency = int(edf_meta.info['sfreq'])
            initial_timestamp = edf_meta.info['meas_date']
            self.current_signal_col = 'ECG'
            ecg_df = edf_meta.to_data_frame()
            signal_df = self.ECGRelativeTimeToAbsoluteTime(ecg_df,initial_timestamp)
            '''
            
        if ecg_or_ppg =='ppg':
            self.current_signal_col = 'BVP'
            signal_df,sampling_frequency = self._loadPPGcsv(ppg_path)

        
        annotation_df = pd.read_excel(annotation_path)
        annotation_df['Epoch Number'] =  annotation_df['Epoch Number']- annotation_df['Epoch Number'].min() +1
        # Convert timestamps
        annotation_df['Event Start Time'] =  pd.to_datetime( annotation_df['Event Start Time'],utc=True)
        initial_annotation_time_stamp = annotation_df['Event Start Time'].iloc[0]
        last_annotation_time_stamp = annotation_df['Event Start Time'].iloc[-1]+ datetime.timedelta(seconds=30)
        

        ecg_trimmed = self.TrimECGSignalToAnnotations(signal_df,initial_annotation_time_stamp,last_annotation_time_stamp)
    
        epochs = self.__generateListOfEpochNumbersForEachSample(len(ecg_trimmed),sampling_frequency)
        
        ecg_trimmed_with_epochs = pd.concat([ecg_trimmed,pd.Series(epochs,name="Epochs")],axis=1)
    
        
        ecg_with_annotations = self.AssignSleepStagesToEpochs(ecg_trimmed_with_epochs,annotation_df)
        


        epoch_df = self.__createEpochDataFrameWithAnnotations(ecg_with_annotations)   
        stack = np.stack(epoch_df["signal"].values)
        alligned_df = pd.DataFrame(stack)
        alligned_df["labels"] = epoch_df["labels"].values
        alligned_df['epoch'] = epoch_df["epoch"].values
        alligned_df['time'] = epoch_df["time"].values
        
        return alligned_df
    

def edf_path_12_13_helper_function(edf_path_list):
    
    def temp_func(path):
        if 'subject_12' in path:
            temp_path = path.split('subject_12')[0]+'subject_12'
        elif 'subject_13' in path:
            temp_path = path.split('subject_13')[0]+'subject_13'        
        else:
            temp_path = path
        return temp_path 
    
    fixed_list = sorted(list(set([temp_func(p) for p in edf_path_list]))) # Removed the _X part of subject 12 and 13, removes duplicated an returns intro a sorted list
    return fixed_list

def GenerateAlignedECGandPPGdatasets(root_folder,generate_plots = True,save_output = True):
    edf_path_list = glob.glob(os.path.join(root_folder,'edfs','*'))
    ann_path_list =  glob.glob(os.path.join(root_folder,'annotations','*'))
    empatica_path_list =  glob.glob(os.path.join(root_folder,'empatica','*','BVP.csv'))
    
    
    edf_path_list = edf_path_12_13_helper_function(edf_path_list)
    subjects = []
    #skip_list = ['subject_13']
    
    ecg_al_list = []
    ppg_as_list = []
    
    #aligned_df_ecg = AllignECGandPPG().AllignOneSubject(edf_path=edf_path_list[-1], annotation_path=ann_path_list[-1],ecg_or_ppg='ecg')
    #aligned_df_ecg,_ = AllignECGandPPG()._mergeSubject13(edf_path_list[-1])#.AllignOneSubject(edf_path=edf_path_list[-1], annotation_path=ann_path_list[-1],ecg_or_ppg='ecg')
    #t = aligned_df_ecg['time']
    '''
    ecg_time_stamps = aligned_df_ecg['time']
    ecg_time_stamps_long = np.hstack(ecg_time_stamps)
    ecg_label =aligned_df_ecg['labels']
    ecg_label_pr_stamp = [[x]*6000 for x in ecg_label]
    ecg_label_pr_stamp = np.hstack(np.array(ecg_label_pr_stamp)) 
    '''
    #plt.figure()
    #
    #plt.plot(t,np.arange(0,len(t),1))
    
    #exit()
    if save_output == True:
        save_folder = os.path.join(os.getcwd(),'aligned_sleep_data_set')
        ppg_save_folder = os.path.join(save_folder,'ppg')
        ecg_save_folder  = os.path.join(save_folder,'ecg')
        plot_save_folder = os.path.join(save_folder,'alignment_figures')
        os.makedirs(ppg_save_folder,exist_ok = True)
        os.makedirs(ecg_save_folder,exist_ok = True)
        os.makedirs(plot_save_folder,exist_ok = True)

    
    for ann_file in tqdm(ann_path_list,position=0,leave=True):
        #print(ann_file)
        current_subject = ann_file.split(os.sep)[-1].split('_manual')[0]
        #if current_subject not in skip_list:#!= 'subject_11':#
        #    continue
        subjects.append(current_subject)
        
        print(current_subject)        
        empatica_path = [s for s in empatica_path_list if current_subject in s][0]
        print(empatica_path)

        aligned_df_ppg = AllignECGandPPG().AllignOneSubject(ppg_path=empatica_path, annotation_path=ann_file,ecg_or_ppg='ppg')
        ppg_as_list.append(aligned_df_ppg)
        
        
        edf_path = [s for s in edf_path_list if current_subject in s][0]
        print(edf_path)
        aligned_df_ecg = AllignECGandPPG().AllignOneSubject(edf_path=edf_path, annotation_path=ann_file,ecg_or_ppg='ecg')
        ecg_al_list.append(aligned_df_ecg)        
        #break
        
        if save_output == True:
            aligned_df_ppg.to_pickle(os.path.join(ppg_save_folder,current_subject+'_ppg.pkl'))
            aligned_df_ecg.to_pickle(os.path.join(ecg_save_folder,current_subject+'_ecg.pkl'))

    
        if generate_plots == True:
    
            ppg_time_stamps = aligned_df_ppg['time']    
            ppg_time_stamps_long = np.hstack(ppg_time_stamps)
            ppg_label =aligned_df_ppg['labels']
            ppg_label_pr_stamp = [[x]*1920 for x in ppg_label]
            ppg_label_pr_stamp = np.hstack(np.array(ppg_label_pr_stamp))
            
            ecg_time_stamps = aligned_df_ecg['time']
            ecg_time_stamps_long = np.hstack(ecg_time_stamps)
            ecg_label =aligned_df_ecg['labels']
            ecg_label_pr_stamp = [[x]*6000 for x in ecg_label]
            ecg_label_pr_stamp = np.hstack(np.array(ecg_label_pr_stamp))    
            
            if 'subject_12' in edf_path:
                
                ecg_original_df,initial_timestamp = AllignECGandPPG()._mergeSubject12(edf_path)    
                ecg_original_timestamps = ecg_original_df['time']
            elif 'subject_13' in edf_path: 
                ecg_original_df,initial_timestamp = AllignECGandPPG()._mergeSubject13(edf_path)    
                ecg_original_timestamps = ecg_original_df['time']
            else:
                edf_meta = mne.io.read_raw_edf(edf_path,verbose=0,include=["time",'ECG'],preload=False)
                ecg_original_df = edf_meta.to_data_frame()
                initial_timestamp = edf_meta.info['meas_date']
                ecg_original_df = AllignECGandPPG().ECGRelativeTimeToAbsoluteTime(ecg_original_df,initial_timestamp)    
                ecg_original_timestamps = ecg_original_df['time']
            
            
        
            ppg_original,_ = AllignECGandPPG()._loadPPGcsv(empatica_path)
            annotation_original = pd.read_excel(ann_file)
            # Create subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # First subplot for PPG
            axs[0].plot(ppg_time_stamps_long, ppg_label_pr_stamp)
            axs[0].text(ppg_time_stamps_long[0], 4, 'Start time:'+str(ppg_time_stamps_long[0]) + '\nEnd time:'+str(ppg_time_stamps_long[-1]) +'\nTotal duration:'+str(np.timedelta64(ppg_time_stamps_long[-1]-ppg_time_stamps_long[0],'m')), fontsize=10,
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            axs[0].set_title('PPG Data')
            axs[0].set_xlabel('Time Stamps')
            axs[0].set_ylabel('PPG Label PR Stamp')
            
            # Second subplot for ECG
            axs[1].plot(ecg_time_stamps_long, ecg_label_pr_stamp)
            axs[1].text(ecg_time_stamps_long[0], 4, 'Start time:'+str(ecg_time_stamps_long[0]) + '\nEnd time:'+str(ecg_time_stamps_long[-1]) +'\nTotal duration:'+str(np.timedelta64(ecg_time_stamps_long[-1]-ecg_time_stamps_long[0],'m')), fontsize=10,
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            axs[1].set_title('ECG Data')
            axs[1].set_xlabel('Time Stamps')
            axs[1].set_ylabel('ECG Label PR Stamp')
            
            axs[2].plot(ecg_original_timestamps, [1]*len(ecg_original_timestamps),color = 'blue',label ='ECG')
            
            axs[2].set_title('Data original durations')
            axs[2].set_xlabel('Time Stamps')
            axs[2].set_ylabel('Data Source')
            
            axs[2].plot(ppg_original['time'], [2]*len(ppg_original),color = 'red',label ='PPG')
            axs[2].plot(annotation_original['Event Start Time'], [3]*len(annotation_original),color = 'green',label ='Annotation')
            #axs[2].legend()
            axs[2].set_ylim(-2,10)
            
            np.timedelta64(ppg_time_stamps_long[-1]-ppg_time_stamps_long[0],'m')
            ecg_duration_string = '\nTotal duration: '+str(np.timedelta64(ecg_original_timestamps.iloc[-1]-ecg_original_timestamps.iloc[0],'m'))
            ppg_duration_string = '\nTotal duration: '+str(np.timedelta64(ppg_original['time'].iloc[-1]-ppg_original['time'].iloc[0],'m'))
            ann_duration_string = '\nTotal duration: '+str(np.timedelta64(annotation_original['Event Start Time'].iloc[-1]-annotation_original['Event Start Time'].iloc[0],'m'))
        
            text_box_string_1 = 'ECG Time:\n'+'Start time:'+str(ecg_original_timestamps.iloc[0]) + '\nEnd time:'+str(ecg_original_timestamps.iloc[-1]) +'\nTotal duration:'+ecg_duration_string 
            text_box_string_2 = 'PPG Time:\n'+'Start time:'+str(ppg_original['time'].iloc[0]) + '\nEnd time:'+str(ppg_original['time'].iloc[-1]) +'\nTotal duration:'+ppg_duration_string 
            text_box_string_3 = 'Annotation Time:\n'+'Start time:'+str(annotation_original['Event Start Time'].iloc[0]) + '\nEnd time:'+str(annotation_original['Event Start Time'].iloc[-1]) +'\nTotal duration:'+ann_duration_string
            
            axs[2].text(ecg_original_timestamps.iloc[0], 5,text_box_string_1 , fontsize=10,
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            axs[2].text(ecg_original_timestamps.iloc[int(len(ecg_original_timestamps)*2/5)], 5,text_box_string_2, fontsize=10,
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            axs[2].text(ecg_original_timestamps.iloc[int(len(ecg_original_timestamps)*4/5)], 5,text_box_string_3, fontsize=10,
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
            axs[2].set_yticks([1,2,3])
            axs[2].set_yticklabels(['ECG','PPG','Annotaion'])
            plt.suptitle(current_subject)
        
            plt.tight_layout()
            plt.savefig(os.path.join(plot_save_folder,current_subject+'.pdf'))
            
            
if __name__ == "__main__":

    root_folder = r'C:\Users\Shagen\OneDrive - Aalborg Universitet\Dokumenter\PhD\Sleep Study Dataset'
    #generate_plots = True
    #save_output = True

            
    GenerateAlignedECGandPPGdatasets(root_folder)
    
