import torch
import os
import pickle
import random
import pandas as pd
random.seed(42)
from dataloader_utilities import getLabelMappings,CollapseSleepStages,getLabelMappingsFromPaper,CollapseSleepStagesFromPaper,LightSleepVsAllSleepStageCollapse




class ECGPeakMask2(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_path:str, shuffle_recording = False, number_of_sleep_stage:int = 5,augment_data:bool = False,normalize:bool=True,window_length_in_min:int=5, hz = 256,get_ibi:bool=False,resample_ibi = False, ibi_frequency = 4, normalize_ibi = False,light_sleep_vs_all_bool:bool = False):
        self.file_path = file_path
        self.subject_name = file_path.split(os.sep)[-1]
        self.shuffle_recording = shuffle_recording
        self.number_of_sleep_stage = number_of_sleep_stage
        self.light_sleep_vs_all_bool = light_sleep_vs_all_bool
        self.processed_ECG_dataset,self.labels = self.__prepareECGData()
        self.label_dict = self.__getLabelDict()

    def __len__(self):
        #self.processed_ECG_dataset,self.labels = self.__prepareECGData()
        return self.processed_ECG_dataset.shape[0]

    def __getitem__(self,idx):
        x = self.processed_ECG_dataset[idx]
        y = self.labels[idx]
        x = x.type(torch.FloatTensor)
        return x, y, self.subject_name
        
    def __prepareECGData(self):
        #return [self.__createXandY(path) for path in self.filenames]
        with open(self.file_path, 'rb') as handle:
            d = pickle.load(handle) 
        x = d["X"]
        x =x.bool()
        y = self.__collapseSleepStage(d['Y'])        
        return x, y
    
    def __getLabelDict(self):
        #tick_name = "ticks"
        #dict_name = ""
        if self.light_sleep_vs_all_bool==True:
            label_dict = getLabelMappings(number_of_sleep_stage = self.number_of_sleep_stage,light_sleep_vs_all_bool=self.light_sleep_vs_all_bool)
        else:
            label_dict = getLabelMappingsFromPaper(number_of_sleep_stage = self.number_of_sleep_stage)
        return label_dict     
    
    def __collapseSleepStage(self,y):
        if self.light_sleep_vs_all_bool==True:
            y = CollapseSleepStagesFromPaper(y, number_of_sleep_stage = 5) # Required to convert
            y = LightSleepVsAllSleepStageCollapse(y) #### I AM HERE! CONTINUE FROM HERE
        else:
            
            y = CollapseSleepStagesFromPaper(y, self.number_of_sleep_stage)
        return y    
    


if __name__ == '__main__':
    #test = r"H:\processed_mesa_ecg_mask\mesa-sleep-0852.pkl"
    #testdata = ECGPeakMask2(test)
    #with open(test, 'rb') as handle:
    #    d = pickle.load(handle)
    #x = d["X"]
    #y = d['Y'].cpu().numpy()

    None