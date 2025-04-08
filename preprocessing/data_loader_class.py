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


class LoadSleepData:
    """
    Class to load and preprocess .edf files from the MESA and SHHS1 and SHHS2 datasets. The resulting files that are 
    chuncked dataframes that contain 40 epochs each. The epochs are rowwise and each column is an ECG sample.
    The column "labels" is the sleep labels. These are what the follwing labels correspond to:
        0: Wake
        1: Stage 1 sleep
        2: Stage 2 sleep
        3: Stage 3 sleep
        5: REM sleep


    Methods
    ----------
    loadSingleFile : 
        Loads a single .edf file as a DataFrame containing epochs rowwise and samples column wise.
    
    runAllFiles: 
        Processes all the .edf files in the given directory and saves chunked op dataFrames of the ECG signal.
        
    GetListOfHealthyParticipants:
        Returns a list of all the particiaps in either MESA, SHHS1 or SHHS2 that are healthy.
        

    """

    def __init__(self,dataset = "", root_folder:str = "sample_dataset", output_folder:str = "processed_data",pre_root:str = os.getcwd()):
        """
        Initialization of the LoadSleepData class.

        Parameters
        ----------
        dataset : str, optional
            name of which dataset is being processed. Only acceptable input is:
                'mesa'
                'shhs1'
                'shhs2'
            
        root_folder : str, optional
            The root folder from where to parse data. The folder must have contain the original structure from MESA and SHHS respectivly.
            Example is:
                ./root_folder/
                            mesa/
                                datasets/
                                polysomnography/
                            shhs/
                                datasets/
                                polysomnography/                            
                
                The default is "sample_dataset" which contains a smaller version for easier testing.

         output_folder : str, optional
             Name of the output folder. Default is "processed_data":

        """
        
        if dataset == "mesa":
            self.dataset_name = "mesa"
            self.sampling_frequency = 256
            self.ecg_col_name = "EKG"
            self.signal_folder_path = os.path.join(pre_root,root_folder,'mesa','polysomnography','edfs')
            self.annotation_folder_path = os.path.join(pre_root,root_folder,'mesa','polysomnography','annotations-events-nsrr')
            self.summary_datasets_path = os.path.join(pre_root,root_folder,'mesa','datasets')
        
        elif dataset == "shhs1":
            self.dataset_name = "shhs1"
            self.sampling_frequency = 125
            self.ecg_col_name = "ECG"
            self.signal_folder_path = os.path.join(pre_root,root_folder,'shhs','polysomnography','edfs','shhs1')
            self.annotation_folder_path = os.path.join(pre_root,root_folder,'shhs','polysomnography','annotations-events-nsrr','shhs1')
            self.summary_datasets_path = os.path.join(pre_root,root_folder,'shhs','datasets')
        elif dataset == "shhs2":
            self.dataset_name = "shhs2"
            self.sampling_frequency = 250
            self.ecg_col_name = "ECG"
            self.signal_folder_path = os.path.join(pre_root,root_folder,'shhs','polysomnography','edfs','shhs2')
            self.annotation_folder_path = os.path.join(pre_root,root_folder,'shhs','polysomnography','annotations-events-nsrr','shhs2')
            self.summary_datasets_path = os.path.join(pre_root,root_folder,'shhs','datasets')
        else:
            raise Exception("Wrong dataset chosen. Choose between 'mesa', 'shh1' and 'shhs2'") 
        
        self.epoch_length_in_sec = 30
        self.samples_in_epoch = self.sampling_frequency * self.epoch_length_in_sec
        

    def __loadEDFtoDataFrame(self,path:str) ->pd.DataFrame:
        """
        Loads a single .edf file and converts it to a pd.DataFrame

        Parameters
        ----------
        path : str
            Path to .edf file.

        Returns
        -------
        df : pd.DataFrame
            .edf file as pd.DataFrame.

        """
        
        data = mne.io.read_raw_edf(path,verbose=0,include=["time",self.ecg_col_name])
        df = data.to_data_frame()  
        return df
    
    def __parseAnnotationXMLtoDataFrame(self,ann_path:str) -> pd.DataFrame:
        """
        Parses the annotation file that contains sleep event labels. The annotaton file is stored as an XML.
        The labels and corresponding start and duration times of each event is extracted and returned in a DataFrame.
        Time units are in miliseconds

        Parameters
        ----------
        ann_path : str
            path to the XML annotation file.

        Returns
        -------
        ann_df : pd.DataFrame
            DataFrame containing ['Sleep Stage','Label','Start','Duration'] of each event.

        """
        xtree = et.parse(ann_path)
        xroot = xtree.getroot()
        
        # Parse through xml for MESA annotation to extract event df
        sleep_stage_lst = []
        label_lst = []
        start_lst = []
        duration_lst = []
        for eType,eConcept,start,duration in zip(xroot.iter('EventType'),
                          xroot.iter('EventConcept'),
                          xroot.iter('Start'),
                          xroot.iter('Duration')):
            if eType.text == 'Stages|Stages':
                #print("step4")
                concept = eConcept.text.split('|') 
                sleep_stage_lst.append(concept[0])
                label_lst.append(int(concept[1]))
                start_lst.append(int(float(start.text)))
                duration_lst.append(int(float(duration.text)))
                #print(eType.text, eConcept.text, start.text, duration.text)
        ann_df = pd.DataFrame(list(zip(sleep_stage_lst, label_lst, start_lst, duration_lst)),
                              columns = ['Sleep Stage','Label','Start','Duration'])
        
        # Convert from seconds to miliseconds
        #ann_df["Start"] =  ann_df["Start"]*1000
        #ann_df["Duration"] =  ann_df["Duration"]*1000        
        return ann_df

    def __chooseECGSignal(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Returns only the ECG signal and timestamps

        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        ecg_df : pd.DataFrame
            DataFrame containing timestamps and ECG samples.

        """
        ecg_df = df[["time",self.ecg_col_name]]
        return ecg_df
          
    def __generateListOfEpochNumbersForEachSample(self, length_of_vector):
    # Create a vector of epoch numbers. Epoch number starts at 1
        epoch_list = []
        for i in range(1,int(np.floor(length_of_vector/self.samples_in_epoch))):
            epoch_list.append([i]*self.samples_in_epoch)     
        flat = list(itertools.chain.from_iterable(epoch_list))    
        return flat
    
    def __appendEpochListToDataFrame(self, epoch_list, df):
        """
        Append a flat list of epoch numbers for each sample to dataframe. If lengths don't match the rest will be filled with NaNs

        """
        return pd.concat([df,pd.Series(epoch_list,name="Epochs")],axis=1)
        
    def __appendSleepLabelsToEachSampleInDataFrame(self, signal_df:pd.DataFrame, annotation_df:pd.DataFrame) -> pd.DataFrame:
        """
        

        Parameters
        ----------
        signal_df : pd.DataFrame
            DataFrame containing the signal and timestamps.
        annotation_df : pd.DataFrame
            DataFrame containin the annotation for each event.

        Returns
        -------
        signal_df : pd.DataFrame
            returns signal_df with a new column "labels" that contains the epoch number for each timestamp

        """
        signal_df["labels"] = np.nan

        for index, annotation_row in annotation_df.iterrows():
            signal_df["labels"][signal_df["time"].between(annotation_row["Start"],
                                                    annotation_row["Start"]+annotation_row["Duration"])] = annotation_row['Label']
        
        return signal_df
        
    def __createEpochDataFrameWithAnnotations(self,df):
    
        def select_columns_func(x):
            signal = x[self.ecg_col_name].values
            labels = x["labels"].iloc[0]
            time = x["time"].values
            return{"signal":signal,"labels":labels,"time":time}
        
        temp = df.groupby(["Epochs"]).apply(select_columns_func)
        epoch_df = pd.DataFrame(list(temp.values))    
        return epoch_df

    def __runProcess(self,signal_path:str,annotation_path:str,augment_data:bool=False) -> pd.DataFrame:
        """
        Performs all transformations from .edf and annotation file into a single DataFrame that row wise contains
        the 30 second epochs. The columns contains each ECG sample. The "label" column contains the sleep stage.
        The process runs as follows:
            * Loads the .edf file as a pd.DataFrame and creates an complimentary DataFrame with annotations.
            * The ECG signal is extracted from the dataframe and a list of epochs is created so each sample has an associated epoch number.
            * Finally the DataFrame is formated such that each row contains an epoch and all the columns are the samples of the epoch. 
            The width of the DataFrame depends on the sampling frequency.

        Parameters
        ----------
        signal_path : str
            path to the .edf file containing the raw PSG data.
        annotation_path : str
            path to the annotation XML file that contains sleep stage labels for each event.
        augment_data : bool
            create extra casses of sleep stages 2, 3 and REM by augmenting the dataset.            

        Returns
        -------
        epoch_df_stacked : pd.DataFrame
            DataFrame containing epochs rowwise and samples columnwise. Last column is "labels" which contains the sleep stage.

        """
        df = self.__loadEDFtoDataFrame(signal_path)
        annotation_df = self.__parseAnnotationXMLtoDataFrame(annotation_path)
        signal_df = self.__chooseECGSignal(df)
        epoch_list = self.__generateListOfEpochNumbersForEachSample(len(signal_df))
        signal_df_epoch_list = self.__appendEpochListToDataFrame(epoch_list,signal_df)
        signal_df_epoch_list = self.__appendSleepLabelsToEachSampleInDataFrame(signal_df_epoch_list,annotation_df)
        epoch_df = self.__createEpochDataFrameWithAnnotations(signal_df_epoch_list)
        epoch_df_stacked = self.__epochDfStack(epoch_df)
        #del df, annotation_df, signal_df_epoch_list,epoch_df
        if 9 in epoch_df_stacked['labels'].values:
            print('Removing rows with sleep stage 9')
            epoch_df_stacked = epoch_df_stacked[epoch_df_stacked['labels'] != 9] # File sleep-4816 and mesa-sleep-2292 contains 9 which is an unlabeled sleep event. mesa-sleep-2292 only contains wake and N2
            epoch_df_stacked = epoch_df_stacked.reset_index()
            
        if augment_data == True:
            print("Creating augmented dataset")
            epoch_df_stacked = self.__augmentDataset(epoch_df_stacked)
        print("contains nan: ", epoch_df_stacked["labels"].isnull().values.any())
        return epoch_df_stacked


    def __epochDfStack(self,epoch_df):
        stack = np.stack(epoch_df["signal"].values)
        new_df = pd.DataFrame(stack)
        new_df["labels"] = epoch_df["labels"].values
        return new_df
    
    def loadSingleFile(self,signal_path:str,annotation_path:str,augment_data:bool=False) -> pd.DataFrame:
        """
        Loads a single .edf file DataFrame containing epochs rowwise and samples columnwise.

        Parameters
        ----------
        signal_path : str
            Direct path to the .edf file.
        annotation_path : str
            Direct path to the XML file.
        augment_data : bool
            create extra casses of sleep stages 1, 3 and REM by augmenting the dataset.  
        Returns
        -------
        pd.DataFrame
            DataFrame containing epochs rowwise and samples columnwise. Last column is "labels" which contains the sleep stage

        """
        
        return self.__runProcess(signal_path,annotation_path,augment_data)
    
    def runAllFiles(self,save_as_chunks:bool = True, chunk_size:int = 40, overwrite:bool = False, only_healthy_subjects:bool = True, root_output_folder:str = "processed_data",augment_data:bool=False):
        """
        Loops through all files in the given dataset directory and performs the process of
        converting from .edf files into chuncked .pkl files where rows are epochs and columns are ECG samles.

        Parameters
        ----------
        save_as_chunks : bool, optional
            Parameter to save the dataframes as chuncks instead of complete file. The default is true.        
        chunk_size : int, optional
            Parameter to determine the number of sleep epochs in a single file. The default is 40.
        overwrite : bool, optional
            DESCRIPTION. The default is False.
        only_healthy_subjects : bool, optional
            Only use healthy subjects. The default is True.
        root_output_folder : str, optional
            Name of the root of the outputfolder. Subfolders are named after their dataset name. The default is "processed_data".           
        augment_data : bool
            create extra casses of sleep stages 1, 3 and REM by augmenting the dataset.              

        Returns
        -------
        None.

        """
        output_folder = self.__createOutputFolder(root_folder = root_output_folder, output_folder =self.dataset_name)
        healthy_subject_list = self.__healthySubjectsList()
        
        #print(os.path.join(self.annotation_folder_path,"*"))
        for ann_file in tqdm(glob.iglob(os.path.join(self.annotation_folder_path,"*")),position=0,leave=True,):
            subject_file = os.path.split(ann_file)[-1].split("-nsrr.xml")[0]
            signal_file = os.path.join(self.signal_folder_path,subject_file)
            output_file_name = os.path.join(output_folder, subject_file)
            

            if (only_healthy_subjects == True) & (int(subject_file.split('-')[-1]) not in healthy_subject_list):
                print("Skipped:" + subject_file)
                continue
                
                
            #if os.path.exists(glob.glob(output_file_name+'?*')[0]) == True and overwrite == True:
            #    print("skipped file: ", output_file_name)
            #    continue 
            if ((glob.glob(output_file_name+'?*')) and overwrite == False):
                print("skipped file: ", output_file_name)
                continue         
            else:
                try:
                    print("Started recording: ", signal_file)
                    epoch_df = self.__runProcess(signal_file+".edf",ann_file,augment_data=augment_data)
                    if save_as_chunks == True:
                        split = self.__splitDataframe(epoch_df,chunk_size)
                        for count, chunk in enumerate(split):
                            chunk.to_pickle(output_file_name+"_"+str(count+1)+".pkl")       
                            print("Chunk contains nan: ", chunk["labels"].isnull().values.any())
                    else:
                        epoch_df.to_pickle(output_file_name+".pkl")       
                        print("Dataframe contains nan: ", epoch_df["labels"].isnull().values.any())
    
    
                    #epoch_df.to_pickle(output_file_name)
                    print("saved: ",output_file_name)
                except:
                    print("failed: ",subject_file)
            
            
    def __createOutputFolder(self,output_folder:str="", root_folder:str ="processed_data") -> str:
        """
        Creates a folder called "processed_data" to store files in. Also creates sub folders for each dataset.

        Parameters
        ----------
        output_folder : str, optional
            DESCRIPTION. The default is "".
        root_folder : str, optional
            DESCRIPTION. The default is "processed_data".            

        Returns
        -------
        str
            name of the created folder.

        """
        root = root_folder
        folder = os.path.join(root,output_folder)
        isExist = os.path.exists(folder)
        if not isExist:         
          # Create a new directory because it does not exist 
          os.makedirs(folder)
          print("New output folder named ",folder, " created")
        return folder
    
    @staticmethod
    def __splitDataframe(df:pd.DataFrame, chunk_size:int = 40): 
        """
        Splits dataframe into chunck of size chunk_size. The last chunk will be the size of whatefer remains
        Source: https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be chuncked.
        chunk_size : int, optional
            DESCRIPTION. The default is 40.

        Returns
        -------
        chunks : List
            List of the dataframes that are of size chunk_size.

        """
        chunks = list()
        num_chunks =  num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks    
    
    def __healthySubjectsList(self, ahi_limit = 5):
        """
        Creates a list that only contains healthy participants. This end up being 179 subjects for MESA, 868 for SHSS1,
        and 641 for SHHS2. MESA, SHHS1 and SHHS2 have used different screeing questions
        and variables. Full list variables can be found here:
            https://sleepdata.org/datasets/mesa/variables?folder=Sleep+Questionnaires
            https://sleepdata.org/datasets/shhs/variables?folder=Sleep+Questionnaires
        
        For MESA we used the following criteria:
            A full scoring, Apnea-Hypopnea Index below ahi_limit (5) and no diagnosed sleep apnea,
            restless legsyndrom and insomnia.
            
        For SHHS1 we used to followin criteria:
            A full scoring, Apnea-Hypopnea Index below ahi_limit (5) and no diagnosed sleep apnea.
            
        For SHHS2 we used to followin criteria:
            The healthy participants from SHHS1 and further screening of no diagnosed restless legsyndrom,
            insomnia and other sleep disorders.

        Parameters
        ----------
        ahi_limit : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        relevant_subjects : list
            List of ints containin the subject ID for healthy participants.

        """
        if self.dataset_name == "mesa":
            relevant_subjects = self.__healthySubjectsMESA(ahi_limit=ahi_limit)
        elif self.dataset_name == "shhs1":
            relevant_subjects = self.__healthySubjectsSHHS(ahi_limit=ahi_limit)
        
        elif self.dataset_name=="shhs2":                
            relevant_subjects = self.__healthySubjectsSHHS(ahi_limit=ahi_limit)
        return relevant_subjects
    
    def __healthySubjectsMESA(self,ahi_limit = 5):
        """
        :meth:`LoadSleepData.___healthySubjectsList`

        Parameters
        ----------
        ahi_limit : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        relevant_subjects : TYPE
            DESCRIPTION.

        """

        dataset_name = "mesa-sleep-dataset-0.5.0.csv"
        harmonized_dataset_name = "mesa-sleep-harmonized-dataset-0.5.0.csv"
        id_col_name = "mesaid"

        
        slp_variables = pd.read_csv(os.path.join(self.summary_datasets_path,dataset_name))
        harmonized_variables = pd.read_csv(os.path.join(self.summary_datasets_path,harmonized_dataset_name))
        apnea_variable_name = "nsrr_ahi_hp3r_aasm15"
        sleep_apnea_diagnosis_col = "slpapnea5"
        restless_leg_diagosis_col = "rstlesslgs5"
        insomnia_diagnosis_col = "insmnia5"    
        
        subjects_with_low_ahi = harmonized_variables[(harmonized_variables[apnea_variable_name]<=ahi_limit) &
                                             (harmonized_variables['nsrr_flag_spsw']=='full scoring')]
    
        subjects_with_no_disorders = slp_variables[(slp_variables[sleep_apnea_diagnosis_col]==0) &
                                             (slp_variables[restless_leg_diagosis_col]==0) &
                                             (slp_variables[insomnia_diagnosis_col]==0)]    
        relevant_subjects = list(set(subjects_with_low_ahi[id_col_name]) & set(subjects_with_no_disorders[id_col_name]))        
        return relevant_subjects
    

    def __healthySubjectsSHHS(self,ahi_limit = 5):

        shhs1_dataset_name = "shhs1-dataset-0.18.0.csv"
        harmonized_dataset_name = "shhs1-harmonized-dataset-0.18.0.csv"
        shhs2_dataset_name = "shhs2-dataset-0.18.0.csv"

        shhs1_slp_variables = pd.read_csv(os.path.join(self.summary_datasets_path,shhs1_dataset_name),encoding='mac_roman')
        shhs2_slp_variables = pd.read_csv(os.path.join(self.summary_datasets_path,shhs2_dataset_name),encoding='mac_roman')
        harmonized_variables = pd.read_csv(os.path.join(self.summary_datasets_path,harmonized_dataset_name),encoding='mac_roman')
        
        id_col_name = "nsrrid"
        apnea_variable_name = "nsrr_ahi_hp3r_aasm15"
        sleep_apnea_diagnosis_col = "mdsa02" # 0 = no
        restless_leg_diagosis_col = "sh318b" # 0 = no
        insomnia_diagnosis_col = "sh318a" # 0 = no
        narcolepsy_diagnosis_col = "sh318c" #0 = no
        other_sleep_disorder_col = "sh318d" #0 = no
        sleep_apnea_interrim_col = "sleepapnea" # not found in datasets
        other_sleep_disorder_interim = "sleepdisorder" # not found in datasets
        
        subjects_with_low_ahi = harmonized_variables[(harmonized_variables[apnea_variable_name]<=ahi_limit) &
                                             (harmonized_variables['nsrr_flag_spsw']=='full scoring')]
        
        shhs1_subjects_with_no_disorders = shhs1_slp_variables[(shhs1_slp_variables[sleep_apnea_diagnosis_col]==0)]

        shhs2_subjects_with_no_disorders = shhs2_slp_variables[(shhs2_slp_variables[narcolepsy_diagnosis_col]!=1) &
                                             (shhs2_slp_variables[restless_leg_diagosis_col]!=1) &
                                             (shhs2_slp_variables[insomnia_diagnosis_col]!=1) &
                                             (shhs2_slp_variables[other_sleep_disorder_col]!=1)]        
        if self.dataset_name == "shhs1":
            relevant_subjects = list(set(subjects_with_low_ahi[id_col_name]) & set(shhs1_subjects_with_no_disorders[id_col_name]))        

        
        elif self.dataset_name=="shhs2":                
            relevant_subjects = list(set(subjects_with_low_ahi[id_col_name]) & 
                                     set(shhs1_subjects_with_no_disorders[id_col_name]) &
                                     set(shhs2_subjects_with_no_disorders[id_col_name]))        

    
        return relevant_subjects
    
    def DebugAnn1(self,annotation_path):
        ann_df = self.__parseAnnotationXMLtoDataFrame(annotation_path)
        return ann_df
    
    def DebugSignal1(self,signal_path):
        df = self.__loadEDFtoDataFrame(signal_path)
        signal_df = self.__chooseECGSignal(df)
        epoch_list = self.__generateListOfEpochNumbersForEachSample(len(signal_df))
        signal_df_epoch_list = self.__appendEpochListToDataFrame(epoch_list,signal_df)
        return signal_df_epoch_list
    
    def GetListOfHealthyParticipants(self, ahi_limit = 5):
        return self.__healthySubjectsList(ahi_limit)
    
    def DebugSignal2(self,signal_path):
        df = self.__loadEDFtoDataFrame(signal_path)
        signal_df = self.__chooseECGSignal(df)
        return signal_df
        
        
    def __createAugmentedSignal(self,s0,s1,label,sampling_frequency:int=256,step_in_seconds:int=2,start_offset_in_seconds:int = 5,stop_offset_in_seconds:int = 4 ) -> np.ndarray:
        """
        Creates augmented ECG signals for classes sleep stage 1, 3 and REM sleep. Slides a window of 30 seconds through two conescutive epochs (total of 60 seconds)
        starting at stop_offset_in_seconds and sliding with a step size of step_in_seconds. It stops before (60 seconds - stop_offset_in_seconds). 
        S0 and S1 are concataneted and to create the 60 second signal so slide trhough. 
        
        
        Implentation of method found in 
        "Performance of a Convolutional Neural Network Derived from PPG Signal in Classifying Sleep Stages" by Habib et al. 2022
        DOI: 10.1109/tbme.2022.3219863
        

        Parameters
        ----------
        s0 : TYPE
            The first epoch of ECG data. Should be an np.array of length 30 seconds.
        s1 : TYPE
            The second epoch of ECG data. Should be an np.array of length 30 seconds.
        label : TYPE
            The sleep stage label of s0 and s1.
        sampling_frequency : int, optional
            The sampling frequency of the ECG. The default is 256.
        step_in_seconds : int, optional
            Steps of the sliding window. The default is 2.
        start_offset_in_seconds : int, optional
            The offset that the sliding begins at. The default is 5.
        stop_offset_in_seconds : int, optional
            The offset that the sliding stops at. The default is 4.

        Returns
        -------
        TYPE
            Return a numpy array where the rows are augmented epochs of ECG data. The columns are the samples and the last column is the label.

        """
        start_offset = start_offset_in_seconds*sampling_frequency
        stop_offset = stop_offset_in_seconds* sampling_frequency
        steps = step_in_seconds*sampling_frequency
        epoch_length = sampling_frequency*30
        signal_to_augment = np.concatenate([s0,s1])
        augmented_list = []
        
        for i in np.arange(start_offset,len(signal_to_augment)-stop_offset,steps):
            if i+epoch_length<=len(signal_to_augment)-stop_offset:
                augmented_list.append(np.concatenate([signal_to_augment[i:i+epoch_length],[label]]))
        return np.vstack(augmented_list)      
    
    def __augmentDataset(self,df):
        """
        Runs self.__createAugmentedSignal through the dataframe and appends the newly created epochs to the bottom of the dataframe

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
        sleep_stages_for_augmentation = [1,3,5] # 1 = NREM1 (Sleep Stage 1), 3 = NREM3 (Sleep Stage 3), 5 = REM    
        num_of_REM = df['labels'][df['labels']==5].sum()
        num_of_SS2 = df['labels'][df['labels']==2].sum()

        for index, row in df.iterrows():
            #print(row)
            if (row["labels"] in sleep_stages_for_augmentation) and (index<len(df)-1):
                if df["labels"].iloc[index]==df["labels"].iloc[index+1]:
                    #print(row["labels"])
                    s0 = df.iloc[index][0:-1].values
                    s1 = df.iloc[index][0:-1].values
                    label = row["labels"]               
                    if (label == 5) and (num_of_REM >= num_of_SS2):
                        break
                    elif label == 5:
                        num_of_REM += 1
                    augmented_signals.append(self.__createAugmentedSignal(s0,s1,label,sampling_frequency=self.sampling_frequency))
        augmented_signals_stacked = np.vstack(augmented_signals)  
        temp_df = pd.DataFrame(augmented_signals_stacked, columns=list(df))
        df = df.append(temp_df) 
        print("Augmented dataset")              
        return df

def GetlabelDictionary(dict_number:int = 1):
    dict1 = {"Wake":0,"Stage 1 sleep":1,"Stage 2 sleep":2, "Stage 3 sleep": 3, "REM sleep": 5}
    dict2 = {0:"Wake",1:"Stage 1 sleep",2:"Stage 2 sleep", 3:"Stage 3 sleep", 5:"REM sleep"}
    
    if dict_number == 1:
        return dict1
    else:
        return dict2        
    
    
if __name__ == "__main__":
    def setCWDToScriptLocation():
        pathname = os.path.dirname(sys.argv[0])
        if pathname == "":
            print("No filename")
            return
        os.chdir(pathname)
        print("Current working directory set to: ",os.getcwd())    
        return
    setCWDToScriptLocation()    


    """
    LoadSleepData('mesa',root_folder="sample_data_healthy_training").runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=True,
                                                                                 root_output_folder="augmented_balanced/processed_data_train",
                                                                                 augment_data=True)
    LoadSleepData('mesa',root_folder="sample_data_healthy_test").runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=True,
                                                                                 root_output_folder="non-augmented/processed_data_test",
                                                                                 augment_data=False)
        
    LoadSleepData('shhs1',root_folder="sample_data_healthy_training").runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=True,
                                                                                 root_output_folder="non-augmented_shhs1/processed_data_test",
                                                                                 augment_data=False)
    """
    
    LoadSleepData('mesa',root_folder="",pre_root='D:').runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=False,
                                                                                 root_output_folder='D:processed_mesa_test/',
                                                                                 augment_data=False)
    
    """
    LoadSleepData('shhs1',root_folder="shhs",pre_root='H:').runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=False,
                                                                                 root_output_folder='H:processed_shhs1/',
                                                                                 augment_data=False)
    LoadSleepData('shhs2',root_folder="shhs",pre_root='H:').runAllFiles(save_as_chunks=False,
                                                                                 overwrite=False,
                                                                                 only_healthy_subjects=False,
                                                                                 root_output_folder='H:processed_shhs2/',
                                                                                 augment_data=False)
    """
    