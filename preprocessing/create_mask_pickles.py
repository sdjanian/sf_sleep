import os
import glob
from tqdm import tqdm
import random
from dataloader_ecg_respiration_sleep_staging import ECGPeakMask
import pickle


random.seed(42)
     
    
def createDataMasks(subject:str,output_folder = "processed_mesa_ecg_mask"):
    one_recording = ECGPeakMask(subject,number_of_sleep_stage=5,hz=256)
    X = one_recording.processed_ECG_dataset    
    Y = one_recording.labels   
    d = {"X":X,"Y":Y}
    with open(os.path.join(output_folder,one_recording.subject_name), 'wb') as handle:
        pickle.dump(d, handle,protocol=pickle.HIGHEST_PROTOCOL) 

def __collapseSleepStage(y,number_of_sleep_stage):
    
    if number_of_sleep_stage == 2:
        y[y!=0] = 1 # Group all sleep into one class
    if number_of_sleep_stage == 3:
        # Group into Wake, NREM, REM
        y[y==2] = 1
        y[y==3] = 1
        y[y==5] = 2
    if number_of_sleep_stage == 4:
        # Group into Wake, Light, Deep, REM
        y[y==2] = 1
        y[y==3] = 2
        y[y==5] = 3            
 
    if number_of_sleep_stage == 5:
        # Group into Wake, NREM, REM
        y[y==5] = 4
    return y    


if __name__ == '__main__':
    
    training_path_mesa = r'D:\processed_mesa_test\mesa\\'
    output_folder = r"D:\processed_mesa_ecg_mask_test\\"
    os.makedirs(output_folder, exist_ok=True)

    training_fileNames_mesa = glob.glob(os.path.join(training_path_mesa,"*"))
    existing_files = glob.glob(os.path.join(output_folder,"*"))
    f1 = [x.split(os.sep)[-1].split('.')[0] for x in training_fileNames_mesa]
    f2 = [x.split(os.sep)[-1].split('.')[0] for x in existing_files]
    f3 = list(set(f1) ^ set(f2))
    f4 = [training_path_mesa+x+'.pkl' for x in f3]
    print(f4)

    for subject in tqdm(f4,total=len(f4)):
        try:
            createDataMasks(subject,output_folder)
        except Exception as e:
            print(f"Error: Couldn't create masks for subject {subject}. Exception: {e}")
