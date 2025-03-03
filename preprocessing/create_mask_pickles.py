import os
import glob
from tqdm import tqdm
import random
from dataloader_ecg_respiration_sleep_staging import ECGPeakMask
import pickle


random.seed(42)
     
    
def createDataMasks(subject:str):
    one_recording = ECGPeakMask(subject,number_of_sleep_stage=5,hz=256)
    X = one_recording.processed_ECG_dataset    
    Y = one_recording.labels   
    d = {"X":X,"Y":Y}
    with open(os.path.join(r'D:\sleep_classification_software\weird',one_recording.subject_name), 'wb') as handle:
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
    
    all_subject = glob.glob(r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa_ecg_mask/*')
    all_subject = [x.split(os.sep)[-1].split('.')[0] for x in all_subject]
    all_processed_subjects = glob.glob(r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa_ecg_mask/*')
    all_processed_subjects = [x.split(os.sep)[-1].split('.')[0] for x in all_processed_subjects]
    missing_subjects = list(set(all_subject) - set(all_processed_subjects))
    
    #num_processes = multiprocessing.cpu_count()
    # Generators  
    training_path_mesa = r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa/mesa/*'#os.path.join("non-augmented","processed_data_train",'mesa',"*")
    output_folder = r"/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa_ecg_mask/*"
    #training_path_mesa = r"/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa/mesa/*"
    #output_folder = r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa_ecg_mask/*'
    training_fileNames_mesa = glob.glob(training_path_mesa)
    existing_files = glob.glob(output_folder)
    f1 = [x.split(os.sep)[-1].split('.')[0] for x in training_fileNames_mesa]
    f2 = [x.split(os.sep)[-1].split('.')[0] for x in existing_files]
    f3 = list(set(f1) ^ set(f2))
    #f4 = [r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa/mesa'+x+'.pkl' for x in f3]
    f4 = [r'/home/cs.aau.dk/fq73oo/data_loader/sleep_data/processed_mesa/mesa/'+x+'.pkl' for x in f3]
    print(f4)
    #f4 = [1,2,3,4]

    #test_1 = ECGDataSetSingle2(r'D:\processed_mesa\mesa\mesa-sleep-3013.pkl')
    
    createDataMasks(r'D:\processed_mesa\mesa\mesa-sleep-3013.pkl')

    #createData(f4[0])

    #pool = multiprocessing.Pool(processes=4)
    #results = pool.map(createDataMasks, f4)
    #pool.close()
    #pool.join()    
    # for subject in tqdm(f4,total=len(f4)):
    #      try:
    #          createDataMasks(subject)
    #      except Exception as e:
    #        #msg = "Couldn't do create masks for subject: " + str(subject)
    #        #raise Exception(msg) from e 
    #        print(f"Error: Couldn't create masks for subject {subject}. Exception: {e}")
    
    

    #one_recording = ECGPeakMask(f4[1],number_of_sleep_stage=5,hz=256)
    
    
    #one_recording = ECGPeakMask('D:\\processed_mesa\\mesa\\\\mesa-sleep-0079.pkl',number_of_sleep_stage=5,hz=256)

