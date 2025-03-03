import torch
import numpy as np
import os
import glob
import os
from datetime import datetime
import sys
from sklearn.utils import class_weight
import random
random.seed(42)

    
    
def setCWDToScriptLocation():
    pathname = os.path.dirname(sys.argv[0])
    if pathname == "":
        print("No filename")
        return
    os.chdir(pathname)
    print("Current working directory set to: ",os.getcwd())    
    return

def experiment_folder_path(subfolder = "experiment_number",root_folder = "experiment"):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        folder = os.path.join(root_folder,subfolder+"_"+current_time)
        isExist = os.path.exists(folder)
        if not isExist:         
          # Create a new directory because it does not exist 
          os.makedirs(folder,exist_ok = True)
          print("New output folder named ",folder, " created")
        return folder
    

def get_weights(dataset, batch_size = 1):
    list_of_labels = []
    generator = torch.utils.data.DataLoader(dataset,batch_size=batch_size)

    for x, y, file_name in generator:
        if torch.cuda.is_available():   
            x, y = x.to('cuda'), y.to('cuda')

        x = x.squeeze(0)
        y = y.squeeze()
        list_of_labels.append(y.cpu().numpy())
    flat_list = [item for sublist in list_of_labels for item in sublist]
    labels = np.array(flat_list)
    weights = class_weight.compute_class_weight(class_weight ="balanced", classes= np.unique(labels),y=labels)  
    return weights

def get_class_proportional_weights(dataset, batch_size = 1):
    list_of_labels = []
    generator = torch.utils.data.DataLoader(dataset,batch_size=batch_size)

    for x, y, file_name in generator:
        if torch.cuda.is_available():   
            x, y = x.to('cuda'), y.to('cuda')

        x = x.squeeze(0)
        y = y.squeeze()
        list_of_labels.append(y.cpu().numpy())
    flat_list = [item for sublist in list_of_labels for item in sublist]
    labels = np.array(flat_list)
    unique, counts = np.unique(labels, return_counts=True)
    weights = counts/len(labels)
    #weights = class_weight.compute_class_weight(class_weight ="balanced", classes= np.unique(labels),y=labels)  
    return weights


def getDistributionOfGeneratorLabels(generator):
    
    augment_label_list = []
    for batch, labels, file_name in generator:
        
        if torch.cuda.is_available():   
            batch, labels = batch.to('cuda'), labels.to('cuda')  
        
        batch = batch.squeeze(0)
        labels = labels.squeeze()
        augment_label_list.append(labels.cpu().detach().numpy())
    
    
    augment_unique, augment_counts = np.unique(np.concatenate(augment_label_list), return_counts=True)    
    d = dict(zip(augment_unique, augment_counts))
    
    return d
    
if __name__ == '__main__':
    None
