def getLabelMappings(number_of_sleep_stage:int,tick_name:str = "ticks",dict_name:str="mapping",light_sleep_vs_all_bool:bool = False):
    if light_sleep_vs_all_bool == False:
        if number_of_sleep_stage == 2:
            label_dict = {tick_name:{"tick":[0,1],"label":["Wake","Sleep"]},
                          dict_name:{"Wake":0,"Sleep":1}}
            
        if number_of_sleep_stage == 3:
            # Group into Wake, NREM, REM
            label_dict = {tick_name:{"tick":[0,1,2],"label":["Wake","NREM","REM"]},
                          dict_name:{"Wake":0,"NREM":1,"REM":2}}
        if number_of_sleep_stage == 4:
            # Group into Wake, Light, Deep, REM
            label_dict = {tick_name:{"tick":[0,1,2,3],"label":["Wake","Light","Deep","REM"]},
                          dict_name:{"Wake":0,"Light":1,"Deep":2,"REM":3}}
        if number_of_sleep_stage == 5:
            # Group into Wake, NREM, REM
            label_dict = {tick_name:{"tick":[0,1,2,3,4],"label":["Wake","N1","N2","N3","REM"]},
                          dict_name:{"Wake":0,"N1":1,"N2":2,"N3":3,"REM":4}}          
    if light_sleep_vs_all_bool == True:
        if number_of_sleep_stage == 2:
            label_dict = {tick_name:{"tick":[0,1],"label":["Light","NotLight"]},
                          dict_name:{"Light":0,"NotLight":1}}    
        else:
            raise Exception("Light vs NotLight wash chosen but label_dict was unable to be created. number_of_sleep_stage was set to: "+str(number_of_sleep_stage))
    return label_dict

def CollapseSleepStages(y, number_of_sleep_stage = 5):
    
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


def LightSleepVsAllSleepStageCollapse(y):
    #N1 = 1, N2= 2
    y[y==5] = 4 # Set to the standard 5 stages labels
    offset = 50
    y = y +offset # Offset all labels to avoid problems
    y[y==1+offset] = 0#Set N1 to 0
    y[y==2+offset] = 0#Set N2 to 0
    y[y==offset] = 1#Set Wake to 0
    y[y==3+offset] = 1#Set N3 to 0
    y[y==4+offset] = 1#Set Rem to 0
    return y        

def CollapseSleepStagesFromPaper(y, number_of_sleep_stage = 5):
    """
    Look at function getLabelMappingsFromPaper to see which paper

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    number_of_sleep_stage : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    #y = y-1
    #print(type(y),y.max())
    if y.max()==5:
        y = y-1
    if number_of_sleep_stage == 2:
        y[y!=4] = 0 # Group all sleep into one class
        y[y==4] = 1 # Group all sleep into one class
    if number_of_sleep_stage == 3:
        # Group into Wake, NREM, REM
        y[y==0] = 0
        y[y==1] = 0
        y[y==2] = 0
        y[y==3] = 1
        y[y==4] = 2 # Group all sleep into one class

    if number_of_sleep_stage == 4:
        # Group into Wake, Light, Deep, REM
        y[y==2] = 1
        y[y==3] = 2
        y[y==4] = 3            
 
    if number_of_sleep_stage == 5:
        # Group into Wake, NREM, REM
        None
    return y    
    
def getLabelMappingsFromPaper(number_of_sleep_stage:int,tick_name:str = "ticks",dict_name:str="mapping"):
    """
    Labels originate from this github repo and paper https://github.com/bdsp-core/ecg_respiration_sleep_staging

    Parameters
    ----------
    number_of_sleep_stage : int
        DESCRIPTION.
    tick_name : str, optional
        DESCRIPTION. The default is "ticks".
    dict_name : str, optional
        DESCRIPTION. The default is "mapping".

    Returns
    -------
    label_dict : dict
        DESCRIPTION.

    """

    if number_of_sleep_stage == 2:
        label_dict = {tick_name:{"tick":[0,1],"label":["Sleep","Wake"]},
                      dict_name:{"Sleep":0,"Wake":1}}
        
    if number_of_sleep_stage == 3:
        # Group into Wake, NREM, REM
        label_dict = {tick_name:{"tick":[0,1,2],"label":["NREM","REM","Wake"]},
                      dict_name:{"NREM":0,"REM":1,"Wake":2}}
    if number_of_sleep_stage == 4:
        # Group into Wake, Light, Deep, REM
        label_dict = {tick_name:{"tick":[0,1,2,3],"label":["Deep","Light","REM","Wake"]},
                      dict_name:{"Deep":0,"Light":1,"REM":2,"Wake":3}}
    if number_of_sleep_stage == 5:
        # Group into Wake, NREM, REM
        label_dict = {tick_name:{"tick":[0,1,2,3,4],"label":["N3","N2","N1","REM","Wake"]},
                      dict_name:{"Wake":4,"REM":3,"N1":2,"N2":1,"N3":0}}
    return label_dict


def fromPaperLabelsToMyLabels(y):
    """
        Needs to be 5 stages first. Paper labels are:
            "Wake":4
            "REM":3
            "N1":2
            "N2":1
            "N3":0
        My labels are :
            "Wake":0
            "N1":1
            "N2":2
            "N3":3
            "REM":4    
    """
    offset = 50
    y = y +offset # Offset all labels to avoid problems
    y[y==4+offset] = 0#Wake
    y[y==2+offset] = 1#N1
    y[y==1+offset] = 2#N2
    y[y==0+offset] = 3#N3
    y[y==3+offset] = 4#REM
    return y

        
    