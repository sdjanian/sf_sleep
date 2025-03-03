# SoundFocus Sleep
This is the GitHub repo for the paper "PAPER TITLE". It contains the training code for the SoundFocus Sleep Classification model along side the SoundFocus dataset.

# How to use

## Datasets
The full model was trained on the MESA dataset which be accessed [here](https://sleepdata.org/datasets/mesa) and the SoundFocus dataset can be downloaded from [here](). An example data set for this was created and stored in the folder "example_data" which contains small snipets of randomized data in the correct format for testing the code.

### Alligned SoundFocus data
If the folder "aligned_sleep_data_set" does not exist it can be prepared using `preprocessing/soundfocus_sleep_dataset_alignment.py` if the dataset has already been downloaded and stored as "Sleep Study Dataset"

## Installing environment
The code was written in python and a virtual environment with all the dependencies can be created using the "environment_installer.yaml" file. In the Anaconda Prompt it can be installed using `conda env create -f environment_installer.yaml`. To test if the environment is setup correctly, run `check_setup_example.py`.

## Running the code
There are two files for training the models. `deep_learning_training_loop.py` is used for pretraining the models using the MESA dataset and `deep_learning_cross_validation.py` Leave-One-Subject-Out Cross-Validation training or finetuning models using the SoundFocus dataset. To test if `deep_learning_training_loop.py` works correctly use the following command 
```python
python deep_learning_training_loop.py -b 10 -e 3 -ss 5 -dtype rawecg -m ECGSleepNetAdaptable -wloss 1 -norm 1 -shuffle 1 -optim rmsprop -resample 1 -resample_hz 64 -norm_type zscore -fname test -w_size 270 -dset example
```
To train a model from scratch using PPG data use this command after making sure the environment is installed and the SoundFocus dataset ready 
```python
python deep_learning_cross_validation.py -b 32 -e 40 -ss 5 -m ECGSleepNetAdaptable -wloss 1 -norm 1 -shuffle 1 -optim rmsprop -resample 1 -resample_hz 64 -norm_type zscore -fname PPG_sleep_model -w_size 270 -filter 1 -dtype rawppg
```
