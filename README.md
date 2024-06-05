# Arrhythmia detection of 12-lead ECG using deep learning techniques based on the PTB-XL dataset

## Abstract
Arrhythmia detection is crucial for early diagnosis and treatment of various heart diseases. This paper proposes a CNN with an attention wise mechanism to classify two-class and five-class arrhythmia using PTB-XL dataset. Upsampling is used to handle imbalanced features, and cross-validation is employed for evaluation. The model achieves 84.0% and 88.8% accuracy for two-class and five-class arrhythmia detection, respectively, and is compared to four other academic papers. These results demonstrate the effectiveness of the proposed approach for arrhythmia identification.

![img](https://github.com/Bettycxh/DS4Healthcare-Group-4/blob/main/architecture.png)


## Requirements
- Python==3.6
- Keras==2.3.1
- TensorFlow==1.14.0

## Data Preparation
To download the dataset used for training and testing, please refer to [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.1/)

- Download the [ZIP file](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip)
- Run [example_physionet.py](https://physionet.org/content/ptb-xl/1.0.1/example_physionet.py) to get the data

## Usage
To train the model, execute the python file

- Five classes detection (Normal, MI, STTC, CD, HYP)
  Run [train_5.py](https://github.com/Bettycxh/DS4Healthcare-Group-4/blob/main/train_5.py)

- Two classes detection (Normal, Arrhythmia)
  Run [train_2.py](https://github.com/Bettycxh/DS4Healthcare-Group-4/blob/main/train_2.py)
