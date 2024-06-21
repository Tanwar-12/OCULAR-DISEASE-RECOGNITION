# PROJECT TITLE : OCULAR DISEASE RECOGNITION USING DEEP LEARNING
## 1.INTRODUCTION:
_Ocular diseases, such as glaucoma, diabetic retinopathy, cataracts, and age-related macular degeneration, are major causes of visual impairment and blindness worldwide. Early detection and timely treatment are critical for preventing vision loss and improving patient outcomes. Traditionally, the diagnosis of these conditions has relied heavily on manual examination by trained ophthalmologists using various imaging modalities like fundus photography, optical coherence tomography (OCT), and fluorescein angiography._


![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/2e68cc5b-feca-4abf-aff0-1653c9aaf763)![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/e3d13449-db8d-48bd-909c-0193d7ce159c)

## 2.DATA SUMMARY

**Data Collection**

**Dataset** : https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

_Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and doctors' diagnostic keywords from doctors._

_This dataset is meant to represent ‘‘real-life’’ set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions. Annotations were labeled by trained human readers with quality control management. They classify patient into eight labels includingmalities (O)._

1.Normal (N),

2.Diabetes (D),

3.Glaucoma (G),

4.Cataract (C),

5.Age related Macular Degeneration (A),

6.Hypertension (H),

7.Pathological Myopia (M),

8.Other diseases/abnormalities (O)


## 3.TASK : MULTICLASS CLASSIFICATION

### 4.AIM OF PROJECT:
_The primary objective of the ocular disease recognition project is to develop a robust AI-based system capable of accurately identifying various ocular diseases from medical images_.

## 5. PYTHON IMPLIMENTATION
* Import the necessary libraries : Pandas,Numpy...
  
## 6. LOAD THE DATA

* Perform basic checks and ensure data integrity.

## 7.DATA PREPROCESSING:

* Data preprocessing is a crucial step in the development of any machine learning model. For this project, the preprocessing steps include image resizing, normalization, and augmentation to ensure the dataset is suitable for training deep learning models

![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/e6ef31e1-a674-4b13-9c6a-751abe876f71)

## 8.TRAIN-TEST SPLITTING

* The dataset is divided into training and validation sets. Typically, 80% of the data is used for training and 20% for validation.

### 9.USE VGG19 PRE-TRAINED MODEL

## 10.COMPILE AND TRAIN VGG19 MODEL

 ### 11.CLASSIFICATION REPORT:
 print(classification_report(y_test,y_pred1))
 
              precision    recall  f1-score   support

           0       0.98      0.96      0.97       109
           1       0.96      0.98      0.97       109

    accuracy                           0.97       218
    macro avg       0.97      0.97      0.97       218
    weighted avg       0.97      0.97      0.97       218

## PLOTTING
![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/d7235e33-9b5a-4039-b06d-b19433089207)

## PREDICTION
![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/372d04e9-bdce-4e90-9d50-d0b50e14a729)

### USE RESNET50 PRE-TRAINED MODEL

#### MODEL EVALUATION
**CLASSIFICATION REPORT**

        precision    recall   f1-score   support

           0          0.97      0.97       0.97       109
           1          0.97      0.97       0.97       109

    accuracy                               0.97       218
    macro avg          0.97     0.97       0.97       218
    weighted avg       0.97     0.97       0.97       218
    
**CONFUSION MATRIX**

   ![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/d460b965-c055-4d35-9984-f0ee15b11425)
    
  ## PLOTTING
  
  ![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/8ab2d737-887f-4a35-8acb-e89ad914f7e9)

### PREDICTION

   ![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/b7db7eee-588c-4bd6-8847-35a2cbbc5306)

## VISION TRANSFORMERS

![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/b2ae4cc1-0599-4710-aed3-14588d1784ea)

![image](https://github.com/Tanwar-12/OCULAR-DISEASE-RECOGNITION/assets/110081008/a426d962-bed1-4e82-8a45-cba474e0444c)

### MODEL COMPARING:

_**The performance of two deep learning models, VGG19 and ResNet50, was evaluated for ocular disease recognition. Both models achieved perfect training accuracy, indicating they fit the training data extremely well.**_

#### VGG19 Model:

* Training Accuracy: The VGG19 model achieved a training accuracy of 1.0, demonstrating its ability to perfectly classify all training samples.
  
* Training Loss: The training loss for the VGG19 model was 0.000085, indicating minimal error during training.
  
* Validation Accuracy: On the validation set, VGG19 achieved an accuracy of 0.972477, showcasing its strong performance on unseen data, though slightly lower than its training accuracy.
 
* Validation Loss: The validation loss for VGG19 was 0.321402, suggesting some discrepancy between the model's performance on the training data versus the validation data.
  
#### ResNet50 Model:

* Training Accuracy: Similar to VGG19, the ResNet50 model also achieved a perfect training accuracy of 1.0.
  
* Training Loss: The training loss for ResNet50 was even lower than VGG19, at 0.000026, reflecting a very tight fit to the training data
  
* Validation Accuracy: ResNet50 outperformed VGG19 slightly on the validation set, with a validation accuracy of 0.977064.
  
* Validation Loss: The validation loss for ResNet50 was 0.214302, which is lower than that of VGG19, indicating better generalization to the validation data.
  
In summary, both models performed exceptionally well, with ResNet50 slightly outperforming VGG19 in terms of validation accuracy and validation loss, suggesting that ResNet50 may generalize better to new, unseen data.








