![](UTA-DataScience-Logo.png)

# Predict Heath Outcomes of Horses

This repository holds an attempt to train a model using machine learning that applies prediction outcome algorithms to determine whether a horse lived, died, or euthanized. Kaggle challenge: https://www.kaggle.com/competitions/playground-series-s3e22/overview 

## Overview

The task, as defined by the Kaggle challenge is to use the given various medical indicators to predict the health outcomes of horses.The approach in this repository formulates the problem as a multi binary classification task, using different classification methods of machine learning like LGBMClassifier, Random Forest. The performance of each was compared between the different methods. The best model was able to predict the health outcome with a 67% accuracy.


### Data

* Data:
 * Train.csv consisted of 1235 rows and 29 columns.
 * Test.csv consisted of 824 rows and 28 columns.
    * Input: horse health, CSV file: filename -> train.csv
    * Output: lived/died/euthanized within outcome variable.
  * Instances (Train, Test, Validation Split): 1235 for training, 824 for testing, none for validation

#### Preprocessing / Clean up

* Identified any missing and/or duplicate variables.
   * Numerical Columns: Impute missing values with the median.
   * Categorical Columns: Impute missing values with the most frequent value (mode) or use a placeholder like 'Unknown'.
* One-hot encoding used to convert categorical variables to numerical, for better machine learning.
* Used Min-Max scaling:
   * Since the numerical features in the dataset have unknown distributions, Min-Max scaling provides a consistent range.
   * Also, brings all features into the same range (0 to 1), ensuring that no feature dominates the learning process due to larger scales.

#### Data Visualization

Shown is the outcomes (lived, died, and euthanized) for each variable in the data set.
![](im1.png) 
![](im2.png) 
![](im3.png) 
![](im4.png) 
![](im5.png) 
![](im6.png) 
![](im7.png) 

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* To fully reproduce my results:
   * Download the datasets from Kaggle (and any other info needed).
   * Ensure the necessary libraries are installed (can use different ones if applicable).
   * Download the Kaggle_Project notebook attached in the directory and run it.
Useful Resources:
Sklearn website; the sklearn website is a great place to find detailed explainations of the models and tools used in this project.

### Overview of files in repository

* Directory Structure: the directory contains a README.md, the code for my Kaggle project, and image files for my visuals.
Relevent Files:
* KaggleProject.ipynb: this notebook contains all of the code for the final submission of my Kaggle project.

### Software Setup
* Required Packages:
   * Numpy
   * Pandas
   * Sklearn
* Installation Process: (All packages were installed via the Linux subsystem for Windows.)
   * pip install numpy
   * pip install pandas
   * pip install scikit-learn

### Data

* Data can be downloaded from: https://www.kaggle.com/competitions/playground-series-s3e22/data 
   * Select the train and test data for this project

### Training

* Modles were trained using scikit-learn.
* By running the KaggleProject file, you can see how the data set was trained.

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







