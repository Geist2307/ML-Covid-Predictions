# Detecting High Risk COVID-19 patients

## Motivation

<p>

In this short project, I built a classifier that detects whether a patient diagnosed with COVID-19 is at high risk of death or not, based on the incidence of other diseases and lifestyle factors. The initial dataset was taken from [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/), and subsequently cleaned and pre-processed into a dataset that contains more than 300.000 observations of patients diagnosed with COVID-19. The classes are highly imbalanced (16: 100), with the minority class (patients at risk of death) being only 16% present. 

</p>

Since the classes are imbalanced, it is very important not only that the classifier detect high risk patients with high accuracy, but also that the detected positives are true positives (actual high risk patients). This is because the cost of misclassifying a high risk patients is potential loss of life, while the cost of misclassifying a lower risk patient is less. This means that recall is more important than accuracy.

## Files

This repo contains several files. These are as follow:

<ul>

<li> A Jupyter Notebook that contains the end-to-end workflow : cleaning the data, building a pre-processing pipeline, testing and selecting the best model, fine-tuning the best model with undersampling methods and finally saving the final model. </li>

<li> The kaggle_covid_data contains the most recent version of the data downloaded through the Kaggle API. </li>

<li> preprocessing.pkl is the saved pre-processing pipeline for the data. </li>

<li> covid_data.html is a html report showing an automated exploratory analysis. </li>

</ul>

## Workflow

The Jupyter Notebook contains the following main sections :

This end-to-end Machine Learning Project employs the following workflow. To quickly navigate to the desired section, please click on the title of the section



### Frame the problem 

- This section clarifies what is the objective of the analysis, that is, detecting high risk COVID-19 positive patients based on comorbidity and life style factors. As classes are imbalanced, it is crucial that the final model achieves very high recall, at the potential expense of accuracy.
 

### Clean the Data 

- This section deals with imputation of missing values and removal of clearly erroneous results. 




### Split the Data 

- We split the data into a training set and a test set, and we perform all subsequent exploration on a copy of the training set.

### EDA 

- Look for correlations and experiment with columns transformations/


### Pre-process Data for Machine Learning Algorithms 

- Create pre-processing pipeline to transform the data, define one evaluation function that can be used to evaluate all ML models, and create final datasets for training and testing.

### Select and Train models 

- Select many classifiers to test against the evaluation metric defined previously, and use hyper-parameter tuning with BayesCV for each model class. 




### Voting Classifier 

- Create a voting classifier based on three best models and see if better performance can be obtained.


### Undersampling

Test several methods of undersampling the training set to address the imbalance of classes and improve recall on the test set. 

### Oversampling 

- Make use of SMOTE methods to augment the minority class and evaluate the performance on the test set. 


### Generalisation Error

- Make predictions with the best model and the best sampling technique and calculate several performance metrics on the test set


### Launch!

- Save model and pre-processing pipeline for further use




## Use and External Validation

The code in the Jupyter notebook can be used to (i) transform and (ii) make predictions on any dataset that contains the same columns as the one inclued in kaggle_covid_data. In general, any dataset that contains information on patients diagnosed with COVID-19, other factors and whetehr they survived or not can be analyzed with the methods proposed in the Notebook.

## Results

By undersampling the majority class on the training set and tuning the hyperparameters of an MLP classifer, I was able to increase the recall at over 93%, which means that about 7% High Risk patients are still misclassified. The best result for TPR was with the Borderline SMOTE method, achieving over 95% recall. It should be noted that higher recall comes at a cost of more false positives, however, as we are interested in misclassifying as few high risk patients as possible, the increase in False Positives is not a major problem.

