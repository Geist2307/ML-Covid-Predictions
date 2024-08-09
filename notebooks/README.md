# Detecting High Risk COVID-19 patients

## Motivation

<p>

In this short project, I built a classifier that detects whether a patient diagnosed with COVID-19 is at high risk of death or not, based on the incidence of other diseases and lifestyle factors. The initial dataset was taken from [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/), and subsequently cleaned and pre-processed into a dataset that contains more than 300.000 observations of patients diagnosed with COVID-19. The classes are highly imbalanced. 86% are Low Risk, and approximately 14% are High Risk. 
</p>

Since the classes are imbalanced, it is very important the detected positives are true positives (actual high risk patients). This is because the cost of misclassifying a high risk patients is potential loss of life, while the cost of misclassifying a lower risk patient would simply result in an examination. The aim of this system is to find the minority class, to flag them correctly. The metric that I choose to evaluate my model is recall. 

## Notebooks

There are two notebooks. "Experiments" show the full end-to-end workflow for cleaning and pre-processing the data, and finding the the best model-class - it contains all observations and observations of the process.

<ul>

<li> "Experiments" show the full end-to-end workflow for cleaning and pre-processing the data, and finding the the best model-class - it contains all observations and observations of the process. </li>

<li> "Pipeline" simply implements the selected transformations and uses the trained and fine-tuned model found to be the be performing (Recall) in Experiments </li>



</ul>



## Use and External Validation

The model has been trained on real data. In general, any dataset that contains information on patients diagnosed with COVID-19, other factors and whetehr they survived or not can be analyzed with the methods proposed in the Notebook.

## Results

By undersampling the majority class on the training set and tuning the hyperparameters of an MLP classifer, I was able to increase the recall at over 96%, which means that about 4% High Risk patients are still misclassified. The best result for TPR was with the Borderline SMOTE method, achieving over 96% recall. It should be noted that higher recall comes at a cost of more false positives, however, as we are interested in misclassifying as few high risk patients as possible, the increase in False Positives is not a major problem.

