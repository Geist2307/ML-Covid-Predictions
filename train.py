import json
import os

# #############################################################################
# Data processing
import joblib
from joblib import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #############################################################################
# SMOTE
from imblearn.over_sampling import BorderlineSMOTE

# #############################################################################
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import skopt
from skopt import BayesSearchCV

# #############################################################################
# Model Class
from sklearn.neural_network import MLPClassifier


# #############################################################################
# Transformations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer


# #############################################################################
# Read the data into dataframe
csv_path = '/home/jovyan/data/Covid Data.csv'

# Check if the file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The file {csv_path} does not exist.")

try:
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to read the CSV file: {e}")

# #############################################################################
# Data cleaning

def get_clean_data(df):

    """ Function that implements the important cleaning operations on the dataset
    df: the dataset to be cleaned"""

    # List of conditions
    conditions = ['USMER', 'INTUBED', 'PNEUMONIA',  'PREGNANT', 
              'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
              'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
              'RENAL_CHRONIC', 'TOBACCO', 'ICU']
    
    #replace "2" with 0 in categorical columns that show the presence of a symptom
    for condition in conditions:
        df[condition] = df[condition].replace(2, 0)

    # Deal with 'SEX' column
    df["SEX"] = df['SEX'].astype('string')
    df['SEX'] = df['SEX'].replace({"1": 'Female', "2": 'Male'})

    # Deal with 'PATIENT_TYPE' column
    df["PATIENT_TYPE"] = df['PATIENT_TYPE'].astype('string')
    df['PATIENT_TYPE'] = df['PATIENT_TYPE'].replace({"1": 'Home', "2": 'Hospital'})

    # Create response variables
    df["Lower_Risk"] = df["DATE_DIED"].apply(lambda x: 1 if x == "9999-99-99" else 0)
    df["Higher_Risk"] = df["DATE_DIED"].apply(lambda x: 1 if x != "9999-99-99" else 0)

    # Replace 97, 98, 99 with NaN
    df = df.replace({97: np.nan, 98: np.nan, 99: np.nan})

    # Impute median for age
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())

    # missing feats
    mis_features = [ 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA',
       'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO', 'ICU', 'PREGNANT', 'INTUBED']
    
    # Impute 0.5 for missing values
    for col in mis_features:
        df[col] = df[col].fillna(0.5)

    # Select positive cases
    df['COVID_POSITIVE'] = df['CLASIFFICATION_FINAL'].apply(lambda x: 1 if x<=3 else 0)
    df['COVID_INCONCLUSIVE'] = df['CLASIFFICATION_FINAL'].apply(lambda x: 1 if x>=4 else 0)

    # create new dataframe only with positive
    df_positive = df[df['COVID_POSITIVE'] == 1]

    # Drop columns
    df_positive = df_positive.drop(['COVID_POSITIVE', 'COVID_INCONCLUSIVE', 'DATE_DIED', 'Lower_Risk'], axis=1)


    return df_positive

df_clean = get_clean_data(df)

# #############################################################################
# split the data
train, test = train_test_split(df_clean, test_size=0.2, stratify =df_clean['Higher_Risk'], random_state=42)
XTrain = train.drop('Higher_Risk', axis=1)
XTest = test.drop('Higher_Risk', axis=1)

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ## Apply your transformation
        return X.astype(str)

    def get_feature_names_out(self, input_features=None):
        ## Implement this method to return the feature names
        return input_features
    
##this creates a pipeline that transforms to string then does one-hot encoding
cat_pipeline_default = make_pipeline(
                CustomTransformer(),
                OneHotEncoder(handle_unknown='ignore')
                ) 

##this creates a pipeline that only does one-hot encoding
cat_pipeline = make_pipeline(
                OneHotEncoder(handle_unknown='ignore')
                )

##Now I am creating two pipelines for discretizing, and one-hot-encoding.

bucket_pipeline_1 = make_pipeline(
    KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans', subsample = 200_000, random_state=41),
     OneHotEncoder(handle_unknown='ignore')) 


## now for standard scaling
num_pipeline = make_pipeline(
    StandardScaler()
)


preprocessor = ColumnTransformer([
        
        ('categorical', cat_pipeline_default, ['USMER', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD',
        'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
        'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']),
        ('numerical', num_pipeline, ['AGE']),
        ('categorical simple', cat_pipeline, ['SEX', 'PATIENT_TYPE']),
         ('drop', 'drop', ['CLASIFFICATION_FINAL', 'MEDICAL_UNIT']), # we do not need medical_unit
    ], remainder='passthrough')

# transform the data
preprocessor.fit(XTrain)

# save the preprocessor locally
joblib.dump(preprocessor, 'preprocessor.pkl')



# Transform the training data
X_transformed = preprocessor.transform(XTrain)
y = train["Higher_Risk"]

# Create DataFrame with transformed data
X = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out(), index=XTrain.index)


# Transform the test data
XX_transformed = preprocessor.transform(XTest)
yy = test["Higher_Risk"]

# Create DataFrame with transformed data
XX = pd.DataFrame(XX_transformed, columns=preprocessor.get_feature_names_out(), index=XTest.index)


# Balance the data
print("Balancing the data using BorderlineSMOTE")
bsmote = BorderlineSMOTE(random_state=42)
X_bsmote, y_bsmote = bsmote.fit_resample(X, y)

# #############################################################################
# Model instance
mlp_final = MLPClassifier(activation= 'relu', alpha= 0.038, learning_rate= 'constant', solver= 'adam', random_state=1918)

# #############################################################################
# Fit the model
print("Fitting the model")
mlp_final.fit(X_bsmote, y_bsmote)

# #############################################################################
# Recall
print("Calculating recall scores")
train_recall = recall_score(y, mlp_final.predict(X))
test_recall = recall_score(yy, mlp_final.predict(XX))

metadata = { "train_recall": train_recall, "test_recall": test_recall }

# #############################################################################
# Serialise the model and metadata
model_path = "/home/jovyan/model/mlp_final.joblib"
metadata_path = "/home/jovyan/model/metadata.json"

print(f"Serialising model to: {model_path}")
dump(mlp_final, model_path)

print(f"Serialising metadata to: {metadata_path}")
with open(metadata_path, "w") as f:
    json.dump(metadata, f)

print("Model and metadata serialization complete.")


