from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from joblib import load
import pandas as pd
import os

# #############################################################################
# Transformations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer


app = Flask(__name__)
api = Api(app)

# Load the model and metadata
MODEL_DIR = os.environ.get("MODEL_DIR", "/home/jovyan/model")
MODEL_FILE = os.environ.get("MODEL_FILE", "mlp_final.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

print(f"Loading model from: {MODEL_PATH}")
model = load(MODEL_PATH)

# Define preprocessor

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ## Apply your transformation
        return X.astype(str)

    def get_feature_names_out(self, input_features=None):
        ## Implement this method to return the feature names
        return input_features
    
# Load fitted preprocessor
preprocessor = load('preprocessor.pkl')


# Add a simple route to handle requests to the root URL
@app.route('/')
def home():
    return "Welcome to the Flask API!"
         

# Define prediction class
class Prediction(Resource):


    def __init__(self):
         
         
         self.reqparse = reqparse.RequestParser()
         self.reqparse.add_argument('USMER', type=int, required=True, help='No USMER provided', location='json')
         self.reqparse.add_argument('SEX', type=str, required=True, help='No SEX provided', location='json')
         self.reqparse.add_argument('PATIENT_TYPE', type=str, required=True, help='No PATIENT_TYPE provided', location='json')
         self.reqparse.add_argument('INTUBED', type=float, required=True, help='No INTUBED provided', location='json')
         self.reqparse.add_argument('PNEUMONIA', type=float, required=True, help='No PNEUMONIA provided', location='json')
         self.reqparse.add_argument('AGE', type=float, required=True, help='No AGE provided', location='json')
         self.reqparse.add_argument('PREGNANT', type=float, required=True, help='No PREGNANT provided', location='json')
         self.reqparse.add_argument('DIABETES', type=float, required=True, help='No DIABETES provided', location='json')
         self.reqparse.add_argument('COPD', type=float, required=True, help='No COPD provided', location='json')
         self.reqparse.add_argument('ASTHMA', type=float, required=True, help='No ASTHMA provided', location='json')
         self.reqparse.add_argument('INMSUPR', type=float, required=True, help='No INMSUPR provided', location='json')
         self.reqparse.add_argument('HIPERTENSION', type=float, required=True, help='No HIPERTENSION provided', location='json')
         self.reqparse.add_argument('OTHER_DISEASE', type=float, required=True, help='No OTHER_DISEASE provided', location='json')
         self.reqparse.add_argument('CARDIOVASCULAR', type=float, required=True, help='No CARDIOVASCULAR provided', location='json')
         self.reqparse.add_argument('OBESITY', type=float, required=True, help='No OBESITY provided', location='json')
         self.reqparse.add_argument('RENAL_CHRONIC', type=float, required=True, help='No RENAL_CHRONIC provided', location='json')
         self.reqparse.add_argument('TOBACCO', type=float, required=True, help='No TOBACCO provided', location='json')
         self.reqparse.add_argument('CLASIFFICATION_FINAL', type=int, required=True, help='No CLASIFFICATION_FINAL provided', location='json')
         self.reqparse.add_argument('ICU', type=float, required=True, help='No ICU provided', location='json')
         super(Prediction, self).__init__()


    def post(self):
        args = self.reqparse.parse_args()
        data = pd.DataFrame([args])
        data = preprocessor.transform(data)
        X = pd.DataFrame(data, columns=preprocessor.get_feature_names_out())
        prediction = model.predict(X)
        prediction_prob = model.predict_proba(X)
        return jsonify({
                'prediction': prediction.tolist(),
                'probability': prediction_prob.tolist()
            })
    

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug= True)

