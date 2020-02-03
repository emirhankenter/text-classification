from django.apps import AppConfig
from sklearn.externals import joblib
import pickle

class MyapiConfig(AppConfig):
    name = 'MyAPI'
    myModel = joblib.load("/Users/emirh/Desktop/bitirme/finalized_model.pkl")
    with open('/Users/emirh/Desktop/bitirme/tokenizer.pickle', 'rb') as handle: tokenizer = pickle.load(handle)