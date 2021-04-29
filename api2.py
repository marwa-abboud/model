import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier

app = Flask(__name__)

path="C:/Users/hojei/venv/p7/PT7/"

path_df = path +"dataframe.csv"
val_set = pd.read_csv(path_df)

path_model=path +"lgbm_shap.pickle"
with open(path_model, 'rb') as file:
    lgbm_shap=pickle.load(file)
		
#model = pickle.load( open( "model.pickle", "rb" ) )
#val_set = pickle.load( open( "val_set.pickle", "rb" ) )

#path='/content/drive/MyDrive/Projet7/


@app.route('/')
def home():
    return 'Entrer une ID client dans la barre URL'

@app.route('/<int:ID>/')
def requet_ID(ID):
   
    if ID not in list(val_set['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de donnée'
    else:
        val_set_ID=val_set[val_set['SK_ID_CURR']==int(ID)]

        y_proba=lgbm_shap.predict_proba(val_set_ID.drop(['SK_ID_CURR','TARGET'],axis=1))[:, 1]#model
   
        if y_proba >= 0.5 :
           result=('ce client est solvable avec un taux de risque de '+ str(np.around(y_proba*100,2))+'%')

        else :
            result=('ce client est non solvable avec un taux de risque de '+ str(np.around(y_proba*100,2))+'%')
 
    return result


if __name__ == '__main__':
    app.run()