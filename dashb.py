import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math

import seaborn as sns

from lightgbm import LGBMClassifier
import pickle
import plotly.express as px
import shap 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py


#path='/content/drive/MyDrive/projet7/'
#path="C:\Users\hojei\venv\p7\model\"


#path="C:/Users/hojei/venv/p7/model/"
@st.cache #mise en cache de la fonction pour exécution unique

def chargement():

    path_df = "https://raw.githubusercontent.com/marwa-abboud/model/master/dataframe.csv"
    path_predict= "https://raw.githubusercontent.com/marwa-abboud/model/master/prediction.csv"
    #path_features = path + "shap_features.pickle"
    dataframe = pd.read_csv(path_df,error_bad_lines=False)
    #dataframe.drop(columns='TARGET',inplace=True)
    predi = pd.read_csv(path_predict,error_bad_lines=False)
	
    #with open(path_features, 'rb') as file:
        #features=pickle.load(file)
    #list_id = dataframe['SK_ID_CURR'].tolist()
    customer = dataframe['SK_ID_CURR'] 
    customer=customer.astype('int64')
	
    #return dataframe, customer,predi #,features
    return dataframe, customer,predi
	

#dataframe,customer, predi, features = chargement(path)
dataframe,customer, predi = chargement()


		
		
st.title("Implémentez un modèle de scoring")




id_client = st.text_input('Veuillez saisir identifiant du client:', 'x')

if (id_client == 'x'):
	st.error('Merci de rentrer un ID client valide')
	
elif(int(id_client) not in customer.values):
    st.write('Client non reconnu : Veuillez résaisir un idenfifiant')
	
else:
    with st.spinner('Chargement...'):
        class_cust = int(predi[predi['SK_ID_CURR']==int(id_client)]['predict'].values[0])
        proba = predi[predi['SK_ID_CURR']==int(id_client)]['proba'].values[0]
        classe_vrai = int(predi[predi['SK_ID_CURR']==int(id_client)]['TARGET'].values[0])
        st.subheader("Prédictions de la capacité d'un client à rembourser son prêt")

        if class_cust == 1: 
            reponse= 'Le risque de défaut de paiement de ce client est élevé'
            st.markdown('<style>p{color: orange;}</style>', unsafe_allow_html=True)
        else:
            reponse=  'Le risque de défaut de paiement de ce client est faible'

        classe_vrai = str(classe_vrai).replace('0', 'sans défaut de paiement').replace('1', 'avec défaut de paiement')
 
        chaine = 'Prédiction : **' + reponse +  '** avec ' + str(round(proba*100)) + '% \n(classe réelle : '+str(classe_vrai) + ')'
		#

        st.markdown(chaine)

	
        list_infos = ['SK_ID_CURR','NAME_CONTRACT_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','DAYS_EMPLOYED','DAYS_BIRTH','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
        df_clients = dataframe[dataframe['SK_ID_CURR'] == int(id_client)]
        df_clients = df_clients.loc[:,list_infos] 
        #df_clients.columns = ['SK_ID_CURR','AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED','NAME_CONTRACT_TYPE']
        df_clients.index = ['Information']       
        #df_clients['age']=int(np.abs(df_clients['DAYS_BIRTH'].tolist()[0])/365)
        df_clients["Age "]= int(df_clients["DAYS_BIRTH"]/-365) #.values
        df_clients['DAYS_EMPLOYED'] = -df_clients['DAYS_EMPLOYED']	
        st.success('Informations client : ')
        st.write(df_clients)
        mask_1 = (dataframe['TARGET'] == 1)
        mask_0 = (dataframe['TARGET'] == 0)
        data= dataframe.dropna()

        data_1 = [data.loc[mask_0,'EXT_SOURCE_1'],data.loc[mask_1,'EXT_SOURCE_1']]
        data_2 = [data.loc[mask_0,'EXT_SOURCE_2'],data.loc[mask_1,'EXT_SOURCE_2']]
        data_3 = [data.loc[mask_0,'EXT_SOURCE_3'],data.loc[mask_1,'EXT_SOURCE_3']]
        data_4 = [data.loc[mask_0,'AMT_CREDIT'],data.loc[mask_1,'AMT_CREDIT']]
        group_labels = ['Défaillant', 'Non Défaillant']
        colors = ['#EB89B5', '#37AA9C']
        st.subheader("Distributions des features les plus importantes pour la prediction")

	
        fig1 = ff.create_distplot(data_1, group_labels, show_hist=False, colors=colors,show_rug=False)
        fig1.update_layout(title={'text': "Source Extérieure 1",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 1",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
        fig1.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_1'])[0], np.array(df_clients['EXT_SOURCE_1'])[0]], 
						  y=[0,5], mode="lines", name='Client', line=go.scatter.Line(color="red")))
        st.plotly_chart(fig1)
        fig2 = ff.create_distplot(data_2, group_labels, show_hist=False, colors=colors,show_rug=False)
        fig2.update_layout(title={'text': "Source Extérieure 2",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 2",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
        fig2.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_2'])[0], np.array(df_clients['EXT_SOURCE_2'])[0]], 
						  y=[0,3], mode="lines", name='Client', line=go.scatter.Line(color="red")))
        st.plotly_chart(fig2)
        fig3 = ff.create_distplot(data_3, group_labels, show_hist=False, colors=colors,show_rug=False)
        fig3.update_layout(title={'text': "Source Extérieure 3",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 3",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
        fig3.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_3'])[0], np.array(df_clients['EXT_SOURCE_3'])[0]], 
						  y=[0,3], mode="lines", name='Client', line=go.scatter.Line(color="red")))
        st.plotly_chart(fig3)
        fig4 = ff.create_distplot(data_4, group_labels, show_hist=False, colors=colors,show_rug=False)
        fig4.update_layout(title={'text': "Montant total du crédit",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="AMT CREDIT ",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
    #fig4.add_trace(go.Scatter(x=[np.array(df_clients['AMT_CREDIT'])[0], np.array(df_clients['AMT_CREDIT'])[0]], 
						 #y=[4,40], mode="lines", name='Client', line=go.scatter.Line(color="red")))
        st.plotly_chart(fig4)
		
			
#Déterminer les individus les plus proches du client dont l'id est séléctionné
check_voisins = st.checkbox("Afficher dossiers similaires?")

if check_voisins:
    #AMT_CREDIT : montant total du crédit
    #index0 = dataframe[dataframe["SK_ID_CURR"] == int(id_client)].index.values
    dataframe['age']= - (dataframe["DAYS_BIRTH"]/365)
    job = dataframe[dataframe["SK_ID_CURR"] == int(id_client)][['AMT_CREDIT','age','CODE_GENDER']]
	# On filtre nos donnée avec les plus proches voisins de notre client.
    df_voisin= dataframe[dataframe['CODE_GENDER']== job.at[job.index[0],'CODE_GENDER']]
    df_voisin=df_voisin.replace(np.nan,0)
    df_voisin=df_voisin[(df_voisin['age']<(job.at[job.index[0],'age']+5))&(df_voisin['age']>(job.at[job.index[0],'age']-5))]
    df_voisin=df_voisin[((df_voisin['AMT_CREDIT']/(job.at[job.index[0],'AMT_CREDIT']))<=1.2) & ((df_voisin['AMT_CREDIT']/(job.at[job.index[0],'AMT_CREDIT']))>=0.8)]
    #similar_id= df_voisin['SK_ID_CURR']
    #st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
    #st.write(similar_id)
    mask_target1 = (df_voisin['TARGET'] == 1)
    mask_target0 = (df_voisin['TARGET'] == 0)
   

    data_source1 = [df_voisin.loc[mask_target0,'EXT_SOURCE_1'],df_voisin.loc[mask_target1,'EXT_SOURCE_1']]
    data_source2 = [df_voisin.loc[mask_target0,'EXT_SOURCE_2'],df_voisin.loc[mask_target1,'EXT_SOURCE_2']]
    data_source3 = [df_voisin.loc[mask_target0,'EXT_SOURCE_3'],df_voisin.loc[mask_target1,'EXT_SOURCE_3']]
    data_source4 = [df_voisin.loc[mask_target0,'AMT_CREDIT'],df_voisin.loc[mask_target1,'AMT_CREDIT']]
    group_labels = ['Défaillant', 'Non Défaillant']
    colors = ['#EB89B5', '#37AA9C']


	
    fig1 = ff.create_distplot(data_source1, group_labels, show_hist=False, colors=colors,show_rug=False)
    fig1.update_layout(title={'text': "Source Extérieure 1",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 1",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
    fig1.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_1'])[0], np.array(df_clients['EXT_SOURCE_1'])[0]], 
						  y=[0,5], mode="lines", name='Client', line=go.scatter.Line(color="red")))
    st.plotly_chart(fig1)
    fig2 = ff.create_distplot(data_source2, group_labels, show_hist=False, colors=colors,show_rug=False)
    fig2.update_layout(title={'text': "Source Extérieure 2",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 2",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
    fig2.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_2'])[0], np.array(df_clients['EXT_SOURCE_2'])[0]], 
						  y=[0,3], mode="lines", name='Client', line=go.scatter.Line(color="red")))
    st.plotly_chart(fig2)
    fig3 = ff.create_distplot(data_source3, group_labels, show_hist=False, colors=colors,show_rug=False)
    fig3.update_layout(title={'text': "Source Extérieure 3",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="Source Ext 3",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
    fig3.add_trace(go.Scatter(x=[np.array(df_clients['EXT_SOURCE_3'])[0], np.array(df_clients['EXT_SOURCE_3'])[0]], 
						  y=[0,3], mode="lines", name='Client', line=go.scatter.Line(color="red")))
    st.plotly_chart(fig3)
    fig4 = ff.create_distplot(data_source4, group_labels, show_hist=False, colors=colors,show_rug=False)
    fig4.update_layout(title={'text': "Montant total du crédit",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, width=900,height=450,xaxis_title="AMT CREDIT ",yaxis_title=" ",font=dict(size=15, color="#7f7f7f"))
    #fig4.add_trace(go.Scatter(x=[np.array(df_clients['AMT_CREDIT'])[0], np.array(df_clients['AMT_CREDIT'])[0]], 
						 #y=[4,40], mode="lines", name='Client', line=go.scatter.Line(color="red")))
    st.plotly_chart(fig4)
	
    

    # Plot the distribution of ages in years
    #fig2=px.histogram(df_voisin['AMT_CREDIT'],nbins=30)
    #fig2.update_layout(title={'text': 'Montant total du credit','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'bottom'},transition_duration=500)
    #st.plotly_chart(fig2)
    #fig3=px.bar(df_voisin['AMT_CREDIT'])
    #fig1 = px.histogram(df_voisin.loc[mask_target0, 'AMT_CREDIT'])
    #fig1= px.histogram(df_voisin['AMT_CREDIT'])
    #fig1.update_layout(title={'text': 'days','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'bottom'},transition_duration=500)
    #st.plotly_chart(fig2)
    #fig4=sns.kdeplot(df_voisin.loc[mask_target0,'AMT_CREDIT'], label = 'target == 0')
    # plot loans that were not repaid
    #fig4=sns.kdeplot(df_voisin.loc[mask_target1, 'AMT_CREDIT'], label = 'target == 1')
    #fig4.figure 
   
     
else:
    st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)