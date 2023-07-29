
from joblib import load
import pandas as pd

#Carga del modelo
Modelo, Preprocesador = load('modelo_prepro.joblib') 

def get_X_y(tipo,barrio,sup,habs):
    X_pred=pd.DataFrame(columns=['tipo','barrio','sup','habs'])
    X_pred.loc[0,:]=[tipo,barrio,sup,habs]
    X_pred_dummies=Preprocesador.transform(X_pred)    
    return X_pred_dummies, Modelo.predict(X_pred_dummies)[0]


# creamos la lista de opciones
lista_features=list(Preprocesador.get_feature_names_out())

lista_tipos=[x.split('tipo_')[1] for x in lista_features if 'tipo' in x]
lista_barrios=[x.split('barrio_')[1] for x in lista_features if 'barrio' in x]

features=[x.split('__')[1] for x in lista_features]


# el modelo ya esta importado y listo
# generamos la web 
import streamlit as st


st.header('Elija las variables de la propiedad que quiere predecir:')

tipo = st.selectbox(
     'Tipo de inmueble?',lista_tipos)

barrio= st.selectbox(
     'Barrio del inmueble?',lista_barrios)

sup = st.slider('Superficie del inmueble?', 9, 400, 2)


habs= st.slider('Cantidad de habitaciones?', 1, 10, 1)



X_pred,y_pred=get_X_y(tipo,barrio,sup,habs)

st.write('El precio aproximado por m2 es: ', y_pred)


import shap
import streamlit.components.v1 as components

# funcion que me permite integrar un grafico de shap con streamlit
def st_shap(plot, height=None):
    js=shap.getjs()
    shap_html = f"<head>{js}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

explainer = shap.Explainer(Modelo[0])
# genero la expliacion para los datos del test



shap_value = explainer.shap_values(X_pred)
st_shap(shap.force_plot(explainer.expected_value, shap_value[0],feature_names=features))