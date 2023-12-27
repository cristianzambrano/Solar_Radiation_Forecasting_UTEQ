from operator import index
import os
import io
import os.path
import pickle
import requests
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import altair as alt
  

def  saveTempToday():
    mes = date.today().strftime("%m-%Y")
    dia = date.today().strftime("%d")
    #fichero_data = "/data/%s/%s.csv" %mes %dia
    #if not os.path.isfile(fichero_data):

@st.cache_data
def  getListaDiasBD():
    df = pd.read_csv('data/datos.csv', sep=",")
    df2 = df.groupby(['Fecha','Hora'], as_index=False)['Radiacion'].count()
    df2 = df2.groupby(['Fecha'], as_index=False).count()
    df2=df2.loc[(df2['Radiacion'] == 24)]

    return df2['Fecha'].drop_duplicates().sort_values(ascending=False)

def  getPromedioTemperaturaDia(dia_predecir):
    df = pd.read_csv('data/datos.csv', sep=",")
    df2=df.loc[(df['Fecha'] == dia_predecir)]

    df2.rename(columns={'Radiacion':'RadiacionO'}, inplace=True)
    df2['TemperaturaC']  = df2['Temperatura'] - 273.15
    df2 = df2.groupby('Hora', as_index=False)[['Temperatura','RadiacionO', 'TemperaturaC']].mean()
    return df2

@st.cache_data
def getDataTempCurrentFromWheatherAPI():
    parameters = {'key': st.secrets["meteosource_key"],
              'place_id': 'quevedo',
              'timezone':'America/Guayaquil',
              'units':'metric',
              'sections':'current'}
    datos  = requests.get("https://www.meteosource.com/api/v1/free/point", parameters).json()
    return  datos['current']

@st.cache_data
def getDataFromWheatherAPI():
    parameters = {'key': st.secrets["meteosource_key"],
             'place_id': 'quevedo',
              'timezone':'America/Guayaquil',
              'units':'metric',
              'sections':'current,hourly'}
    datos  = requests.get("https://www.meteosource.com/api/v1/free/point", parameters).json()
    datos_tiempo = pd.DataFrame(columns=['TemperaturaC', 'Temperatura', 'Hora', 'Fecha', 'Icon'])
    for datos_hora in datos['hourly']['data']:
        fecha=datetime.strptime(datos_hora['date'],'%Y-%m-%dT%H:%M:%S')
        datos_tiempo.loc[len(datos_tiempo)] = [datos_hora['temperature'],datos_hora['temperature'] + 273.15,
                             fecha.hour, fecha.strftime('%d %b - %H:%M'), './app/static/icons/%s.png' %datos_hora['icon']]
    return datos_tiempo

def getModelName(ModelName):
    if ModelName == None:
        abrevModel ='GBR'
    else:
        if ModelName =='Polynomial Regression':
            abrevModel ='RP'
        elif ModelName =='K-Nearest Neighbors':
            abrevModel ='KNN'
        elif ModelName =='Random Forest Regression':
            abrevModel ='RFG'
        else:
            abrevModel ='GBR'

    return 'models/' + abrevModel + '.model'




def predict(tipo,  dia_predecir, fuentedatos, nombreModel, temperatura, hora):
    model = pickle.load(open(nombreModel, 'rb'))

    if tipo=='Predict':
      if fuentedatos=='API Wheather':
        datos_tiempo = getDataFromWheatherAPI()
      else:
        datos_tiempo = getDataFromWheatherAPI()
    elif tipo=='Test':
      datos_tiempo = getPromedioTemperaturaDia(dia_predecir)
    else:
      datos_tiempo =  pd.DataFrame([[temperatura,hora]], columns=['Temperatura','Hora'])
      

    X_test = datos_tiempo[['Temperatura','Hora']]
    if(nombreModel=='models/RP.model'):
          X_test = PolynomialFeatures(degree=2).fit_transform(X_test)
    y_pred = model.predict(X_test)
    datos_tiempo['Radiacion'] = y_pred
    return datos_tiempo

def generate_chart(tipo, dia_predecir, predicciones, mostrar_graf):
    if tipo=='Predict':
        fig = alt.Chart(predicciones, 
                        title=alt.TitleParams(
                                    text='Solar Radiation Forecasting the next 24 hours',
                                    anchor='middle')).mark_circle(size=80).encode(
                x=alt.X('Hora', type='nominal', sort=None).title('Time of the day'),
                y=alt.Y('Radiacion').title('Solar Radiation (W/m2)'),
               color=alt.Color('TemperaturaC:Q', scale=alt.Scale(scheme='reds'), legend=alt.Legend(title='Temp °C')),
               tooltip=['Fecha','Hora', 'TemperaturaC', 'Radiacion'])
        return fig
    else:
        fig=plt.figure(figsize=(10,6))
        if (mostrar_graf=='Pyranometer') or (mostrar_graf=='Both'):
            axe=sns.scatterplot(data= predicciones, x='Hora', y='RadiacionO',  label='Pyranometer')
        if (mostrar_graf=='Model') or (mostrar_graf=='Both'):
            axe=sns.scatterplot(data= predicciones, x='Hora', y='Radiacion',  label='Model')
        #axe=sns.scatterplot(data= predicciones, x='Hora', y=['Radiacion','Radiacion'], hue='TemperaturaC', size = 30)
        axe.set_title('Solar Radiation Forecasting day ' + dia_predecir)
        axe.set_ylabel('Radiation')
        axe.set_xlabel('Time of the day')
        axe.tick_params(axis='x',labelsize=8)
        return fig

modelo ="GBR"
fuente_datos_predecir='API Wheather'
st.set_page_config(page_title="Solar Radiation Forecasting in UTEQ using Machine Learning", layout="wide", page_icon="")
tipo='Predict'
dia_predecir=''
mostrar_graf = 'Both'

#Suscriptores
#df_suscripciones = pd.read_csv('data/suscritos.csv', sep=",", dtype={'Telefono': str})

col1, col2, col3= st.columns((1,5,1))
with col1:
    image = Image.open('imgs/logouteq.png')
    st.image(image, width=100)

with col2:
   texto="<table width=""100%"" style=""border:0px""><tr><td style='font-size: 36px; text-align: center;'><b>State Technical University of Quevedo</b></td></tr>" 
   texto+="<tr><td style='font-size: 32px; text-align: center;'><b>Department of Engineering Science</b></td></tr>" 
   texto+="</table>" 
   st.markdown(texto, unsafe_allow_html=True)   
        

with col3:
    image = Image.open('imgs/logofci.jpg')
    st.image(image, width=200)

col1, col2, col3 = st.columns((0.8,6.4,0.8))
with col2:
    texto="<p style='font-size: 42px; text-align: center;'><b>Solar Radiation Forecasting in the UTEQ using Machine Learning</b></p>"
    st.markdown(texto, unsafe_allow_html=True)  

if st.checkbox("Settings"):
    st.sidebar.header("Settings")
    tipo = st.sidebar.radio("Action",    ('Predict', 'Test'), index=0, horizontal=True)
    if tipo=='Predict':
        fuente_datos_predecir = st.sidebar.radio("Source ",('API Wheather', 'Mean of 7 days'), index=0, horizontal=True)
    else:
        dias = getListaDiasBD()
        dia_predecir=st.sidebar.selectbox('Choose day', options= dias, index=0)
        mostrar_graf = st.sidebar.radio("Show figures",('Pyranometer', 'Model', 'Both'), index=0, horizontal=True)
    
    modelo=st.sidebar.selectbox('Choose Model', options= ['Polynomial Regression','K-Nearest Neighbors','Random Forest Regression','Gradient Boosting Regression'], index=3)        
     

nombreModel = getModelName(modelo)
if os.path.isfile(nombreModel):
    predicciones = predict(tipo,  dia_predecir, fuente_datos_predecir, nombreModel,0,0)
    fig= generate_chart(tipo, dia_predecir, predicciones, mostrar_graf)

    datos_current_wheather = getDataTempCurrentFromWheatherAPI()
    predicciones_current = predict("Current", "", fuente_datos_predecir, nombreModel,
                                  datos_current_wheather['temperature'] + 273.15,datetime.now().hour)

    
    col1, col2, col3 = st.columns((3,3,3))
    with col2:
        texto="<p style='font-size: 36px; text-align: center;'>Forecasting now %s" %datetime.now().strftime("%H:%M") + "</p>"
        st.markdown(texto, unsafe_allow_html=True)  

    col1, col2, col3, col4, col5 = st.columns((3,1,1,1,3))
    image = Image.open('static/icons/%s.png' %datos_current_wheather['icon_num'])
    col2.image(image, caption=datos_current_wheather['summary'])
    col3.metric("Temperature (°C)", "%.2f" %datos_current_wheather['temperature'],'')
    col4.metric("Radiation (W/m2)", "%.4f" %predicciones_current['Radiacion'])

    col1, col2, col3, col4, col5= st.columns((1,2.5,0.5,3.5,1))
    with col2:
         if tipo=='Predict':
            st.markdown("<center><b>Temperature from " + fuente_datos_predecir + " and Solar Radiation Prediction</b></center>", unsafe_allow_html=True)
            st.dataframe(predicciones[['Hora','Icon','TemperaturaC','Radiacion']],
                      column_config={
                          "Hora": st.column_config.TextColumn("Hour"), 
                          "TemperaturaC": st.column_config.NumberColumn("Temperature °C",format="%.2f", width='small'), 
                          "Radiacion": st.column_config.NumberColumn("Solar Radation (W/m2)",format="%.4f", width='small'),
                          "Icon": st.column_config.ImageColumn("Wheather", help="Tiempo", width='small')
                        },
                       use_container_width=True, hide_index=True)
         else:
             st.markdown("<center><b>Pyranometer Data day " + dia_predecir +"</b></center>", unsafe_allow_html=True)
             st.dataframe(predicciones[['Hora','TemperaturaC','RadiacionO', 'Radiacion']],
                      column_config={
                          "Hora": st.column_config.TextColumn("Hour"), 
                          "TemperaturaC": st.column_config.NumberColumn("Temperatur °C",format="%.2f", width='small'), 
                          "RadiacionO": st.column_config.NumberColumn("Actual Radiation",format="%.4f", width='small'),
                          "Radiacion": st.column_config.NumberColumn("Predicted Radiation",format="%.4f", width='small')
                        },
                       use_container_width=True, hide_index=True)
    with col4:
        if tipo=='Predict':
            fig.properties( height=200)
            st.altair_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig, use_container_width=True)


      


