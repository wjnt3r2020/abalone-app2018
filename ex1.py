import streamlit as st
import pandas as pd
import pickle as pk

st.write("""
Hello
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

st.header('Application of Abalone\'s Age Prediction:')
st.subheader('User Input:')


def get_input():
    #widgets
        v_Sex = st.sidebar.radio('Sex', ["Male","Female","Infant"])
        v_Length = st.sidebar.slider('Length', 0.075000, 0.745000, 0.506790) #1
        v_Diameter = st.sidebar.slider('Diameter', 0.055000, 0.600000, 0.400600) #2
        v_Height = st.sidebar.slider('Height', 0.010000	, 0.240000 ,0.138800) #3
        v_Whole_weight = st.sidebar.slider('Whole_weight', 0.002000	, 2.550000 , 0.785165) #4
        v_Shucked_weight = st.sidebar.slider('Shucked_weight', 0.001000	, 1.070500 ,0.308956) #5
        v_Viscera_weight = st.sidebar.slider('Viscera_weight', 0.000500	, 0.541000 ,0.170249)#6
        v_Shell_weight = st.sidebar.slider('Shell_weight', 0.001500	, 1.005000 ,0.249127) #7

        if v_Sex == 'Male': v_Sex = 'M'
        elif v_Sex == 'Female': v_Sex = 'F'
        else: v_Sex = 'I'

        

    #dictionary
        data = {'Sex': v_Sex,'Length': v_Length,'Diameter': v_Diameter,'Height': v_Height,'Whole_weight': v_Whole_weight,'Shucked_weight': v_Shucked_weight,'Viscera_weight': v_Viscera_weight,'Shell_weight':v_Shell_weight}
        
        


    #create data frame
        data_df = pd.DataFrame(data, index=[0])
        return data_df

df = get_input()
st.write(df)


data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
#st.write(df)

cat_data = pd.get_dummies(df[['Sex']])
#st.write(cat_data)


X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1]
X_new = X_new.drop(columns=['Sex'])
st.write(X_new)

load_nor = pk.load(open('normalization.pkl', 'rb'))
X_new = load_nor.transform(X_new)
load_knn = pk.load(open('best_knn.pkl', 'rb'))


prediction = load_knn.predict(X_new)
st.write(prediction)
