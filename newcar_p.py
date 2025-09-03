import streamlit as st
import pandas as pd
import joblib

st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Select a page', ['Prediction web app' , 'Code Material'])

if option == 'Prediction web app':
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.pexels.com/photos/28467944/pexels-photo-28467944.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
            
    [data-testid = "stHeader"]{
            background:rgba(0,0,0,0);
            height : 0px;
            visibility:hidden
            }
    }
    </style>
    """, unsafe_allow_html=True)

    df = pd.read_csv('Cars_24.csv')
    p = joblib.load('car_price_modell.joblib')


    st.title('Car price prediction we app')


    year = st.number_input('Enter Car Manufacturing Year' , min_value= 1990 , max_value=2025)
    km_driven = st.number_input('Enter Kilometers Driven' ,min_value= 1 , max_value= 4000000)
    mileage = st.number_input("Enter Mileage(km)" , min_value=1 , max_value=500)
    engine = st.number_input('Enter Engine Capacity' , min_value= 0 , max_value=4000)
    max_power = st.number_input('Enter Maximum Power' ,min_value= 1 , max_value= 2000)
    age = st.number_input('Enter Car Age' , min_value=1 , max_value=200)

    values =  df.groupby('make')['model'].unique().apply(list).to_dict()

    make = st.selectbox('Enter Car Brand' , list(values.keys()))
    model = st.selectbox('Enter Car Model' , values[make])

    col1 , col2 = st.columns(2)
    
    with col1:
        Individual1 = st.radio('Individual' , ['Yes','No'])
        Individual = 1 if Individual1 == "Yes" else 0
    with col2:
        TrustmarkDealer1 = st.radio('Trustmark Dealer' , ['Yes','No'])
        TrustmarkDealer = 1 if TrustmarkDealer1 == "Yes" else 0

    col3, col4 = st.columns(2)

    with col3:
        Diesel1 = st.radio('Diesel' , ['Yes','No'])
        Diesel = 1 if Diesel1 == "Yes" else 0
    with col4:
        Electric1 = st.radio('Electric' , ['Yes','No'])
        Electric = 1 if Electric1 == "Yes" else 0

    col5 , col6 = st.columns(2)

    with col5:
        LPG1 = st.radio('LPG' , ['Yes','No'])
        LPG = 1 if LPG1 == "Yes" else 0
    with col6:
        Petrol1 = st.radio('Petrol' , ['Yes','No'])
    Petrol = 1 if Petrol1 == "Yes" else 0

    col7 , col8 = st.columns(2)
    with col7:
        Manual1 = st.radio('Manual' , ['Yes','No'])
        Manual = 1 if Manual1 == "Yes" else 0
    with col8:
        morethen51 = st.radio('More than 5 year' , ['Yes','No'])
        morethen5 = 1 if morethen51 == "Yes" else 0

    user_inputs = pd.DataFrame([{
        'year':year,
        'km_driven':km_driven,
        'mileage':mileage,
        'engine':engine,
        'max_power':max_power,
        'age':age,
        'make':make,

        'model':model,
        'Individual':Individual,
        'Trustmark Dealer':TrustmarkDealer,
        'Diesel':Diesel,
        'Electric':Electric,
        'LPG':LPG,
        'Petrol':Petrol,
        'Manual':Manual,
        '>5':morethen5
    }])

    if st.button('Prediction'):
        prediction = p.predict(user_inputs)
        convert_inr = prediction[0]*100000
        final = round(convert_inr, 2)

        st.success(f'The car price prediction is : \u20B9{final}')

elif option == 'Code Material':
    st.title('Code')
    if st.button('View Code'):
        with open('newcar_.py' , 'r') as f:
            code = f.read()
            st.code(code , language='python')
    st.text('you can see the model traning code ')


    with open('newcar_.py' , 'r') as f:
            code = f.read()
    st.download_button(
        label='Download code',
        data=code,
        file_name='newcar_.py',
        mime ='text/x-python'
        )
    st.text('You can download the model traning code')

    st.title('Data')
    df= pd.read_csv('Cars_24.csv')
    if st.button('View dataset'):
        st.dataframe(df)
    st.text('you can view the cars_24 dataset')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download cars_24 dataset',
        data=csv,
        file_name='cars_24_dataset.csv',
        mime='text/csv'
    )
    st.text('You can download the dataset to use it for your own analysis or model building')








