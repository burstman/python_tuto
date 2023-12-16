import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

#load the file model that was created.
loaded_rf_model = load('streamlit_test/random_forest_model.joblib')

st.title('Streamlit CheckPoint 1')
#these are the encoded values for 2 categoricals features "REGION" and "TENURE"
unique_values = {
    'REGION': {
        'FATICK': 2.0, 'DAKAR': 0.0, 'THIES': 12.0, 'KAOLACK': 4.0, 'TAMBACOUNDA': 11.0,
        'ZIGUINCHOR': 13.0, 'SAINT-LOUIS': 9.0, 'DIOURBEL': 1.0, 'MATAM': 8.0, 'KAFFRINE': 3.0,
        'LOUGA': 7.0, 'KOLDA': 6.0, 'SEDHIOU': 10.0, 'KEDOUGOU': 5.0
    },
    'TENURE': {
        'K > 24 month': 7.0, 'I 18-21 month': 5.0, 'H 15-18 month': 4.0, 'G 12-15 month': 3.0,
        'J 21-24 month': 6.0, 'F 9-12 month': 2.0, 'E 6-9 month': 1.0, 'D 3-6 month': 0.0
    }
}

st.text('Feature')
# Display select boxes for categorical columns
selected_region = st.selectbox("Choose REGION", options=list(unique_values['REGION'].keys()))
selected_region_numeric = unique_values['REGION'][selected_region] # type: ignore

selected_tenure = st.selectbox("Choose TENURE", options=list(unique_values['TENURE'].keys()))
selected_tenure_numeric = unique_values['TENURE'][selected_tenure] # type: ignore

apu = st.number_input("ARPU_SEGMENT")

# Display selected numeric values
st.write(f"Selected REGION numeric value: {selected_region_numeric}")
st.write(f"Selected TENURE numeric value: {selected_tenure_numeric}")
st.write(f"ARPU_SEGMENT: {apu}")

if st.button('Predict'):
    # Create named features
    features = ['ARPU_SEGMENT', 'REGION', 'TENURE']
    # Create input data with named features
    input_data = np.array([[apu, selected_region_numeric, selected_tenure_numeric]])
    input_data = pd.DataFrame(input_data, columns=features)  

    predicted_value = loaded_rf_model.predict(input_data)
    if predicted_value[0] == 1:
        st.write('CHURN CLIENT')
    else:
        st.write('NOT CHURN CLIENT')
