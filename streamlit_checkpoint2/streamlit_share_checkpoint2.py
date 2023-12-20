import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# load the file model that was created.

try:
    # Attempt to load the SVC model
    loaded_svc_model = load('streamlit_checkpoint2/svc_classifier_model.joblib')
    if loaded_svc_model is not None:
        st.write("Model loaded successfully!")
    else:
        st.write("Model loaded but is None. Check the model file or loading process.")
except Exception as e:
    st.write("An error occurred while loading the model:", e)
    loaded_svc_model = None

    
st.title('Streamlit CheckPoint 2')
#Create the values of the features
values = {
        'country': {
            'Kenya': 0.0,
            'Rwanda': 1.0,
            'Tanzania': 2.0,
            'Uganda': 3.0
        },
        'location_type': {
            'Rural': 0.0,
            'Urban': 1.0
        },
        'cellphone_access': {
            'Yes': 1.0,
            'No': 0.0
        },
        'gender_of_respondent': {
            'Female': 0.0,
            'Male': 1.0
        },
        'job_type': {
            'Self employed': 9.0,
            'Government Dependent': 4.0,
            'Formally employed Private': 3.0,
            'Informally employed': 5.0,
            'Formally employed Government': 2.0,
            'Farming and Fishing': 1.0,
            'Remittance Dependent': 8.0,
            'Other Income': 7.0,
            'Dont Know/Refuse to answer': 0.0,
            'No Income': 6.0
        },
        'relationship_with_head': {
            'Spouse': 5.0,
            'Head of Household': 1.0,
            'Other relative': 3.0,
            'Child': 0.0,
            'Parent': 4.0,
            'Other non-relatives': 2.0
        },
        'marital_status': {
            'Married/Living together': 2.0,
            'Widowed': 4.0,
            'Single/Never Married': 3.0,
            'Divorced/Seperated': 0.0,
            'Dont know': 1.0
        },
        'education_level': {
            'Secondary education': 3.0,
            'No formal education': 0.0,
            'Vocational/Specialised training': 5.0,
            'Primary education': 2.0,
            'Tertiary education': 4.0,
            'Other/Dont know/RTA': 1.0
        }
    }

st.text('Feature:')

# Display select boxes for categorical columns
selected_country = st.selectbox("Choose country", options=list(values['country'].keys()))

selected_county_numeric = values['country'][selected_country]# type: ignore

selected_location_type = st.selectbox("Choose location_type", options=list(values['location_type'].keys()))

selected_location_type_numeric = values['location_type'][selected_location_type] # type: ignore

selected_cellphone_access = st.selectbox("Choose cellphone_access", options=list(values['cellphone_access'].keys()))

selected_cellphone_access_numeric = values['cellphone_access'][selected_cellphone_access] # type: ignore

selected_gender_of_respondent = st.selectbox("Choose gender_of_respondent", options=list(values['gender_of_respondent'].keys()))

selected_gender_of_respondent_numeric = values['gender_of_respondent'][selected_gender_of_respondent] # type: ignore

selected_job_type = st.selectbox("Choose job_type", options=list(values['job_type'].keys()))

selected_job_type_numeric = values['job_type'][selected_job_type] # type: ignore

selected_relationship_with_head = st.selectbox("Choose relationship_with_head", options=list(values['relationship_with_head'].keys()))

selected_relationship_with_head_numeric = values['relationship_with_head'][selected_relationship_with_head] # type: ignore

selected_marital_status = st.selectbox("Choose marital_status", options=list(values['marital_status'].keys()))

selected_marital_status_numeric = values['marital_status'][selected_marital_status] # type: ignore

selected_education_level = st.selectbox("Choose education_level", options=list(values['education_level'].keys()))

selected_education_level_numeric = values['education_level'][selected_education_level] # type: ignore


age = st.slider('Select age_of_respondent', 16, 100, 25)
year= st.slider('Select Year',2016,2018,2016)

st.write('country:', selected_country)
st.write('location_type:', selected_location_type)
st.write('cellphone_access:', selected_cellphone_access)
st.write('gender_of_respondent:', selected_gender_of_respondent)
st.write('job_type:', selected_job_type)
st.write('relationship_with_head:', selected_relationship_with_head)
st.write('marital_status:', selected_marital_status)
st.write('education_level:', selected_education_level)
st.write('Year:',year)
st.write('age_of_respondent:', age)

if st.button('Predict'):
        # Create named features
        features = ['country', 'year', 'location_type', 'cellphone_access', 'age_of_respondent',
                    'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']
        if loaded_svc_model is not None:
            input_data = pd.DataFrame([[selected_county_numeric, year, selected_location_type_numeric, selected_cellphone_access_numeric,
                                    age, selected_gender_of_respondent_numeric, selected_relationship_with_head_numeric,
                                    selected_marital_status_numeric, selected_education_level_numeric, selected_job_type_numeric]],
                                    columns=features)
        # Make predictions using the model
            predicted_value = loaded_svc_model.predict(input_data)
            if predicted_value[0] == 1:
                st.write('Good probability to have a bank account')
            else:
                st.write('Bad probability to have a bank account')
        else:
            st.write("Model failed to load. Check for errors in the loading process.")
            
#link of the deployed project
#https://pythontuto-2n22prme65gpuftawozmxo.streamlit.app/
