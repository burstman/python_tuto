import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

# labal encoder initialization
label_encoder = LabelEncoder()
# read the csv file and convert it to a dataframe
df_tunis_air = pd.read_csv('data_profiling_2/Tunisair_flights_dataset.csv')

# Encode the "Flight_ID" column
df_tunis_air['Flight_ID'] = label_encoder.fit_transform(
    df_tunis_air['Flight_ID'])

df_tunis_air['Aircraft_code'] = label_encoder.fit_transform(
    df_tunis_air['Aircraft_code'])

# Make some standart analysis with pandas
print(df_tunis_air.head())

df_tunis_air.info()

print(df_tunis_air.describe())

print(df_tunis_air.isnull().sum())

# coverte the dataframe to a html report
tunis_air_report = ProfileReport(df_tunis_air, title='tunis air report')

tunis_air_report.to_notebook_iframe()

tunis_air_report.to_file(
    'data_profiling_2/report_tunis_air_encoded_flight_id.html')

# In profile file there is 2 Alertes, "STATUS" variable is highly imbalanced we can see that ATA value have 86.9% part in the column.
# "Arrivel  delay" has 38168 zeros 35,4% of the total values of the columun. May be this values are not inserted.
# "STATUS" variable has no influance in "Arrival delay" in the correlation graph
# I have converted "Flight_ID" to see it has an impact in "Arrivel delay", and we see a negative correlation in it.
