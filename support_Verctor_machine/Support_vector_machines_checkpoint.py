import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from ydata_profiling import ProfileReport

data = pd.read_csv(
    'support_Verctor_machine/Electric_cars_dataset.csv')

data.info()

data.describe()

print(data.head())

print(data.isnull().sum())

# function de get the profile of the data
def get_profile(data):
    # Generate the profile report
    profile = ProfileReport(data, title='Electric Vehicule Report')
    # Display the report
    profile.to_notebook_iframe()
    # Or generate an HTML report
    profile.to_file("support_Verctor_machine/electric_vehicule_report.html")

# I have a good correlation between 'Make', 'Electric Vehicle Type', 
# 'Clean Alternative Fuel Vehicle (CAFV) Eligibility' and our target 'Expected Price ($1k)'

# Select relevant features and the target variable
selected_features = ['Make', 'Electric Vehicle Type',
                     'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Expected Price ($1k)']

target_variable = 'Expected Price ($1k)'
data_selected = data[selected_features]
# their are corrupted data in 'Expected Price ($1k)' we delete those rows.

n_values_rows = data_selected[data_selected.apply(
    lambda row: 'N/' in row.values, axis=1)]

data_selected = data_selected[~data_selected.index.isin(n_values_rows.index)]
data_selected.dropna(inplace=True)


# We need to encode the  categorical columns 
label_encoder = LabelEncoder()
for col in ['Make', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility']:
    data_selected[col] = label_encoder.fit_transform(data_selected[col])

# Split data into features and target variable
X = data_selected.drop(columns=[target_variable])
y = data_selected[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM regression model
# these tweeks have give me a 0.74 "R squerred"
svm_model = SVC(kernel='rbf', C=50, gamma=1, verbose=True, shrinking=False)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = svm_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r_squered = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R squerred:{r_squered}")



