import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.svm import SVC


data = pd.read_csv('streamlit_checkpoint2/Financial_inclusion_dataset.csv')

# print(data.head())

# data.info()
# print(data.describe())


def get_profile(data, location, title):
    # Generate the profile report
    profile = ProfileReport(data, title=title)
    # Display the report
    profile.to_notebook_iframe()
    # Or generate an HTML report
    profile.to_file(location)

# Create a profile report

# get_profile(data,'streamlit_checkpoint2/Financial_Inclusion_in_Africa.html',title='Financial_Inclusion_in_Africa')


ordinal_encoder = OrdinalEncoder()

# Identify categorical columns
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

# Apply Ordinal Encoding to each categorical column
data[categorical_cols] = ordinal_encoder.fit_transform(data[categorical_cols])


def get_outlier(data, zscore_threshold):
    from scipy import stats
    z_scores = np.abs(stats.zscore(data))
    outliers_count = (z_scores > zscore_threshold).sum()
    print(outliers_count)
    return outliers_count
# show an outpout for the numbers of outliers
# get_outlier(data[['household_size','age_of_respondent']],3)

# replace the outliers with their modes
def replace_outliers_with_mode(data, columns, zscore_threshold):
    # Create a copy to avoid modifying the original DataFrame
    data_copy = data.copy()

    for col in columns:
        try:
            col_values = pd.to_numeric(data_copy[col], errors='coerce')
            z_scores = np.abs(stats.zscore(col_values.dropna()))

            outliers_mask = z_scores > zscore_threshold

            # Replace outliers with NaN in the column copy
            col_values.loc[col_values.dropna().index[outliers_mask]] = np.nan

            # Fill NaN values with column median
            col_mode = col_values.mode().iloc[0]
            data_copy[col] = col_values.fillna(col_mode)
        except ValueError:
            # Handling non-numeric columns
            print(f"Column '{col}' contains non-numeric data and was skipped.")

    return data_copy


data = replace_outliers_with_mode(
    data, ['household_size', 'age_of_respondent'], 2)

# get_outlier(data[['household_size', 'age_of_respondent']], 3)

#I have made two functions one for the forest model and the other using the SVC model.
#I have found the SVC is better with the accuracy arrive for the class 1 at 71%. 

def train_random_forest_model(data):
    X = data.drop(['household_size', 'bank_account',
                  'uniqueid'], axis=1)  # features
    
    y = data['bank_account']  # target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30)  # splitting data with test size of 30%

    Best_Parameters = {'max_depth': 30, 'max_features': 'sqrt',
                       'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 150}
    clf = RandomForestClassifier(**Best_Parameters, random_state=30)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    dump(clf, 'streamlit_checkpoint2/random_forest_classifier_model.joblib')

    # Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

    # Create confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                                   'Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

    return clf  # Returning the trained model


# clf = train_random_forest_model(data)

def train_svc_model(data, svc_params):
    X = data.drop(['household_size', 'bank_account',
                  'uniqueid'], axis=1)  # features    
    y = data['bank_account']  # target

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40)  # splitting data with test size of 30%

    # SVC parameters
    svc = SVC(**svc_params, verbose=True, shrinking=False)
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    # Save the model in a file to use it with Streamlit later
    dump(svc, 'streamlit_checkpoint2/svc_classifier_model.joblib')
    
    # Print classification report
    print(classification_report(y_test, y_pred, zero_division=0))

    # Create confusion matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                                   'Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.show()


svc_params = {'C': 100, 'kernel': 'rbf', 'gamma': 0.001}

train_svc_model(data, svc_params)


