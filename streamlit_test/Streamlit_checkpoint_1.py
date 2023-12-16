import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from joblib import dump




ordinal_encoder = OrdinalEncoder()

data = pd.read_csv('streamlit_test/Expresso_churn_dataset.csv')
# data.drop()
# print(data.describe())

data.info()

# print(data.isnull().sum())


def get_profile(data, location, title):
    # Generate the profile report
    profile = ProfileReport(data, title=title)
    # Display the report
    profile.to_notebook_iframe()
    # Or generate an HTML report
    profile.to_file(location)


def Fill_empty(df, formula):
    for column in df:
        if formula == 'mode':
            mode_value = data[column].mode()[0]
            data[column].fillna(mode_value, inplace=True)
        elif formula == 'mean':
            mean_value = data[column].mean()
            data[column].fillna(mean_value, inplace=True)
        elif formula == 'median':
            median_value = data[column].median()
            data[column].fillna(median_value, inplace=True)

#replace empty cells with mode for categorical variables and mean for numeric variables
Fill_empty(data[['ZONE1', 'ZONE2', 'FREQUENCE_RECH', 'ON_NET', 'ORANGE',
           'TIGO', 'TOP_PACK', 'FREQ_TOP_PACK', 'REGION', 'FREQUENCE']], 'mode')
Fill_empty(data[['MONTANT', 'REVENUE', 'ARPU_SEGMENT']], 'mean')

# print(data[['MONTANT', 'FREQUENCE_RECH', 'REVENUE']])

# print(data.isnull().sum())


#drop irrelevent variable 
new_data = data.drop(
    ['DATA_VOLUME', 'user_id', 'MRG'], axis=1)
# create the html report
# get_profile(data_droped, 'streamlit_test/zindi_report.html', 'zindi report')


def get_outlier(data, zscore_threshold):
    from scipy import stats
    z_scores = np.abs(stats.zscore(data))
    outliers_count = (z_scores > zscore_threshold).sum()
    print(outliers_count)
    return outliers_count


# get_outlier(new_data[['MONTANT', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE','ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK', 'REGULARITY']], 3)


def replace_outliers_with_median(data, columns, zscore_threshold):
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
            col_median = col_values.median()
            data_copy[col] = col_values.fillna(col_median)
        except ValueError:
            # Handling non-numeric columns
            print(f"Column '{col}' contains non-numeric data and was skipped.")

    return data_copy

#replace outlier with median
new_data = replace_outliers_with_median(new_data, [
                                        'MONTANT', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK', 'REGULARITY'], 2)

# get_outlier(new_data[['MONTANT', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE','ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK', 'REGULARITY']], 3)

new_data[['TOP_PACK', 'REGION', 'TENURE']] = ordinal_encoder.fit_transform(new_data[['TOP_PACK', 'REGION', 'TENURE']])
# new_data.info()        
        

# get_profile(new_data, 'streamlit_test/zindi_report.html', 'z indi report')
new_data = new_data.drop_duplicates()

#split the dataframe
def split_dataframe(df, percentage):
    if percentage is not None:
        split_index = int(len(df) * percentage)
        df_half1 = df.iloc[:split_index, :]
        df_half2 = df.iloc[split_index:, :]
        return df_half1, df_half2
    else:
        return df.copy()
data_part1,data_part2=split_dataframe(new_data,0.5)

#I have tried random forest and SGDClassifier. 
# but the results still  must be improved. 

X = data_part1[['ARPU_SEGMENT','REGION','TENURE']]  # features
y = data_part1['CHURN']  # target
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.42)  # splitting data with test size of 30%
#I have used  gridsearchCV to find the best parameter for random forest. 
'''param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=30), param_grid, cv=5, scoring='average_precision',verbose=2)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_estimator = grid_search.best_estimator

# Retrain with best parameters
best_estimator.fit(x_train, y_train)
y_pred = best_estimator.predict(x_test)# Measuring the accuracy of our model'''
Best_Parameters={'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 150}
# Create and train a Random Forest Classifier 
clf = RandomForestClassifier(**Best_Parameters,random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#save the model in file to use in with stremlit later
dump(clf,'streamlit_test/random_forest_model.joblib')

'''# Create an instance of SGDClassifier with verbose setting
sgd_classifier = SGDClassifier(loss='perceptron',alpha=0.001, random_state=42, verbose=1)

# Fit the classifier on the training data
sgd_classifier.fit(x_train, y_train)

# Make predictions on the scaled test set
y_pred = sgd_classifier.predict(x_test)'''

print(classification_report(y_test, y_pred,zero_division=0))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                               'Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()
