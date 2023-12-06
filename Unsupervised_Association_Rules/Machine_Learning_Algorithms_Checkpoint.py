import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


data = pd.read_csv(
    'Unsupervised_Association_Rules/Microsoft_malware_dataset_min.csv')

print(data)

print(data.describe())

data.info()


# get profile crete a profile with the specified daraframe
def get_profile(data, location, title):
    from ydata_profiling import ProfileReport
    # Generate the profile report
    profile = ProfileReport(data, title=title)
    # Display the report
    profile.to_notebook_iframe()
    # Or generate an HTML report
    profile.to_file(location)


# get_profile(data, "Unsupervised_Association_Rules/microsoft_profile.html")

data = data.drop_duplicates()

print(data.isnull().sum())


    # we don't have a numeric values they are all categoric values we don't this function
def get_outlier(data, zscore_threshold):
    from scipy import stats
    z_scores = np.abs(stats.zscore(data))
    outliers_count = (z_scores > zscore_threshold).sum()
    print(outliers_count)
    return outliers_count


def replace_outliers_with_median(data, columns, zscore_threshold):
    for col in columns:
        col_values = pd.to_numeric(data[col], errors='coerce')

        z_scores = np.abs(stats.zscore(col_values.dropna()))
        outliers_mask = z_scores > zscore_threshold

        # Replace outliers with NaN in the column
        col_values.loc[col_values.dropna().index[outliers_mask]] = np.nan

        # Fill NaN values with column median
        data[col] = col_values.fillna(col_values.median())

    return data


# filling NAN values with mode
for column in ['Wdft_IsGamer', 'Census_IsVirtualDevice', 'SMode']:
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)
# fill 0 the values that are not filled assuming that they are false
data[['Firewall', 'IsProtected']] = data[['Firewall', 'IsProtected']].fillna(0)

print(data.isnull().sum())

# Encoding the categorical data

columns_to_encode = ['Census_OSEdition', 'OsPlatformSubRelease']
encoder = OrdinalEncoder()
data[columns_to_encode] = encoder.fit_transform(data[columns_to_encode])


data.info()

#drop duplicate
new_data = data.drop_duplicates()

#get_profile(new_data, 'Unsupervised_Association_Rules/microsoft_profile.html','microsoft_profile')

# i'm using the IsProtected as target and i have droped the others variables that are w=have no effect in target

X = new_data.drop(['IsProtected', 'CountryIdentifier',
                  'HasDetections', 'SMode'], axis=1)
y = new_data['IsProtected']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=10)




# Create the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='gini', splitter='best',
                             max_leaf_nodes=70, min_samples_leaf=10, max_depth=20)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Predict probabilities for the positive class
y_probs = clf.predict_proba(X_test)[:, 1]  # type: ignore

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate the AUC (Area Under the Curve) score
auc_score = roc_auc_score(y_test, y_probs)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

#best AUC that i have found it is 0.7.