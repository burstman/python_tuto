import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


african_data = pd.read_csv('decision_tree/African_crises_dataset.csv')

print(african_data)
african_data.info()
print(african_data.head())
#print(african_data.isnull().sum())
african_data = african_data.drop(['country_code', 'country'], axis=1)
african_data['banking_crisis'] = african_data['banking_crisis'].map(
    {'crisis': 1, 'no_crisis': 0})
# Generate the profile report
# profile = ProfileReport(african_data, title='Africans dataset Profiling Report')

# Display the report
# profile.to_notebook_iframe()
# Or generate an HTML report
# profile.to_file("decision_tree/Africans_dataset.html")
print(african_data)
#z_scores = np.abs(stats.zscore(african_data))
#print("number of outlier", african_data[z_scores > 3].notnull().sum())
X = african_data[['year', 'exch_usd', 'banking_crisis', 'independence',
                  'domestic_debt_in_default', 'sovereign_external_debt_default',
                  'gdp_weighted_default', 'inflation_annual_cpi', 'currency_crises', 'inflation_crises']]
Y = african_data['systemic_crisis']
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3)
clf = RandomForestClassifier(n_estimators=30)
clf.fit(x_train, y_train)  # Training our model
y_pred = clf.predict(x_test)  # testing our model
# Measuring the accuracy of our model
print("Accuracy:", metrics.average_precision_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

