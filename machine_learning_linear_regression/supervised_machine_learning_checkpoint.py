import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Import you data and perform basic data exploration phase
df = pd.read_csv(
    'machine_learning_linear_regression/5G_energy_consumption_dataset.csv')
# Display general information about the dataset
df.info()
print(df.head())
print(df.describe())
# initialize the labelEncoder
labelEncoder = LabelEncoder()


# the creation of a object L_R_Manip  will help me facilitate the tweeks of the linears regressions.
class L_R_Manip:
    def __init__(self,  df):
        self.df = df

        # Split the data set inti parties, the percentage parameter help me for the size of the cut.
    def split_dataframe(self, percentage):
        if percentage is not None:
            split_index = int(len(self.df) * percentage)
            df_half1 = self.df.iloc[:split_index, :]
            df_half2 = self.df.iloc[split_index:, :]
            return df_half1, df_half2
        else:
            return self.df.copy()

    def get_zscore(self):
        return np.abs(stats.zscore(df))

        # Return the numbers of ouliers
    def number_of_outlier(self, zscore_threshold):
        self.zscore_threshold = zscore_threshold
        z_scores = self.get_zscore()
        outliers_count = (
            self.df[z_scores > self.zscore_threshold]).notnull().sum()
        print(len(self.df))
        return outliers_count

        # Plot data
    def plot_data(self, x_col, y_col, title):
        plt.scatter(self.df[x_col], self.df[y_col], color="r")
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        # Generate a range of values for x
        x_range = np.linspace(self.df[x_col].min(
        ), self.df[x_col].max(), 100).reshape(-1, 1)
        plt.show()
        # Profile report creating

    def get_report(self, title, location):
        df_report = ProfileReport(self.df, title=title)
        df_report.to_notebook_iframe()
        df_report.to_file(location)

        # Methode for deleting the ouliers
    def delete_outliers(self, zscore_threshold):
        z_scores = self.get_zscore()
        print(len(self.df))
        return self.df[~(np.abs(z_scores) >
                         zscore_threshold).any(axis=1)]
        # train a single linear regression

    def train_one_input_and_plot(self, column1, target_column, test_size, random_state, title):
        x = self.df[column1].values.reshape(-1, 1)
        y = self.df[target_column].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
        model = LinearRegression()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))

        self.plot_data(column1, target_column, title)

        # train multi linear regression

    def train_multi_inputs(self, *columns, target_column, test_size, random_state):
        x = self.df[list(columns)].values.reshape(-1, len(columns))
        y = self.df[target_column].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)
        model = LinearRegression()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))

        # methode for training a polynomial regression with plot. The plot take the x axe as the first column added
        # and the y axis as target_column

    def train_polynomial_regression_with_plot(self, *input_columns, target_column, test_size, random_state, degree):
        x = self.df[list(input_columns)].values.reshape(-1, len(input_columns))
        y = self.df[target_column].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        # Create a Polynomial Regression model
        lg = LinearRegression()
        poly = PolynomialFeatures(degree)

        # Fit the model to the training data
        x_train_fit = poly.fit_transform(x_train)

        # Make predictions on the test data
        lg.fit(x_train_fit, y_train)
        x_test_ = poly.fit_transform(x_test)
        predicted = lg.predict(x_test_)
        # Evaluate the model
        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))
        # Plot the results
        plt.scatter(x_test[:, 0], y_test, color="r", label="Actual")
        plt.title("Polynomial Regression")
        plt.xlabel(input_columns[0])  # Adjust as needed
        plt.ylabel(target_column)

        # Sort the values for better plotting
        sort_indices = np.argsort(x_test[:, 0])
        plt.plot(x_test[:, 0][sort_indices], predicted[sort_indices],
                 color="k", label="Regression Line")

        plt.legend()
        plt.show()

        # Training a polynomial regression without a plot

    def train_polynomial_regression(self, *input_columns, target_column, test_size, random_state, degree):
        x = self.df[list(input_columns)].values.reshape(-1, len(input_columns))
        y = self.df[target_column].values
        # split data set
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        # Create a Polynomial Regression model
        lg = LinearRegression()
        poly = PolynomialFeatures(degree)

        # Fit the model to the training data
        x_train_fit = poly.fit_transform(x_train)

        # Make predictions on the test data
        lg.fit(x_train_fit, y_train)
        x_test_ = poly.fit_transform(x_test)
        predicted = lg.predict(x_test_)
        # Evaluate the model
        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))
        print("---------------------------")


# ceate some variable objects
manip0 = L_R_Manip
manip1 = L_R_Manip
manip2 = L_R_Manip
poly_degree_manip = L_R_Manip


# we can convert Time variable to numeric
# deleting the space betwwen numbers
df['Time'] = df['Time'].str.replace(' ', '')
# convert to numeric
df['Time'] = pd.to_numeric(df['Time'])
# Encode categorical features
labelEncoder = LabelEncoder()
df['BS'] = labelEncoder.fit_transform(df['BS'])

manip0(df).get_report(title='5G report', location='machine_learning_linear_regression/5G_report.html')
# no missing or corrupted values and no duplicates row detected from the report

# Handling outliers
df_no_outlier = manip1(df).delete_outliers(zscore_threshold=3)
splitted_data = manip1(df).split_dataframe(0.5)

# the commented lines are some tweeks that i have tried and they have not give me a good results
"""
manip0(first_split[['load', 'Energy']]).train(column1='load', column2='Energy',
                                     test_size=0.35, random_state=40, model=model, title='regression first split')

manip0(second_split[['load', 'Energy']]).train(column1='load', column2='Energy',
                                     test_size=0.35, random_state=40, model=model, title='regression second split')



# manip2(new_df).train_multi_inputs('Energy', 'TXpower',target_column='load', test_size=0.35, random_state=40, model=model)

# manip0(df).plot_data('Energy','TXpower',model,"test")
# manip0(df).train_multi_inputs('TXpower', target_column='Energy',test_size=0.4, random_state=30, model=model)

"""

# I have got a good results with polynomial regression at degree 7 and still we can tweek with different cuts of data.
silce_no_outlier = poly_degree_manip(splitted_data[0]).delete_outliers(3)
#Loop for different degree
for i in range(1, 15):

    print("degree", i)
    poly_degree_manip(silce_no_outlier).train_polynomial_regression(
        'load', 'TXpower', target_column='Energy', test_size=0.35, random_state=40, degree=i)
