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


df = pd.read_csv(
    'machine_learning_linear_regression/5G_energy_consumption_dataset.csv')
# print(df)
df.info()
# print(df.describe())
labelEncoder = LabelEncoder()
model = LinearRegression()


class L_R_Manip:
    def __init__(self,  df):
        self.df = df

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

    def number_of_outlier(self, zscore_threshold):
        self.zscore_threshold = zscore_threshold
        z_scores = self.get_zscore()
        outliers_count = (
            self.df[z_scores > self.zscore_threshold]).notnull().sum()
        print(len(self.df))
        return outliers_count

    def plot_data(self, x_col, y_col, title, model):
        plt.scatter(self.df[x_col], self.df[y_col], color="r")
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        # Generate a range of values for x
        x_range = np.linspace(self.df[x_col].min(
        ), self.df[x_col].max(), 100).reshape(-1, 1)

        # Plot the regression line
        plt.plot(x_range, model.predict(x_range), color="k")
        # plt.plot(x_col, model.predict(x_col), color="k")
        plt.show()

    def get_report(self, title, location):
        df_report = ProfileReport(self.df, title=title)
        df_report.to_notebook_iframe()
        df_report.to_file(location)

    def encode_column(self, labelEncoder):
        self.df = labelEncoder.fit_transform(self.df)

    def delete_outliers(self, zscore_threshold):
        z_scores = self.get_zscore()
        print(len(self.df))
        return self.df[~(np.abs(z_scores) >
                         zscore_threshold).any(axis=1)]

    def train_one_input(self, column1, column2, test_size, random_state, model, title):
        x = self.df[column1].values.reshape(-1, 1)
        y = self.df[column2].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))

        self.plot_data(column1, column2, title, model)

    def train_multi_inputs(self, *columns, target_column, test_size, random_state, model):
        x = self.df[list(columns)].values.reshape(-1, len(columns))
        y = self.df[target_column].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state)

        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        print("MSE:", mean_squared_error(y_test, predicted))
        print("R squared:", r2_score(y_test, predicted))


manip0 = L_R_Manip
manip1 = L_R_Manip
manip2 = L_R_Manip


df['Time'] = df['Time'].str.replace(' ', '')
df['Time'] = pd.to_numeric(df['Time'])

labelEncoder = LabelEncoder()
df['BS'] = labelEncoder.fit_transform(df['BS'])
# print(df)
# df.info()

# print(manip1(df[['load', 'Energy']]).number_of_outlier(3))

new_df=manip1(df).delete_outliers(zscore_threshold=3)
splitted_data  = manip1(new_df).split_dataframe(0.5)
first_split = splitted_data[0]
second_split = splitted_data[1]
# print(manip2(df_split[0]).number_of_outlier(3))
# print(manip2(df_split[1]).number_of_outlier(3))
# print(df_split[1].info())
"""
manip0(first_split[['load', 'Energy']]).train(column1='load', column2='Energy',
                                     test_size=0.35, random_state=40, model=model, title='regression first split')

manip0(second_split[['load', 'Energy']]).train(column1='load', column2='Energy',
                                     test_size=0.35, random_state=40, model=model, title='regression second split')
"""


manip2(first_split).train_multi_inputs('load', 'Time', 'TXpower',target_column='Energy', test_size=0.35, random_state=40, model=model)

manip2(second_split).train_multi_inputs('load', 'Time', 'TXpower',target_column='Energy', test_size=0.35, random_state=40, model=model)

