import pandas as pd
import sklearn.preprocessing as sn
client_0_bills = pd.read_csv(
    "data_preprocessing/STEG_BILLING_HISTORY.csv", delimiter=",")


# print the 10 first lines
print(client_0_bills.head(10))
# print the number of rows
print("the number of row :", len(client_0_bills))


client_0_bills = pd.read_csv(
    "data_preprocessing/STEG_BILLING_HISTORY.csv", delimiter=",")

client_0_bills.info()

categorical_features = client_0_bills.select_dtypes(
    include=['object', 'category'])
num_categorical_features = categorical_features.shape[1]
print("Number of categorical features in the DataFrame:", num_categorical_features)

memory_usage_bytes = client_0_bills.memory_usage().sum()

# Convert bytes to megabytes
memory_usage_megabytes = memory_usage_bytes / (1024 * 1024)

print("Memory usage of the DataFrame:", memory_usage_megabytes, "MB")

# potential missing values
print(client_0_bills.isnull().sum())

# we have detected 48 missing entry in column counter_number, we can neglect those 48 lines because 48<<4476748 lines
client_0_bills.dropna(subset=['counter_number'], inplace=True)

print(client_0_bills.isnull().sum())
# we have detected 4531 missing entry in colomn reading_remarque, we can replace it by their mean because they are numeric
# Impute missing values with the mean of the column
mean_value = client_0_bills['reading_remarque'].mean()
client_0_bills['reading_remarque'].fillna(mean_value, inplace=True)

print(client_0_bills.isnull().sum())

# we have detected a mixing data type in the 4th column (counter_statue)

mixed_data_types = set(
    client_0_bills['counter_statue'].apply(type))  # type: ignore

print("Mixed data types in the column are :", mixed_data_types)

# print the sum of int values and string values
print("20 first string")
string_count = client_0_bills['counter_statue'].apply(
    lambda x: isinstance(x, str)).sum()
print(string_count)
int_count = client_0_bills['counter_statue'].apply(
    lambda x: isinstance(x, int)).sum()
print(
    f"counter_statue: number of int value are {int_count} and string value are {string_count} ")

# Extract and print the first 20 cells with string values in 'counter_statue' we can do the same thing for the int value
print('print the first 20 value of counter_statue column string part:')
string_cells = client_0_bills[client_0_bills['counter_statue'].apply(
    lambda x: isinstance(x, str))]['counter_statue'].head(20).values
print(string_cells)
# all value that i have seen are 0 and 1, i think we can convert the str types to int types to resolve the problème
# when i want to convert to client_0_bills['counter_statue'].astype(int) it's gives an error that tell me that there is
# a string 'o' that can't be converted
# cheking the rows that have the 'o' strings on it
result0 = client_0_bills[client_0_bills['counter_statue'].apply(
    lambda x: isinstance(x, str) and 'o' in x)]
print(result0)
# we have detected only one row number 1900542
client_0_bills.at[1900542, "counter_statue"] = 0
print(client_0_bills.loc[1900542])

#there is 'A' string in counter_statue that i have detected with the same processus 
result1 = client_0_bills[client_0_bills['counter_statue'].apply(
    lambda x: isinstance(x, str) and 'A' in x)]
print(result1)
#I have found 13 rows [1923231:1923276] we can convert them to 1 int
client_0_bills.loc[1923231:1923276, "counter_statue"] = 1

client_0_bills['counter_statue']=client_0_bills['counter_statue'].astype(int)

data_types = set(client_0_bills['counter_statue'].apply(type))  # type: ignore

#test the 'counter_statue' column type
print(" data types in the 'counter_statue' column:", data_types)

# descriptive analysis on numeric features (columns)

print(client_0_bills.describe())

# second methide for filtering the client_id with id = train_Client_0

client_records_method1 = client_0_bills.query('client_id == "train_Client_0"')

print(client_records_method1)

# second methode for filtering the client_id with id = train_Client_0

client_records_method2 = client_0_bills[client_0_bills['client_id'] == 'train_Client_0']

print(print(client_records_method2))

#Transform the 'counter_type' feature to a numeric variable with Labelencoder .
encoder = sn.LabelEncoder()

client_0_bills['counter_type'] = encoder.fit_transform(
    client_0_bills['counter_type'])

#Deleteing the counter_type column
client_0_bills.drop('counter_statue', axis=1)

print(client_0_bills.info())


