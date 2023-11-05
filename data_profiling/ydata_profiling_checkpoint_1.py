import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder

df_report = pd.read_csv("data_profiling/EDUCATION_ATTAINMENT.csv")
print(df_report.info())
print(df_report.describe())

# hear we can count the number of countries in the database
# unique_country_count = df_report['country'].nunique()
# print(unique_country_count)

# Because the country column is a text the profile repot does not get into consideratrion 
# so we can encode the country column

# label encoder initialization
label_encoder = LabelEncoder()

#encode the country column
df_report_encoded = label_encoder.fit_transform(df_report['country'])


# put the encoded values of country column in to dic variable that we can use later 
country_mapping = dict(zip(df_report_encoded, df_report["country"]))  # type: ignore

# this an example of how we can get the country from the country column
# for i in range(119,128):
# print(country_mapping.get(i, "country not found"))  # type: ignore

# we can reduce the number of column to 10 column because the other columns are part of Age grade and 
# have a high postive corrolation 

column_to_profile = ['Ages[15-49]_All_grade1', 'Ages[15-49]_All_grade2', 'Ages[15-49]_All_grade3', 'Ages[15-49]_Male_grade1',
                     'Ages[15-49]_Male_grade2', 'Ages[15-49]_Male_grade3', 'Ages[15-49]_Female_grade1', 'Ages[15-49]_Female_grade2',
                     'Ages[15-49]_Female_grade3','country','year']

# we can average the row by country and by year to reduce the number of rows
numeric_columns=df_report[column_to_profile].select_dtypes(include=['number']).columns
average_by_year_country = df_report[column_to_profile].groupby(['country','year']).mean().reset_index()

# we can sort rows by year
averaged_by_year_and_country_sorted = average_by_year_country.sort_values(by='year')

print(averaged_by_year_and_country_sorted)


averaged_by_year_and_country_sorted['country'] = label_encoder.fit_transform(averaged_by_year_and_country_sorted['country'])

# with this 3 lines we can extract the html non average report 
# report = ProfileReport(df_report[column_to_profile], title="education report")

# report.to_notebook_iframe()

# report.to_file('data_profiling/world_bank_report_with_less_column.html')

# with this 3 lines we can extract the html averege report
average_report= ProfileReport(averaged_by_year_and_country_sorted,title="education_average")

average_report.to_notebook_iframe()

average_report.to_file('data_profiling/average_world_bank_education.html')

# the report say that we have 18 missing cells . one for every column that represent 1% for all data
# we have one duplicate row that we can delete it.
# when we use the average by country and by year we can rid the 2 problems above and reduce the numbers of rows
# we have a strong postive correlation for every column in tha data except the year column that have 0 correletion.



# In summary based on the correlation graph the year and the country have no influence in the rate of the completed grades.
