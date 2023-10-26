import pandas as pd

data = {'Name': ['John', 'Mary', 'Bob', 'Sarah', 'Tom', 'Lisa'],
        'Department': ['IT', 'Marketing', 'Sales', 'IT', 'Finance', 'Marketing'],
        'Age': [30, 40, 25, 35, 45, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Salary': [50000, 60000, 45000, 55000, 70000, 55000],
        'Experience': [3, 7, 2, 5, 10, 4]}

employee_df = pd.DataFrame(data)

# Selecting the first 3 rows using iloc
first_3_rows=employee_df.iloc[:3]
print(first_3_rows)

# Selecting rows where Department is "Marketing" using loc

marketing_rows = employee_df.loc[employee_df['Department'] == 'Marketing']
print("\nRows where Department is 'Marketing' using loc:")
print(marketing_rows)

# Selecting Age and Gender columns for the first 4 rows using iloc
age_gender_first_4_rows = employee_df.iloc[:4, [2, 3]]
print("\nAge and Gender columns for the first 4 rows using iloc:")
print(age_gender_first_4_rows)

# Selecting Salary and Experience columns for rows where Gender is "Male" using loc
salary_experience_male_rows = employee_df.loc[employee_df['Gender'] == 'Male', ['Salary', 'Experience']]
print("\nSalary and Experience columns for rows where Gender is 'Male' using loc:")
print(salary_experience_male_rows)