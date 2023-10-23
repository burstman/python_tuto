import numpy as np
# reading the csv file , (with) ensure the closing of the  file after reading
with open("file_handling/Loan_prediction_dataset.csv", "r") as data:
    next(data)
    loan_amount = [float(line.strip().split(',')[9])
                   for line in data if line.strip().split(',')[9]]

# convert data from csv tp ndarray object
loan_amount_array = np.array(loan_amount)
# perform  basic statistical analysis
print("the mean of the loan amout:", np.mean(loan_amount_array))
print("the median of loan amount:", np.median(loan_amount_array))
print("the standat deviation:", np.std(loan_amount_array))
