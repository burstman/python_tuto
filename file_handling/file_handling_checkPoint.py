import numpy as np
# reading the csv file , (with) ensure the closing of the  file after reading
with open("file_handling/Loan_prediction_dataset.csv", "r") as data:

    # Skip the header row
    next(data)

    # convert the eighth column to float and check if it's not empty. Empty cell discarded
    # with if condition
    loan_amount = [float(line.strip().split(',')[8])
                   for line in data if line.strip().split(',')[8]]

# convert data from csv tp ndarray object
loan_amount_array = np.array(loan_amount)

# perform  basic statistical analysis
print("the mean of the loan amout:", np.mean(loan_amount_array))
print("the median of loan amount:", np.median(loan_amount_array))
print("the standat deviation:", np.std(loan_amount_array))
