from sklearn.preprocessing import StandardScaler
import pandas as pd

# Sample data
data = {'Age': [25, 30, 35, 40],
        'Income': [50000, 80000, 60000, 120000]}

df = pd.DataFrame(data)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df)

# 'scaled_data' will contain the standardized values
print(scaled_data)