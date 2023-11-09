import pandas as pd

import plotly.express as px

# Load the dataset into a data frame using Python.
df = pd.read_csv('data_visualization_with_python/Africa_climate_change.csv')

print("head\n", df.head())

df.info()

print("describe\n", df.describe())

print("null value\n", df.isnull().sum())
# Clean the data as needed.
# I have dicovered some variable with missing values
# PRCP       287240
# TAVG         6376
# TMAX       100914
# TMIN       132058
# we can replace them with their mean because they are numerical variables.

df['PRCP'].fillna(df['PRCP'].mean(), inplace=True)

df['TAVG'].fillna(df['TAVG'].mean(), inplace=True)

df['TMAX'].fillna(df['TMAX'].mean(), inplace=True)

df['TMIN'].fillna(df['TMIN'].mean(), inplace=True)

print("null value\n", df.isnull().sum())

print(df.head())


# Plot a line chart to show the average temperature fluctuations in Tunisia and Cameroon
# Zoom in to only include data between 1980 and 2005, try to customize the axes labels.

# Filter data for Tunisia and Cameroon
filtered_df = df[df['COUNTRY'].isin(['Tunisia', 'Cameroon']) &
                 (df['DATE'].str[:4].astype(int).between(1980, 2005))]

# Extracting years from the DATE column
df['YEAR'] = pd.to_datetime(df['DATE']).dt.year

# Plotting line chart for Tunisia and Cameroon on the same chart
fig = px.line(filtered_df, x='DATE', y='TAVG', color='COUNTRY',
              labels={'TAVG': 'Temperature (°F)', 'DATE': 'Date'},
              title='Average Temperature Fluctuations in Tunisia and Cameroon')

# Update x-axis to display dates
fig.update_xaxes(type='category')

# Show the plot
fig.show()

# we can see that the Cameron and Tunisa have the same period of fluctuations but Tunisia is more cooler tha Cameroon.

# Create Histograms to show temperature distribution in Senegal between [1980,2000] and [2000,2023] (in the same figure)

# Extract year from dataframe
df['YEAR'] = pd.to_datetime(df['DATE']).dt.year

# Filtering data for Senegal
senegal_data = df.loc[df['COUNTRY'] == 'Senegal']


senegal_1980_2000 = senegal_data[(
    senegal_data['YEAR'] >= 1980) & (senegal_data['YEAR'] <= 2000)]
senegal_2000_2023 = senegal_data[(
    senegal_data['YEAR'] >= 2000) & (senegal_data['YEAR'] <= 2023)]

# Creating histograms for temperature distribution
fig = px.histogram(senegal_1980_2000, x='TAVG', nbins=20, histnorm='percent', opacity=0.7, color_discrete_sequence=['blue'],
                   labels={
                       'TAVG': 'Temperature (°F)', 'count': 'Percentage of Days'},
                   title='Temperature Distribution in Senegal (1980-2000)')

# Adding the second histogram for the [2000, 2023] period
fig.add_trace(px.histogram(senegal_2000_2023, x='TAVG', nbins=20, histnorm='percent', opacity=0.7, color_discrete_sequence=['green'],
                           labels={'TAVG': 'Temperature (°F)', 'count': 'Percentage of Days'}).data[0])

# Add custom legend titles using annotations
fig.add_annotation(x=110, y=100, text="[1980-2000]", showarrow=False, font=dict(size=12, color="blue"))
fig.add_annotation(x=110, y=97, text="[2000-2023]", showarrow=False, font=dict(size=12, color="green"))

# Updating the legend
fig.update_traces(showlegend=True)

# Show the plot
fig.show()
# We ca see the temperature ditrebution in between 2000-2023 is lot higher then 1980-2000 it can arrive at 83 °F 32% more than the older period.

fig = px.histogram(df, x='TAVG', color='COUNTRY', marginal='rug',
                   title='Average Temperature Distribution per Country',
                   labels={'TAVG': 'Average Temperature (°F)', 'COUNTRY': 'Country'})

# Customize layout
fig.update_layout(barmode='overlay')  # Overlay histograms for better comparison

# Show the chart
fig.show()