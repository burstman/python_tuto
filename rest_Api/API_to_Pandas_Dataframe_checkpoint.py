import requests
import pandas as pd
import json

#I will use  30days data for APOD
api_url_APOD = 'https://api.nasa.gov/planetary/apod?api_key=zjXIM59bSmkIprgNtGtpaYpxUZLK0HucHuWOGgfn&start_date=2023-11-01&end_date=2023-12-01'
#For NeoWs i have figure oit that i can only get the data for 7 days Max  for every requests.
api_url_NeoWs = 'https://api.nasa.gov/neo/rest/v1/feed?start_date=2023-11-01&api_key=zjXIM59bSmkIprgNtGtpaYpxUZLK0HucHuWOGgfn'

response_APOD = requests.get(api_url_APOD)
response_NeoWs = requests.get(api_url_NeoWs)
#we can print the headers to check our request limitations
#print(response_NeoWs.headers)

# Create a function convert the json data to dataframe 
def NASA_APOD_json_to_dataframe(response):
    if response.status_code == 200:
        data = response.json()  # Convert the JSON response to a Python dictionary
        df = pd.DataFrame(data, columns=['copyright', 'date', 'explanation', 'hdurl', 'media_type',
                          'service_version', 'title', 'url'])  # Create a DataFrame from the dictionary
        return df
    else:
        print("Failed to fetch data from the API. Status code:",
              response.status_code)
        return None

#excute it
# data = NASA_APOD_json_to_dataframe(response_APOD)
# print(data)

#this a tool that i have created to visualize the response requeste in a formatted way
def pretty_print_json(response):
    if response.status_code == 200:
        # Retrieve the JSON data
        json_data = response.json() 

        # Convert JSON data to a formatted string
        formatted_json = json.dumps(json_data, indent=4)

        # Print the formatted JSON
        print(formatted_json)
        print(type(json_data))
    else:
        print("Failed to fetch data from the API. Status code:",
              response.status_code)

# Create a function convert the json data NoeoWs to dataframe and visualize the image of the day
def NASA_NeoWs_json_to_dataframe(response):
    if response.status_code == 200:
        # Convert the JSON response to a Python dictionary
        data_json = response.json()
        
        if 'near_earth_objects' in data_json:
            neo_objects = data_json['near_earth_objects']
            
            # Create an empty list to store NEO data
            data_list = []

            # Iterate through each date's NEOs
            for date, neos in neo_objects.items():
                for neo in neos:
                    asteroid_id = neo.get('id')
                    asteroid_name = neo.get('name')
                    absolute_magnitude = neo.get('absolute_magnitude_h')
                    min_diameter_km = neo.get('estimated_diameter', {}).get(
                        'kilometers', {}).get('estimated_diameter_min')
                    relative_velocity = neo.get('close_approach_data', [{}])[0].get(
                        'relative_velocity', {}).get('kilometers_per_second')

                    data_list.append({
                        'Asteroid ID': asteroid_id,
                        'Asteroid Name': asteroid_name,
                        'Absolute Magnitude': absolute_magnitude,
                        'Minimal Estimated Diameter (km)': min_diameter_km,
                        'Relative Velocity (km/s)': relative_velocity,
                        'Close Approach Date': date  # Adding the date information
                    })

            # Create a DataFrame from the extracted data
            df = pd.DataFrame(data_list)

            # Return the DataFrame
            return df
        else:
            print("No 'near_earth_objects' found in the response.")
            return None  # Return None if the expected key is not found
    else:
        print("Failed to fetch data. Status code:", response.status_code)
        return None  # Return None in case of failure

#pretty_print_json(response_NeoWs)
response_NeoWs = requests.get(api_url_NeoWs)
df = NASA_NeoWs_json_to_dataframe(response_NeoWs)

#df.to_csv('rest_Api/neo.csv')
print(df)

#converting dataframe to a csv file
df.to_csv('rest_Api/neo.csv', index=False)