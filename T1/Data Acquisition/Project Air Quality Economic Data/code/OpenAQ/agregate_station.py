'''
The purpose of this file is to obtain a JSON file with the same keys as the Stations_Table.json file but with a unique measurement station per city
saved as Stations_City_Table.json.
'''

import json
import pandas as pd

stations_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data extraction/Project/Stations_Table.json'

# Load station data
with open(stations_file_path, 'r') as json_file:
    stations_data = json.load(json_file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(stations_data)

# Convert 'city' column values to lowercase for case-insensitive comparison
df['city'] = df['city'].str.lower()

# Drop duplicate rows based on the 'city' column
unique_df = df.drop_duplicates(subset='city')
data_dict = unique_df.to_dict(orient='list')

print(len(data_dict['id']))

# Save the dictionary to a JSON file
with open(r'/Users/vivienconti/Desktop/M2_IASD/Data extraction/Project/Stations_City_Table.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=2)


