'''
This file iterates through all the files in the Measurements_data folder and stores in a dictionary all the dataframes that have the same station_id. 
All dataframes from the same station are then concatenated and saved in the air_quality_data folder with the structure Country/City/station_name.csv.
'''

import os
import pandas as pd
import json
from tqdm import tqdm

stations_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/Stations_City_Table.json'

# Load station data 
with open(stations_file_path, 'r') as json_file:
    stations_data = json.load(json_file)

main_folder = r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/Measurements_data'
result_folder = r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/air_data'
df_by_key_path = r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/df_by_key.json'

df_by_key = {}

for year_folder in tqdm(os.listdir(main_folder), 'year number'):
    year_path = os.path.join(main_folder, year_folder)
    if year_folder != '.DS_Store':
        # print('YY', year_folder)
        #if year_folder == "year=2021":
        
        for month_folder in tqdm(os.listdir(year_path), 'month number'):
            month_path = os.path.join(year_path, month_folder)
            if month_folder != '.DS_Store':
            # if month_folder == "month=01":
                
                for csv_file in tqdm(os.listdir(month_path), 'file number', leave=False):
                    if csv_file.endswith(".csv.gz"):
                        csv_file_path = os.path.join(month_path, csv_file)
                        
                        key = int(csv_file_path.split('-')[1])
                        
                        df = pd.read_csv(csv_file_path, compression='gzip')

                        if key in df_by_key:
                            # Add DataFrame to the list of same key 
                            df_by_key[key].append(df)
                        else:
                            # Create a list for DataFrames with this key
                            df_by_key[key] = [df]

                
# Save each DataFrame in the right path
for key, liste_df in tqdm(df_by_key.items(), 'location id'):
    # Concatenate all DataFrames in the list 
    df_concat = pd.concat(liste_df, ignore_index=True)

    index = stations_data['id'].index(int(key))
    country = stations_data['country'][index]
    city = stations_data['city'][index]

    if (isinstance(country, str) and isinstance(city, str)):
        country_path = os.path.join(result_folder, country)
        if not os.path.exists(country_path):
            os.makedirs(country_path)

        if not os.path.exists(os.path.join(country_path, city)):
            os.makedirs(os.path.join(country_path, city))
        

        # Create name for the current file
        nom_fichier_resultat = f"{country}/{city}/station_{key}.csv"
        
        # Save the file in the right path
        df_concat.to_csv(os.path.join(result_folder, nom_fichier_resultat), index=False)
