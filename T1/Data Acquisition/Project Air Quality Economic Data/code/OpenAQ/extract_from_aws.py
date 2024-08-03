'''
This file is designed to retrieve from the S3 bucket all measurements conducted since 2021 by all stations listed in the Stations_City_Table.json file. 
The downloaded data is stored in a Measurements_data folder with the following structure: Year/Month/station_id_day_of_measure.csv
'''

import os
import json
from tqdm import tqdm

# Load station data
stations_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/Stations_City_Table.json'

with open(stations_file_path, 'r') as json_file:
    stations_data = json.load(json_file)

# Configure AWS CLI
aws_region = 'us-east-1' 

# station_id = [5487, 5240]
completed_station = []

for station in tqdm(stations_data['id']):
    # S3 bucket details
    s3_bucket = 'openaq-data-archive'
    s3_prefix = 'records/csv.gz/locationid=' + str(station) + '/'

    # Local directory to save the data
    local_directory = '/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/Measurements_data'
    years_to_copy = range(2021, 2024)

    for year in years_to_copy:
        s3_path = f's3://{s3_bucket}/{s3_prefix}year={year}'
        local_path = os.path.join(local_directory, f'year={year}')

        # Create the local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)

        # Use the AWS CLI command to copy data
        os.system(f'aws s3 cp --no-sign-request {s3_path} {local_path} --recursive > /dev/null 2>&1')

    completed_station.append(station)
    # Save the processed IDs to a log file
    with open(r'/Users/vivienconti/Desktop/M2_IASD/Data_extraction/Project/id_log_file.txt', 'a') as log_file:
        log_file.write(f"{completed_station}\n")

print("Data extraction completed.")
