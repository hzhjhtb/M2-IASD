'''
This file is designed to retrieve the ID, country, city, and the first measurement date for each of the stations available on the OpenAQ platform. 
These pieces of information are stored in the Stations_Table.json file.
'''

import requests
from tqdm import tqdm
import json 

stations_data = {'id': [], 'country': [], 'city': [], 'first_date': []}
measurements_data = {'id': [], 'station_id': [], 'measurement_date': []}
parameters_list = []

# Get all possible parameters
get_parameters_url = "https://api.openaq.org/v2/parameters?limit=100&page=1&offset=0&sort=asc&order_by=id"

headers = {"accept": "application/json"}

parameters_response = requests.get(get_parameters_url, headers=headers)

if parameters_response.status_code == 200:
    json_data = parameters_response.json()

    for param_info in json_data['results']:
        param_name = param_info['name']
        parameters_list.append(param_name)
        measurements_data[param_name] = []

    # print(len(parameters_list))
    # print(measurements_data)
else:
    print(f"API request failed with status code: {parameters_response.status_code}")


# Get station data 
for page in tqdm(range(1, 486)):
    get_stations_url = "https://api.openaq.org/v2/locations?limit=100&page=" + str(page) + "&offset=0&sort=desc&radius=1000&order_by=lastUpdated&dump_raw=false"
    stations_response = requests.get(get_stations_url, headers=headers)

    if stations_response.status_code == 200:
        json_data = stations_response.json()

        for station_info in json_data['results']:
            station_id = station_info['id']
            station_country = station_info['country']
            station_city = station_info['city']
            station_first_date = station_info['firstUpdated']
            stations_data['id'].append(station_id)
            stations_data['country'].append(station_country)
            stations_data['city'].append(station_city)
            stations_data['first_date'].append(station_first_date)
    else:
        print(f"API request failed for page {page} with status code: {stations_response.status_code}")

print(len(stations_data['id']))

stations_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data extraction/Project/Stations_Table.json'
with open(stations_file_path, 'w') as json_file:
    json.dump(stations_data, json_file)
    print('file saved')

#------------------------ Code not used ---------------------------
# Get measurements data from API for each station but too long

stations_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data extraction/Project/Stations_Table.json'

with open(stations_file_path, 'r') as json_file:
    stations_data = json.load(json_file)

failed = {}
date_to_start = '2021-01-01T00:00:00Z'
date_to_finish = '2023-12-31T00:00:00Z'

# for i in tqdm(range(len(stations_data['id']))):
for i in tqdm(range(1)):
    current_id = stations_data['id'][i]
    print("current_id = ", current_id)
    current_first_date = stations_data['first_date'][i]

    flag = True
    page = 1
    id = 1
    while flag:
        if page % 10 ==0:
            print('page = ', page)
            
        get_measurements_url = "https://api.openaq.org/v2/measurements?date_from=" + str(date_to_start) + "&date_to=" + str(date_to_finish) + "&limit=100&page=" + str(page) + "&offset=0&sort=desc&radius=1000&location_id=" +str(current_id) + "&order_by=datetime"
        measurements_response = requests.get(get_measurements_url, headers=headers)

        if measurements_response.status_code == 200:
            json_data = measurements_response.json()
            # print(json_data)

            if json_data['results'] == []:
                flag = False
            else:
                for measurement_info in json_data['results']:
                    param_name = measurement_info['parameter']
                    param_value = measurement_info['value']
                    measurement_date = measurement_info['date']['utc']
                    
                    measurements_data['id'].append(id)
                    measurements_data['station_id'].append(current_id)
                    measurements_data['measurement_date'].append(measurement_date)
                    measurements_data[param_name].append(param_value)
                    id += 1

            # print(len(parameters_list))
            # print(measurements_data)
        else:
            print(f"API request failed with status code: {measurements_response.status_code}")
            if current_id not in failed:
                failed[current_id] = [page]
            else:
                failed[current_id].append(page)

        page += 1

measurements_file_path = r'/Users/vivienconti/Desktop/M2_IASD/Data extraction/Project/Measurements_Table.json'
with open(measurements_file_path, 'w') as json_file:
    json.dump(measurements_data, json_file)
    print('measurements_data file saved')