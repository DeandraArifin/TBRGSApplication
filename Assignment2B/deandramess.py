import pandas as pd
import numpy as np
import re
from enum import Enum

class Site_Type(Enum):
    INT = 'Intersection'
    POS = 'Pedestrian Crossing'
    FLASHPX = 'Pedestrian Crossing'
    FIREWW = 'Fire Station Exit'
    RBTMTR = 'Roundabout'
    FIRESIG = 'Fire Station Exit'
    AMBUSIG = 'Ambulance Exit'
    RAMPMTR = 'Mamp Metering'
    BUSSIG = 'Bus Signal'
    TMPPOS = 'Temporary Site'
    OHLANE = 'High Occupancy Lane'
    

def remove_non_alphanumeric_characters(str):
    if pd.isna(str):
        return None
    cleaned = re.sub(r'\W+', '', str).upper()
    return cleaned

def map_site_type_to_key(site_type_str):
    key = remove_non_alphanumeric_characters(site_type_str)
    try:
        return Site_Type[key].value  
    except KeyError:
        return 'Unknown' 

def remove_leading_zeroes(val):
    if pd.isna(val):
        return None
    try:
        return str(int(val)) 
    except ValueError:
        return val  

scats_site_data = pd.read_csv('Resources/SCATSSite.csv', sep=';', usecols=range(5), skiprows=9, on_bad_lines='skip', header=0)
scats_data = pd.read_excel('Resources/Scats_Data_October_2006.xls', sheet_name='Data', skiprows=1)
traffic_count_data = pd.read_csv('Resources/Traffic_Count_Locations_with_LONG_LAT.csv')

scats_site_data['Site Type'] = scats_site_data['Site Type'].apply(map_site_type_to_key)
scats_data['CD_MELWAY'] = scats_data['CD_MELWAY'].apply(remove_non_alphanumeric_characters)
scats_data['SCATS Number'] = scats_data['SCATS Number'].apply(remove_leading_zeroes)

scats_data = scats_data.drop(columns=['Location','HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY'])

scats_site_data.rename(columns={'Site Number' : 'SCATS Number'}, inplace=True)
# print(scats_data)
scats_data['SCATS Number'] = scats_data['SCATS Number'].astype(int)
scats_site_data['SCATS Number'] = scats_site_data['SCATS Number'].astype(int)
scats_site_data = scats_site_data.drop_duplicates(subset='SCATS Number', keep='first')

filtered_data = pd.merge(scats_data, scats_site_data, on='SCATS Number', how='inner')

print(filtered_data)

# traffic_count_data = traffic_count_data.drop(columns=['TFM DESC', ])
# matches = scats_data['NB_LATITUDE'].isin(traffic_count_data['Y'])
# print(f"TFM_ID matches: {matches.sum()} out of {len(scats_data)}")
# matched_rows = scats_data[matches]
# print(matched_rows)

# Filter rows in traffic_count_data where TFM_ID is 970
# specific_rows = traffic_count_data[traffic_count_data['ROAD_NBR'] == 2000]
# print(specific_rows)


# Try matching on ROAD_NBR
# matches = scats_data['SCATS Number'].isin(traffic_count_data['ROAD_NBR'])
# print(f"ROAD_NBR matches: {matches.sum()} out of {len(scats_data)}")
