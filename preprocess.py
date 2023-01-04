import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
 
def get_bls():
    df = pd.read_csv('raw_data/bls.csv', dtype = {'AREA': 'str', 'AREA_TYPE': 'str', 'OWN_CODE': 'str'}, low_memory = False)

    df = df.replace('*', '')
    df = df.replace('**', '')

    for column in ['H_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90']:
        df[column] = df[column].str.replace('#', '100')
    for column in ['A_MEAN', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']:
        df[column] = df[column].str.replace('#', '208000')
        
    df['PCT_RPT'] = df['PCT_RPT'].str.replace('~', '0.5')\

    for column in ['TOT_EMP', 'A_MEAN', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']:
        df[column] = df[column].str.replace(',', '')
    for column in ['TOT_EMP', 'EMP_PRSE', 'JOBS_1000', 'LOC_QUOTIENT',
    'PCT_TOTAL', 'PCT_RPT','H_MEAN', 'A_MEAN', 'MEAN_PRSE',
    'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90',
    'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']:
        df[column] = pd.to_numeric(df[column])
    
    return df

def get_state_rpp():
    df = pd.read_csv('raw_data/state_rpp.csv', header = 3, engine = 'python', skipfooter = 3)
    df = df.drop('GeoFips', axis = 1)
    df = df.rename(columns = {'GeoName': 'AREA_TITLE', '2021': 'RPP'})
    df = df.iloc[1:]
    return df

def get_city_rpp():
    df = pd.read_csv('raw_data/msa_rpp.csv', header = 3, engine = 'python', skipfooter = 4)
    df = df.drop('GeoFips', axis = 1)
    df = df.rename(columns = {'GeoName': 'AREA_TITLE', '2021': 'RPP'})
    df = df.iloc[2:]
    df['AREA_TITLE'] = df['AREA_TITLE'].str.replace(' \(Metropolitan Statistical Area\)', '', regex = True)
    df['AREA_TITLE'] = df['AREA_TITLE'].str.replace(' 2/', '')
    return df

def get_occupation_state():
    bls_df = get_bls()
    rpp_df = get_state_rpp()

    df = bls_df.query('AREA_TYPE == "2"')[['AREA_TITLE', 'PRIM_STATE', 'OCC_TITLE', 'TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN']]
    df = pd.merge(df, rpp_df, on ='AREA_TITLE', how = 'left')

    for statistic in ['A_MEAN', 'A_MEDIAN']:
        df['RPP_ADJUSTED_'+statistic] = df[statistic]/df['RPP']*100
    for statistic in ['A_MEAN', 'A_MEDIAN']:
        df = df.drop(statistic, axis = 1)
    df = df.drop('RPP', axis = 1)

    df.to_csv('processed_data/occupation_state.csv', index = False)

def get_occupation_city():
    bls_df = get_bls()
    rpp_df = get_city_rpp()

    df = bls_df.query('AREA_TYPE == "4"')[['AREA_TITLE', 'OCC_TITLE', 'TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN']]
    
    rpp_map = pd.DataFrame(df['AREA_TITLE'].unique(), columns = ['AREA_TITLE'])
    rpp_map = pd.merge(rpp_map, rpp_df, on ='AREA_TITLE', how = 'left')
    for index1, row1 in rpp_map.iterrows():
        if np.isnan(row1['RPP']):
            max_match = 0
            cities1, states1 = row1['AREA_TITLE'].split(', ')
            s1 = set(states1.split('-'))
            c1 = set(cities1.split('-'))
            for index2, row2 in rpp_df.iterrows():
                cities2, states2 = row2['AREA_TITLE'].split(', ')
                s2 = set(states2.split('-'))
                c2 = set(cities2.split('-'))
                if s1.intersection(s2) and c1.intersection(c2):
                    current_match = len(s1.intersection(s2)) + len(c1.intersection(c2))
                    if current_match > max_match:
                        rpp_map.at[index1, 'RPP'] = row2['RPP']
                        max_match = current_match

    df = pd.merge(df, rpp_map, on = 'AREA_TITLE', how = 'left')

    for statistic in ['A_MEAN', 'A_MEDIAN']:
        df['RPP_ADJUSTED_'+statistic] = df[statistic]/df['RPP']*100
    for statistic in ['A_MEAN', 'A_MEDIAN']:
        df = df.drop(statistic, axis = 1)
    df = df.drop('RPP', axis = 1)

    coordinates_map = pd.DataFrame(df['AREA_TITLE'].unique(), columns = ['AREA_TITLE'])
    loc = Nominatim(user_agent = 'GetLoc')
    latitudes = []
    longitudes = []
    for index, row in coordinates_map.iterrows():
        cities, states = row['AREA_TITLE'].split(', ')
        first_city = cities.split('-')[0]
        first_state = states.split('-')[0]
        getLoc = loc.geocode(first_city+', '+first_state)
        latitudes.append(getLoc.latitude)
        longitudes.append(getLoc.longitude)
        print(first_city+', '+first_state)
    coordinates_map['LATITUDE'] = latitudes
    coordinates_map['LONGITUDE'] = longitudes

    df = pd.merge(df, coordinates_map, on ='AREA_TITLE', how ='left')

    df.to_csv('processed_data/occupation_city.csv', index = False)

def get_occupation_industry():
    bls_df = get_bls()

    df = bls_df.query('AREA_TYPE == "1" & O_GROUP in ("total", "major", "detailed") & I_GROUP in ("cross-industry", "sector", "3-digit")')[['NAICS', 'NAICS_TITLE', 'I_GROUP', 'OCC_TITLE', 'TOT_EMP', 'PCT_TOTAL', 'A_MEAN', 'A_MEDIAN']]
    
    id_parent_map = pd.DataFrame(list(df.query('I_GROUP == "cross-industry"')['NAICS'].unique())
    +list(df.query('I_GROUP == "sector"')['NAICS'].unique())
    +list(df.query('I_GROUP == "3-digit"')['NAICS'].unique()), columns = ['NAICS'])

    sector_ids = []
    sector_parents = []
    for naics in df.query('I_GROUP == "sector"')['NAICS'].unique():
        sector_ids.append(naics)
        sector_parents.append('0')

    subsector_ids = []
    subsector_parents = []
    for naics in df.query('I_GROUP == "3-digit"')['NAICS'].unique():
        subsector_ids.append(naics[:3])
        for sector_id in sector_ids:
            if '-' in sector_id:
                start, end = sector_id.split('-')
                if naics[:2] in [str(i) for i in range(int(start), int(end)+1)]:
                    subsector_parents.append(sector_id)
            elif naics[:2] == sector_id:
                subsector_parents.append(sector_id)
            
    id_parent_map['IDS'] = ['0']+sector_ids+subsector_ids
    id_parent_map['PARENTS'] = ['']+sector_parents+subsector_parents

    df = pd.merge(df, id_parent_map, on = 'NAICS', how = 'left')

    df.to_csv('processed_data/occupation_industry.csv', index = False)