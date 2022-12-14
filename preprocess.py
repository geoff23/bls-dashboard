import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
 
def preprocess_bls(bls_df):
    bls_df = bls_df.replace('*','')
    bls_df = bls_df.replace('**','')

    replace_hashtag_columns = ['H_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90']
    for column in replace_hashtag_columns:
        bls_df[column] = bls_df[column].str.replace('#','100')
        
    replace_hashtag_columns2 = ['A_MEAN', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']
    for column in replace_hashtag_columns2:
        bls_df[column] = bls_df[column].str.replace('#','208000')
        
    bls_df['PCT_RPT'] = bls_df['PCT_RPT'].str.replace('~','0.5')

    remove_commas_columns = ['TOT_EMP', 'A_MEAN', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']
    for column in remove_commas_columns:
        bls_df[column] = bls_df[column].str.replace(',','')
        
    make_numeric_columns = ['TOT_EMP', 'EMP_PRSE', 'JOBS_1000', 'LOC_QUOTIENT',
    'PCT_TOTAL', 'PCT_RPT','H_MEAN', 'A_MEAN', 'MEAN_PRSE',
    'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90',
    'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

    for column in make_numeric_columns:
        bls_df[column] = pd.to_numeric(bls_df[column])
    
    return bls_df

def preprocess_bea(bea_df):
    bea_df = bea_df.iloc[2:]
    bea_df = bea_df.drop('GeoFips', axis=1)
    bea_df = bea_df.rename(columns={'GeoName': 'AREA_TITLE', '2020': 'RPP'})
    bea_df['AREA_TITLE'] = bea_df['AREA_TITLE'].str.replace(' \(Metropolitan Statistical Area\)','', regex=True)
    bea_df['AREA_TITLE'] = bea_df['AREA_TITLE'].str.replace(' 1/', '')
    return bea_df

bls_df = pd.read_csv('bls.csv', dtype = {'AREA': 'str', 'AREA_TYPE': 'str', 'OWN_CODE': 'str', 'NAICS': 'str'}, low_memory=False)
bls_df = preprocess_bls(bls_df)
bea_df = pd.read_csv('bea.csv', header=3, engine='python', skipfooter=3)
bea_df = preprocess_bea(bea_df)

def add_rpp(msa_df, bea_df=bea_df):
    df = pd.DataFrame(msa_df['AREA_TITLE'].unique(), columns =['AREA_TITLE'])
    rpp_df = pd.merge(df, bea_df, on ='AREA_TITLE', how ='left')
    for index1, row1 in rpp_df.iterrows():
        if np.isnan(row1['RPP']):
            max_match = 0
            cities1, states1 = row1['AREA_TITLE'].split(', ')
            s1 = set(states1.split('-'))
            c1 = set(cities1.split('-'))
            for index2, row2 in bea_df.iterrows():
                cities2, states2 = row2['AREA_TITLE'].split(', ')
                s2 = set(states2.split('-'))
                c2 = set(cities2.split('-'))
                if s1.intersection(s2) and c1.intersection(c2):
                    current_match = len(s1.intersection(s2)) + len(c1.intersection(c2))
                    if current_match > max_match:
                        rpp_df.at[index1, 'RPP'] = row2['RPP']
                        max_match = current_match
    msa_df = pd.merge(msa_df, rpp_df, on ='AREA_TITLE', how ='left')
    for statistic in ['A_MEAN', 'A_MEDIAN']:
        msa_df['RPP_ADJUSTED_'+statistic] = msa_df[statistic]/msa_df['RPP']*100
    return msa_df

def add_coordinates(msa_df):
    coordinates_df = pd.DataFrame(msa_df['AREA_TITLE'].unique(), columns =['AREA_TITLE'])
    latitudes = []
    longitudes = []
    loc = Nominatim(user_agent='GetLoc')
    for index, row in coordinates_df.iterrows():
        cities, states = row['AREA_TITLE'].split(', ')
        first_city = cities.split('-')[0]
        first_state = states.split('-')[0]
        getLoc = loc.geocode(first_city+', '+first_state)
        latitudes.append(getLoc.latitude)
        longitudes.append(getLoc.longitude)
        print(first_city+', '+first_state)
    coordinates_df['LATITUDE'] = latitudes
    coordinates_df['LONGITUDE'] = longitudes
    msa_df = pd.merge(msa_df, coordinates_df, on ='AREA_TITLE', how ='left')
    return msa_df

msa_df = bls_df.query('AREA_TYPE == "4"')[['AREA_TITLE', 'OCC_TITLE', 'TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN']]
msa_df = add_rpp(msa_df)
msa_df = add_coordinates(msa_df)
msa_df.to_csv('msa.csv', index=False)