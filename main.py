import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.gridspec import GridSpec

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
    bea_df.drop('GeoFips', axis=1)
    bea_df = bea_df.rename(columns={'GeoName': 'AREA_TITLE', '2020': 'RPP'})
    bea_df['AREA_TITLE'] = bea_df['AREA_TITLE'].str.replace(' \(Metropolitan Statistical Area\)','', regex=True)
    return bea_df

def add_rpp(bls_df, bea_df):
    merged_df = pd.merge(bls_df, bea_df, on ='AREA_TITLE', how ='left')

    for index1, row1 in merged_df.iterrows():
        if np.isnan(row1['RPP']):
            max_match = 0
            cities1, states1 = row1['AREA_TITLE'].split(', ')
            s1 = set(states1.split('-'))
            c1 = set(cities1.split('-'))
            for index2, row2 in bea_df.iterrows():
                cities2, states2 = row2['AREA_TITLE'].split(', ')
                s2 = set(states2.split('-'))
                c2 = set(cities2.split('-'))
                current_match = len(s1.intersection(s2)) + len(c1.intersection(c2))
                if current_match > max_match:
                    max_match = current_match
                    merged_df.at[index1,'RPP'] = row2['RPP']
    return merged_df

def get_top_metropolitan_areas(occupation, bls_df, bea_df):
    bls_df = bls_df.query('AREA_TYPE == "4" and OCC_TITLE == "{}"'.format(occupation))
    bls_df = bls_df[['AREA_TITLE', 'TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN']]
    merged_df = add_rpp(bls_df, bea_df)
    for statistic in ['A_MEAN', 'A_MEDIAN']:
        merged_df['RPP_ADJUSTED_'+statistic] = merged_df[statistic]/merged_df['RPP']*100
    merged_df = merged_df.drop('RPP', axis = 1)
        
    column_names = list(merged_df.columns)
    column_names.remove('AREA_TITLE')
    for column_name in column_names:
        merged_df[column_name+'_Z_SCORE'] = (merged_df[column_name] - merged_df[column_name].mean())/merged_df[column_name].std()
    merged_df['AVERAGE_Z_SCORE'] = merged_df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis=1)
    for column_name in column_names:
        merged_df = merged_df.drop(column_name+'_Z_SCORE', axis=1)
    return merged_df

@ticker.FuncFormatter
def kmbt_format(num, pos):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def visualize_top_metropolitan_areas(occupation, bls_df, bea_df):
    names = {'TOT_EMP':'Total Employment', 
             'LOC_QUOTIENT':'Location Quotient', 
             'A_MEAN':'Mean Annual Wage',
             'A_MEDIAN':'Annual Median Wage',
             'RPP_ADJUSTED_A_MEAN': 'RPP-Adjusted Mean Annual Wage',
             'RPP_ADJUSTED_A_MEDIAN': 'RPP-Adjusted Annual Median Wage'}
    df = get_top_metropolitan_areas(occupation, bls_df, bea_df)
    
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(5, 2, figure=fig)
    fig.set_figheight(13)
    fig.set_figwidth(13)
    
    i = 0
    for statistic in names:
        axis = fig.add_subplot(gs[i//2,i%2])
        ranking = df[['AREA_TITLE', statistic]].sort_values(statistic, ascending = False)
        sns.barplot(x=statistic, y='AREA_TITLE', data=ranking.head(5), ax = axis, palette = sns.color_palette('Blues_r', 10))
        sns.despine(left=True, bottom=True)
        axis.set(title = 'Highest '+names[statistic], xlabel='', ylabel='')
        axis.xaxis.set_major_formatter(kmbt_format)
        i += 1
        
    axis = fig.add_subplot(gs[3:, :])
    ranking = df[['AREA_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    sns.barplot(x="AVERAGE_Z_SCORE", y="AREA_TITLE", data=ranking.head(10), ax = axis, palette = sns.color_palette('Blues_r', 20))
    sns.despine(left=True, bottom=True)
    axis.set(title = 'Best Metropolitan Areas Overall', xlabel='Average Z-Score', ylabel='')
    
    fig.suptitle('Best Metropolitan Areas For '+occupation)
    plt.show()

bls_df = pd.read_csv('bls.csv', dtype = {'AREA': 'str', 'AREA_TYPE': 'str', 'OWN_CODE': 'str', 'NAICS': 'str'}, low_memory=False)
bls_df = preprocess_bls(bls_df)
bea_df = pd.read_csv('bea.csv', header=3, engine='python', skipfooter=3)
bea_df = preprocess_bea(bea_df)

from dash import Dash, html, dcc, Input, Output
import plotly.express as px

app = Dash(__name__)

def generate_figures(occupation, bls_df, bea_df):
    df = get_top_metropolitan_areas(occupation, bls_df, bea_df)
    ranking = df[['AREA_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    figs = []
    fig = px.bar(ranking.head(10), x='AVERAGE_Z_SCORE', y='AREA_TITLE',  height = 500, labels={'AREA_TITLE':'Metropolitan Area', 'AVERAGE_Z_SCORE': 'Average Z-Score'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, title='Highest Average Z-Score', title_x = 0.5, xaxis_title='', yaxis_title = '')
    fig.update_traces(hovertemplate='Metropolitan area: %{y} <br>Average z-score: %{x}')
    fig.update_layout(
        xaxis = dict(ticks = 'outside', tickcolor='white', ticklen=5),
        yaxis = dict(ticks = 'outside', tickcolor='white', ticklen=10)
    )
    figs.append(fig)

    names = {'TOT_EMP':'Total Employment', 
            'LOC_QUOTIENT':'Location Quotient', 
            'A_MEAN':'Mean Annual Wage',
            'A_MEDIAN':'Annual Median Wage',
            'RPP_ADJUSTED_A_MEAN': 'RPP-Adjusted Mean Annual Wage',
            'RPP_ADJUSTED_A_MEDIAN': 'RPP-Adjusted Annual Median Wage'}
    for statistic in names:
        ranking = df[['AREA_TITLE', statistic]].sort_values(statistic, ascending = False)
        fig = px.bar(ranking.head(5), x=statistic, y='AREA_TITLE', height = 300, labels={'AREA_TITLE':'Metropolitan Area', statistic: names[statistic]})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title='Highest '+names[statistic], title_x = 0.5, xaxis_title='', yaxis_title = '')
        fig.update_traces(hovertemplate='Metropolitan area: %{y} <br>'+names[statistic].capitalize()+': %{x}')
        fig.update_layout(
            xaxis = dict(ticks = 'outside', tickcolor='white', ticklen=5),
            yaxis = dict(ticks = 'outside', tickcolor='white', ticklen=10)
        )
        figs.append(fig)
    return figs

def get_occupations(bls_df):
    bls_df = bls_df.query('AREA_TYPE == "4"')
    return bls_df['OCC_TITLE'].unique()

@app.callback(
    Output('overall', 'figure'),
    Output('total-employment', 'figure'),
    Output('mean', 'figure'),
    Output('mean-adjusted', 'figure'),
    Output('location-quotient', 'figure'),
    Output('median', 'figure'),
    Output('median-adjusted', 'figure'),
    Input('dropdown', 'value')
)
def update_output(value, bls_df = bls_df, bea_df = bea_df):
    figs = generate_figures(value, bls_df, bea_df)
    return figs[0], figs[1], figs[3], figs[5], figs[2], figs[4], figs[6]


figs = generate_figures('All Occupations', bls_df, bea_df)
app.layout = html.Div([
    html.Div([
        html.Div(children=[
            html.H1(children = 'Best Metropolitan Areas for', style = {'font-family':'arial', 'margin-right':'0.67em', 'font-weight': 'normal', 'font-size': 16}),
            dcc.Dropdown(get_occupations(bls_df), 'Data Scientists', style = {'font-family':'arial', 'flex': 1}, id='dropdown')
        ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-bottom': 50}),
        dcc.Graph(figure=figs[0], id='overall'),
        html.Div([
            html.Div(children=[
                dcc.Graph(figure=figs[1], id='total-employment'),
                dcc.Graph(figure=figs[3], id='mean'),
                dcc.Graph(figure=figs[5], id='mean-adjusted')
            ], style={'flex': 0.5}),
            html.Div(children=[
                dcc.Graph(figure=figs[2], id='location-quotient'),
                dcc.Graph(figure=figs[4], id='median'),
                dcc.Graph(figure=figs[6], id='median-adjusted')
            ], style={'flex': 0.5}) 
        ], style={'display': 'flex', 'flex-direction': 'row'})
    ], style = {'margin-top':100, 'margin-bottom':100, 'margin-left': 200, 'margin-right': 200})
])

if __name__ == '__main__':
    app.run_server(debug=True) #Math occupation does not work