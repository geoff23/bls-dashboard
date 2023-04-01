import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def get_states(occupation):
    occ_state_df = pd.read_csv('processed_data/occupation_state.csv')

    df = occ_state_df.query('OCC_TITLE == "{}"'.format(occupation))
    df = df.drop('OCC_TITLE', axis = 1)
    
    df = df.dropna()
    column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'RPP_ADJUSTED_A_MEAN', 'RPP_ADJUSTED_A_MEDIAN']
    for column_name in column_names:
        series = np.log(df[column_name])
        df[column_name+'_Z_SCORE'] = (series - series.mean())/series.std()
    df['AVERAGE_Z_SCORE'] = df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis = 1)
    for column_name in column_names:
        df = df.drop(column_name+'_Z_SCORE', axis=1)
    return df

def get_map(df):
    fig = go.Figure(data = go.Choropleth(
            locations = df['PRIM_STATE'],
            z = df['AVERAGE_Z_SCORE'],
            locationmode = 'USA-states',
            colorscale = 'viridis',
            text = df['AREA_TITLE']
            +'<br>Average z-score: '+df['AVERAGE_Z_SCORE'].round(2).astype(str)
            +'<br>Total employment: '+df['TOT_EMP'].astype(int).astype(str)
            +'<br>Location quotient: '+df['LOC_QUOTIENT'].round(2).astype(str)
            +'<br>RPP-adjusted mean annual wage: '+df['RPP_ADJUSTED_A_MEAN'].astype(int).astype(str)
            +'<br>RPP-adjusted annual median wage: '+df['RPP_ADJUSTED_A_MEDIAN'].astype(int).astype(str),
            colorbar = dict(
                title = dict(text = 'Average Z-Score', font = dict(color = '#cccccc')),
                tickfont = dict(color = '#cccccc')
            )
        ))

    fig.update_layout(
        geo_scope = 'usa',
        paper_bgcolor = '#333333',
        geo_bgcolor = '#333333',
        geo_lakecolor = '#333333',
        geo_landcolor = '#4d4d4d',
        geo_subunitcolor = '#666666',
        margin = dict(l = 20, r = 20, t = 20, b = 20)
    )
    return fig

def get_chart(df, statistic, bars, height, names):
    ranking = df[['AREA_TITLE', statistic]].sort_values(statistic, ascending = False)
    fig = px.bar(ranking.head(bars), x = statistic, y = 'AREA_TITLE', height = height)
    fig.update_layout(
        title = dict(
            text = 'Highest '+names[statistic][0],
            x = 0.5,
            yanchor = 'middle'
        ),
        xaxis = dict(
            title = '',
            ticks = 'outside',
            ticklen = 5, 
            tickcolor = '#4d4d4d',
            gridcolor = '#4d4d4d', 
            zerolinecolor = '#4d4d4d'
        ),
        yaxis = dict(
            title = '',
            categoryorder = 'array',
            categoryarray = ranking.iloc[::-1]['AREA_TITLE'],
            ticks = 'outside',
            ticklen = 10,
            tickcolor='#4d4d4d',
        ),
        paper_bgcolor = '#333333',
        plot_bgcolor = '#333333',
        font_color = '#cccccc'
    )
    fig.update_traces(
        marker_line_width = 0, 
        hovertemplate = '%{y} <br>'+names[statistic][1]+': %{x}'
    )
    return fig

def generate_figures(occupation):
    figs = []
    df = get_states(occupation)
    
    figs.append(get_map(df))

    names = {'AVERAGE_Z_SCORE':['Average Z-Score', 'Average z-score'],
            'TOT_EMP':['Total Employment', 'Total employment'], 
            'LOC_QUOTIENT':['Location Quotient', 'Location quotient'], 
            'RPP_ADJUSTED_A_MEAN': ['RPP-Adjusted Mean Annual Wage', 'RPP-adjusted mean annual wage'],
            'RPP_ADJUSTED_A_MEDIAN': ['RPP-Adjusted Annual Median Wage', 'RPP-adjusted annual median wage']}
    figs.append(get_chart(df, 'AVERAGE_Z_SCORE', 10, 500, names))
    
    for statistic in names:
        if statistic != 'AVERAGE_Z_SCORE':
            figs.append(get_chart(df, statistic, 5, 300, names))

    return figs


import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(
    __name__,
    path='/occupations/best-states',
    title='Best States')
@callback(
    Output('occ-state-map', 'figure'),
    Output('occ-state-overall', 'figure'),
    Output('occ-state-total-employment', 'figure'),
    Output('occ-state-location-quotient', 'figure'),
    Output('occ-state-mean-adjusted', 'figure'),
    Output('occ-state-median-adjusted', 'figure'),
    Input('occ-state-dropdown', 'value')
)
def update_output(value):
    return generate_figures(value)

figs = generate_figures('Data Scientists')
layout = html.Div([
    html.Div(
        className = 'dropdown', 
        children = [
            html.H1(
                className = 'text', 
                children = ['Best ', dcc.Link('States', href='/occupations/best-metropolitan-areas'), ' for']
                
            ),
            dcc.Dropdown(pd.read_csv('processed_data/occupation_state.csv')['OCC_TITLE'].unique(), 'Data Scientists', id = 'occ-state-dropdown')
        ]
    ),
    dcc.Graph(figure = figs[0], id = 'occ-state-map'),
    dcc.Graph(figure = figs[1], id = 'occ-state-overall'),
    html.Div(
        children = [
            html.Div(
                children = [
                    dcc.Graph(figure = figs[2], id = 'occ-state-total-employment'),
                    dcc.Graph(figure = figs[4], id = 'occ-state-mean-adjusted')
                ],
                style = {'flex': 0.5}
            ),
            html.Div(
                children = [
                    dcc.Graph(figure = figs[3], id = 'occ-state-location-quotient'),
                    dcc.Graph(figure = figs[5], id = 'occ-state-median-adjusted')
                ],
                style = {'flex': 0.5}
            )
        ], 
        style = {'display': 'flex', 'flex-direction': 'row'}
    )
])