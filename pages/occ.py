import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

occ_df = pd.read_csv('occ.csv')

def get_occupations(area, occ_df = occ_df):
    df = occ_df.query('AREA_TITLE == "{}"'.format(area))
    df = df.drop('AREA_TITLE', axis = 1)

    parents = []
    for index, row in df.iterrows():
        if row['O_GROUP'] == 'total':
            total = row['OCC_TITLE']
            parents.append('')
        elif row['O_GROUP'] == 'major':
            major = row['OCC_TITLE']
            parents.append(total)
        else:
            parents.append(major)
    df['PARENTS'] = parents

    if area == 'U.S.':
        df['LOC_QUOTIENT'] = 1
    df = df.dropna()

    for i in ['major', 'detailed']:
        z_score_df = df.query('O_GROUP == "{}"'.format(i))
        column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN']
        for column_name in column_names:
            series = np.log(z_score_df[column_name])
            z_score_df[column_name+'_Z_SCORE'] = (series - series.mean())/series.std()
        z_score_df['AVERAGE_Z_SCORE'] = z_score_df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis = 1)
        df = pd.merge(df, z_score_df[['OCC_TITLE', 'AVERAGE_Z_SCORE']], on = 'OCC_TITLE', how = 'left')

    z_scores = []
    for index, row in df.iterrows():
        if not np.isnan(row['AVERAGE_Z_SCORE_x']):
            z_scores.append(row['AVERAGE_Z_SCORE_x'])
        elif not np.isnan(row['AVERAGE_Z_SCORE_y']):
            z_scores.append(row['AVERAGE_Z_SCORE_y'])
        else:
            z_scores.append(np.nan)
    df['AVERAGE_Z_SCORE'] = z_scores

    values = []
    for index, row in df.iterrows():
        if row['O_GROUP'] == 'detailed':
            values.append(row['TOT_EMP'])
        else:
            values.append(0)
    df['VALUES'] = values
    return df

def map_occupations(df):
    fig = go.Figure(go.Treemap(
        branchvalues = 'remainder',
        labels = df['OCC_TITLE'],
        parents = df['PARENTS'],
        values = df['VALUES'],
        marker = dict(
            colors = df['AVERAGE_Z_SCORE'],
            colorscale = 'magma',
            colorbar = dict(
                title = dict(text = 'Average Z-Score', font = dict(color = '#cccccc')),
                tickfont = dict(color = '#cccccc')
            )
        ),
        hovertemplate = df['OCC_TITLE']
        +'<br>Average z-score: '+df['AVERAGE_Z_SCORE'].round(2).astype(str)
        +'<br>Total employment: '+df['TOT_EMP'].astype(int).astype(str)
        +'<br>Location quotient: '+df['LOC_QUOTIENT'].round(2).astype(str)
        +'<br>RPP-adjusted mean annual wage: '+df['A_MEAN'].astype(int).astype(str)
        +'<br>RPP-adjusted annual median wage: '+df['A_MEDIAN'].astype(int).astype(str)+'<extra></extra>'
    ))
  
    fig.update_layout(
        paper_bgcolor = '#333333',
    )
    return fig

def generate_figures(area):
    figs = []
    df = get_occupations(area)
    figs.append(map_occupations(df))

    df = df.query('O_GROUP == "detailed"')
    ranking = df[['OCC_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    fig = px.bar(ranking.head(10), x = 'AVERAGE_Z_SCORE', y = 'OCC_TITLE', height = 500)
    fig.update_layout(
        title = dict(
            text = 'Highest Average Z-Score',
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
            categoryarray = ranking.iloc[::-1]['OCC_TITLE'],
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
        hovertemplate = '%{y} <br>Average z-score: %{x}'
    )
    figs.append(fig)

    names = {'TOT_EMP':'Total Employment', 
            'LOC_QUOTIENT':'Location Quotient', 
            'A_MEAN': 'Mean Annual Wage',
            'A_MEDIAN': 'Annual Median Wage'}
    for statistic in names:
        ranking = df[['OCC_TITLE', statistic]].sort_values(statistic, ascending = False)
        fig = px.bar(ranking.head(5), x = statistic, y = 'OCC_TITLE', height = 300)
        fig.update_layout(
            title = dict(
                text = 'Highest '+names[statistic],
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
                categoryorder = 'total ascending',
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
            hovertemplate = 'Metropolitan area: %{y} <br>'+names[statistic].capitalize()+': %{x}'
        )
        figs.append(fig)
    return figs


import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(
    __name__,
    path='/best-occupations',
    title='Best Occupations')

@callback(
    Output('treemap', 'figure'),
    Output('overall2', 'figure'),
    Output('total-employment2', 'figure'),
    Output('location-quotient2', 'figure'),
    Output('mean', 'figure'),
    Output('median', 'figure'),
    Input('dropdown', 'value')
)
def update_output(value):
    return generate_figures(value)

figs = generate_figures('U.S.')
layout = html.Div([

    html.Div(
        className = 'dropdown', 
        children = [
            html.H1(
                className = 'text', 
                children = ['Best ', dcc.Link('Occupations', href='/best-metropolitan-areas'), ' in'] 
            ),
            dcc.Dropdown(occ_df['AREA_TITLE'].unique(), 'U.S.', id = 'dropdown')
        ]
    ),
    dcc.Graph(figure = figs[0], id = 'treemap'),
    dcc.Graph(figure = figs[1], id = 'overall2'),
    html.Div(
        children = [
            html.Div(
                children = [
                    dcc.Graph(figure = figs[2], id = 'total-employment2'),
                    dcc.Graph(figure = figs[4], id = 'mean')
                ],
                style = {'flex': 0.5}
            ),
            html.Div(
                children = [
                    dcc.Graph(figure = figs[3], id = 'location-quotient2'),
                    dcc.Graph(figure = figs[5], id = 'median')
                ],
                style = {'flex': 0.5}
            )
        ], 
        style = {'display': 'flex', 'flex-direction': 'row'}
    )
])