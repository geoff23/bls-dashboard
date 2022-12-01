import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import numpy as np
import plotly.graph_objects as go

msa_df = pd.read_csv('msa.csv')

def get_metropolitan_areas(occupation, msa_df = msa_df):
    df = msa_df.query('OCC_TITLE == "{}"'.format(occupation))
    df = df.drop('OCC_TITLE', axis=1)
    
    column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'A_MEAN', 'A_MEDIAN', 'RPP_ADJUSTED_A_MEAN', 'RPP_ADJUSTED_A_MEDIAN']
    for column_name in column_names:
        df[column_name+'_Z_SCORE'] = (df[column_name] - df[column_name].mean())/df[column_name].std()
    df['AVERAGE_Z_SCORE'] = df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis=1)
    for column_name in column_names:
        df = df.drop(column_name+'_Z_SCORE', axis=1)
    return df

@ticker.FuncFormatter
def kmbt_format(num, pos):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def visualize_metropolitan_areas(occupation):
    names = {'TOT_EMP':'Total Employment', 
             'LOC_QUOTIENT':'Location Quotient', 
             'A_MEAN':'Mean Annual Wage',
             'A_MEDIAN':'Annual Median Wage',
             'RPP_ADJUSTED_A_MEAN': 'RPP-Adjusted Mean Annual Wage',
             'RPP_ADJUSTED_A_MEDIAN': 'RPP-Adjusted Annual Median Wage'}
    df = get_metropolitan_areas(occupation)
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
    sns.barplot(x='AVERAGE_Z_SCORE', y='AREA_TITLE', data=ranking.head(10), ax = axis, palette = sns.color_palette('Blues_r', 20))
    sns.despine(left=True, bottom=True)
    axis.set(title = 'Best Metropolitan Areas Overall', xlabel='Average Z-Score', ylabel='')
    
    fig.suptitle('Best Metropolitan Areas For '+occupation)
    plt.show()

def map_metropolitan_areas(df):
    df = df.dropna()
    fig = go.Figure(data=go.Scattergeo(
            lon = df['LONGITUDE'],
            lat = df['LATTITUDE'],
            hoverinfo='text',
            text = df['AREA_TITLE'],
            marker = dict(
                sizemode = 'area',
                size = df['TOT_EMP']**0.5*4,
                color = df['AVERAGE_Z_SCORE'],
                colorscale = 'viridis'
                ),
            ))
    fig.update_layout(geo_scope='usa')
    return fig

from dash import Dash, html, dcc, Input, Output
import plotly.express as px

app = Dash(__name__)

def generate_figures(occupation):
    figs = []
    df = get_metropolitan_areas(occupation)
    
    figs.append(map_metropolitan_areas(df))
    ranking = df[['AREA_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    fig = px.bar(ranking.head(10), x='AVERAGE_Z_SCORE', y='AREA_TITLE', height = 500)
    fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray':ranking.iloc[::-1]['AREA_TITLE']}, title='Highest Average Z-Score', title_x = 0.5, xaxis_title='', yaxis_title = '')
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
        fig = px.bar(ranking.head(5), x=statistic, y='AREA_TITLE', height = 300)
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title='Highest '+names[statistic], title_x = 0.5, xaxis_title='', yaxis_title = '')
        fig.update_traces(hovertemplate='Metropolitan area: %{y} <br>'+names[statistic].capitalize()+': %{x}')
        fig.update_layout(
            xaxis = dict(ticks = 'outside', tickcolor='white', ticklen=5),
            yaxis = dict(ticks = 'outside', tickcolor='white', ticklen=10)
        )
        figs.append(fig)
    return figs

@app.callback(
    Output('map', 'figure'),
    Output('overall', 'figure'),
    Output('total-employment', 'figure'),
    Output('mean', 'figure'),
    Output('mean-adjusted', 'figure'),
    Output('location-quotient', 'figure'),
    Output('median', 'figure'),
    Output('median-adjusted', 'figure'),
    Input('dropdown', 'value')
)
def update_output(value):
    figs = generate_figures(value)
    return figs[0], figs[1], figs[2], figs[4], figs[6], figs[3], figs[5], figs[7]

figs = generate_figures('Data Scientists')
app.layout = html.Div([
    html.Div([
        html.Div(children=[
            html.H1(children = 'Best Metropolitan Areas for', style = {'font-family':'arial', 'margin-right':'0.67em', 'font-weight': 'normal', 'font-size': 16}),
            dcc.Dropdown(msa_df['OCC_TITLE'].unique(), 'Data Scientists', style = {'font-family':'arial', 'flex': 1}, id='dropdown')
        ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-bottom': 50}),
        dcc.Graph(figure=figs[0], id='map'),
        dcc.Graph(figure=figs[1], id='overall'),
        html.Div([
            html.Div(children=[
                dcc.Graph(figure=figs[2], id='total-employment'),
                dcc.Graph(figure=figs[4], id='mean'),
                dcc.Graph(figure=figs[6], id='mean-adjusted')
            ], style={'flex': 0.5}),
            html.Div(children=[
                dcc.Graph(figure=figs[3], id='location-quotient'),
                dcc.Graph(figure=figs[5], id='median'),
                dcc.Graph(figure=figs[7], id='median-adjusted')
            ], style={'flex': 0.5}) 
        ], style={'display': 'flex', 'flex-direction': 'row'})
    ], style = {'margin-top':100, 'margin-bottom':100, 'margin-left': 200, 'margin-right': 200})
])

if __name__ == '__main__':
    app.run_server(debug=True)