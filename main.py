import pandas as pd
import plotly.graph_objects as go

msa_df = pd.read_csv('msa.csv')
msa_df = msa_df.dropna()

def get_metropolitan_areas(occupation, msa_df = msa_df):
    df = msa_df.query('OCC_TITLE == "{}"'.format(occupation))
    df = df.drop('OCC_TITLE', axis = 1)
    
    column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'RPP_ADJUSTED_A_MEAN', 'RPP_ADJUSTED_A_MEDIAN']
    for column_name in column_names:
        df[column_name+'_Z_SCORE'] = (df[column_name] - df[column_name].mean())/df[column_name].std()
    df['AVERAGE_Z_SCORE'] = df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis = 1)
    for column_name in column_names:
        df = df.drop(column_name+'_Z_SCORE', axis=1)
    return df

def map_metropolitan_areas(df):
    fig = go.Figure(data = go.Scattergeo(
            lon = df['LONGITUDE'],
            lat = df['LATITUDE'],
            hoverinfo = 'text',
            text = df['AREA_TITLE']
            +'<br>Average z-score: '+df['AVERAGE_Z_SCORE'].round(2).astype(str)
            +'<br>Total employment: '+df['TOT_EMP'].astype(int).astype(str)
            +'<br>Location quotient: '+df['LOC_QUOTIENT'].round(2).astype(str)
            +'<br>RPP-adjusted mean annual wage: '+df['RPP_ADJUSTED_A_MEAN'].astype(int).astype(str)
            +'<br>RPP-adjusted annual median wage: '+df['RPP_ADJUSTED_A_MEDIAN'].astype(int).astype(str),
            
            marker = dict(
                sizemode = 'area',
                size = df['TOT_EMP']*0.01,
                color = df['AVERAGE_Z_SCORE'],
                colorscale = 'viridis',
                line_color = '#cccccc',
                colorbar = dict(
                    title = dict(text = 'Average Z-Score', font = dict(color = '#cccccc')),
                    tickfont = dict(color = '#cccccc')
                )
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


from dash import Dash, html, dcc, Input, Output
import plotly.express as px

app = Dash(__name__)

def generate_figures(occupation):
    figs = []
    df = get_metropolitan_areas(occupation)
    
    figs.append(map_metropolitan_areas(df))
    
    ranking = df[['AREA_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    fig = px.bar(ranking.head(10), x = 'AVERAGE_Z_SCORE', y = 'AREA_TITLE', height = 500)
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
        hovertemplate = '%{y} <br>Average z-score: %{x}'
    )
    figs.append(fig)

    names = {'TOT_EMP':'Total Employment', 
            'LOC_QUOTIENT':'Location Quotient', 
            'RPP_ADJUSTED_A_MEAN': 'RPP-Adjusted Mean Annual Wage',
            'RPP_ADJUSTED_A_MEDIAN': 'RPP-Adjusted Annual Median Wage'}
    for statistic in names:
        ranking = df[['AREA_TITLE', statistic]].sort_values(statistic, ascending = False)
        fig = px.bar(ranking.head(5), x = statistic, y = 'AREA_TITLE', height = 300)
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

@app.callback(
    Output('map', 'figure'),
    Output('overall', 'figure'),
    Output('total-employment', 'figure'),
    Output('location-quotient', 'figure'),
    Output('mean-adjusted', 'figure'),
    Output('median-adjusted', 'figure'),
    Input('dropdown', 'value')
)
def update_output(value):
    return generate_figures(value)

figs = generate_figures('Data Scientists')
app.layout = html.Div([
    html.Div(
        className = 'dropdown', 
        children = [
            html.H1(
                className = 'text', 
                children = 'Best Metropolitan Areas for'
            ),
            dcc.Dropdown(msa_df['OCC_TITLE'].unique(), 'Data Scientists', id = 'dropdown')
        ]
    ),
    dcc.Graph(figure = figs[0], id = 'map'),
    dcc.Graph(figure = figs[1], id = 'overall'),
    html.Div(
        children = [
            html.Div(
                children = [
                    dcc.Graph(figure = figs[2], id = 'total-employment'),
                    dcc.Graph(figure = figs[4], id = 'mean-adjusted')
                ],
                style = {'flex': 0.5}
            ),
            html.Div(
                children = [
                    dcc.Graph(figure = figs[3], id = 'location-quotient'),
                    dcc.Graph(figure = figs[5], id = 'median-adjusted')
                ],
                style = {'flex': 0.5}
            )
        ], 
        style = {'display': 'flex', 'flex-direction': 'row'}
    )
])

if __name__ == '__main__':
    app.run_server(debug = True)
'''
def get_occupations(metropolitan_area, msa_df = msa_df):
    df = msa_df.query('AREA_TITLE == "{}"'.format(metropolitan_area)).query('O_GROUP == "detailed"')
    df = df.drop('AREA_TITLE', axis=1)

    column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'RPP_ADJUSTED_A_MEAN', 'RPP_ADJUSTED_A_MEDIAN']
    for column_name in column_names:
        df[column_name+'_Z_SCORE'] = (df[column_name] - df[column_name].mean())/df[column_name].std()
    df['AVERAGE_Z_SCORE'] = df[[column_name+'_Z_SCORE' for column_name in column_names]].mean(axis=1)
    for column_name in column_names:
        df = df.drop(column_name+'_Z_SCORE', axis=1)
    return df

df = get_occupations('New York-Newark-Jersey City, NY-NJ-PA')
print(df.isna().sum())
df = df.dropna()
df.to_csv('ASD.csv', index = False)'''