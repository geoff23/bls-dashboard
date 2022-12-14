import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.gridspec import GridSpec

msa_df = pd.read_csv('msa.csv')

def get_metropolitan_areas(occupation, msa_df = msa_df):
    df = msa_df.query('OCC_TITLE == "{}"'.format(occupation))
    df = df.drop('OCC_TITLE', axis=1)
    
    column_names = ['TOT_EMP', 'LOC_QUOTIENT', 'RPP_ADJUSTED_A_MEAN', 'RPP_ADJUSTED_A_MEDIAN']
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
             'RPP_ADJUSTED_A_MEAN': 'RPP-Adjusted Mean Annual Wage',
             'RPP_ADJUSTED_A_MEDIAN': 'RPP-Adjusted Annual Median Wage'}
    df = get_metropolitan_areas(occupation)
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(4, 2, figure=fig)
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
        
    axis = fig.add_subplot(gs[2:, :])
    ranking = df[['AREA_TITLE', 'AVERAGE_Z_SCORE']].sort_values('AVERAGE_Z_SCORE', ascending = False)
    sns.barplot(x='AVERAGE_Z_SCORE', y='AREA_TITLE', data=ranking.head(10), ax = axis, palette = sns.color_palette('Blues_r', 20))
    sns.despine(left=True, bottom=True)
    axis.set(title = 'Best Metropolitan Areas Overall', xlabel='Average Z-Score', ylabel='')
    
    fig.suptitle('Best Metropolitan Areas For '+occupation)
    plt.show()

visualize_metropolitan_areas('Data Scientists')