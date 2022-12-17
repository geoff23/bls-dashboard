import plotly.express as px
 
df = px.data.tips()
print(df)
fig = px.treemap(df, path=['day', 'time', 'tip'],
                 values='total_bill')
 
fig.show()