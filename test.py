import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/96c0bd/sunburst-coffee-flavors-complete.csv')
df.to_csv('ASDF.csv', index = False)
print(df)
fig = go.Figure(go.Treemap(
    ids = df.ids,
    labels = df.labels,
    parents = df.parents,
    pathbar_textfont_size=15,
    root_color="lightgrey"
))
fig.update_layout(
    uniformtext=dict(minsize=10, mode='hide'),
    margin = dict(t=50, l=25, r=25, b=25)
)
fig.show()