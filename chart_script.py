import pandas as pd
import plotly.express as px
import json

# Load the JSON data
data = [{"Age": 29.97, "Survived": 0}, {"Age": 27.92, "Survived": 0}, {"Age": 17.70, "Survived": 1}, {"Age": 10.22, "Survived": 0}, {"Age": 5.59, "Survived": 1}, {"Age": 27.50, "Survived": 0}, {"Age": 23.31, "Survived": 1}, {"Age": 21.89, "Survived": 0}, {"Age": 43.23, "Survived": 1}, {"Age": 34.56, "Survived": 0}, {"Age": 28.45, "Survived": 1}, {"Age": 32.11, "Survived": 0}, {"Age": 45.67, "Survived": 1}, {"Age": 12.34, "Survived": 0}, {"Age": 56.78, "Survived": 1}, {"Age": 41.23, "Survived": 0}, {"Age": 25.89, "Survived": 1}, {"Age": 39.45, "Survived": 0}, {"Age": 31.67, "Survived": 1}, {"Age": 22.11, "Survived": 0}, {"Age": 48.33, "Survived": 1}, {"Age": 26.78, "Survived": 0}, {"Age": 35.44, "Survived": 1}, {"Age": 19.56, "Survived": 0}, {"Age": 42.89, "Survived": 1}, {"Age": 33.22, "Survived": 0}, {"Age": 29.11, "Survived": 1}, {"Age": 37.88, "Survived": 0}, {"Age": 24.55, "Survived": 1}, {"Age": 44.22, "Survived": 0}]

# Create DataFrame
df = pd.DataFrame(data)

# Create survival status labels
df['Status'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})

# Create the histogram
fig = px.histogram(
    df, 
    x='Age', 
    color='Status',
    nbins=10,
    title='Titanic Age Distribution by Survival',
    labels={'Age': 'Age', 'count': 'Count'},
    color_discrete_map={
        'Not Survived': '#DB4545',  # Bright red
        'Survived': '#1FB8CD'       # Strong cyan
    }
)

# Update layout
fig.update_layout(
    xaxis_title='Age',
    yaxis_title='Count',
    bargap=0.1
)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Center the legend under the title (2 legend items, so <= 5)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))

# Save as PNG and SVG
fig.write_image("titanic_histogram.png")
fig.write_image("titanic_histogram.svg", format="svg")

fig.show()