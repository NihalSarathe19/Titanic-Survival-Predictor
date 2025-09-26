import pandas as pd
import plotly.express as px

# Load the data from JSON
data = [{"Age": 29.97, "Fare": 24.32, "Survived": 0}, {"Age": 27.92, "Fare": 31.62, "Survived": 0}, {"Age": 17.70, "Fare": 21.88, "Survived": 1}, {"Age": 10.22, "Fare": 22.85, "Survived": 0}, {"Age": 5.59, "Fare": 0.00, "Survived": 1}, {"Age": 27.50, "Fare": 15.75, "Survived": 0}, {"Age": 23.31, "Fare": 45.67, "Survived": 1}, {"Age": 21.89, "Fare": 8.90, "Survived": 0}, {"Age": 43.23, "Fare": 89.45, "Survived": 1}, {"Age": 34.56, "Fare": 12.34, "Survived": 0}, {"Age": 28.45, "Fare": 78.23, "Survived": 1}, {"Age": 32.11, "Fare": 19.67, "Survived": 0}, {"Age": 45.67, "Fare": 134.56, "Survived": 1}, {"Age": 12.34, "Fare": 7.89, "Survived": 0}, {"Age": 56.78, "Fare": 156.78, "Survived": 1}, {"Age": 41.23, "Fare": 25.67, "Survived": 0}, {"Age": 25.89, "Fare": 67.89, "Survived": 1}, {"Age": 39.45, "Fare": 18.23, "Survived": 0}, {"Age": 31.67, "Fare": 89.34, "Survived": 1}, {"Age": 22.11, "Fare": 11.45, "Survived": 0}, {"Age": 48.33, "Fare": 178.90, "Survived": 1}, {"Age": 26.78, "Fare": 14.56, "Survived": 0}, {"Age": 35.44, "Fare": 98.76, "Survived": 1}, {"Age": 19.56, "Fare": 9.87, "Survived": 0}, {"Age": 42.89, "Fare": 145.67, "Survived": 1}, {"Age": 33.22, "Fare": 20.34, "Survived": 0}, {"Age": 29.11, "Fare": 56.78, "Survived": 1}, {"Age": 37.88, "Fare": 16.89, "Survived": 0}, {"Age": 24.55, "Fare": 45.23, "Survived": 1}, {"Age": 44.22, "Fare": 13.45, "Survived": 0}]

df = pd.DataFrame(data)

# Create survival status labels for legend
df['Survival'] = df['Survived'].map({0: 'No', 1: 'Yes'})

# Create scatter plot
fig = px.scatter(df, x='Age', y='Fare', color='Survival',
                title="Age vs Fare by Survival",
                labels={'Age': 'Age', 'Fare': 'Fare', 'Survival': 'Survived'},
                color_discrete_map={'No': '#DB4545', 'Yes': '#1FB8CD'})

# Update layout and traces
fig.update_traces(cliponaxis=False)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))

# Save as both PNG and SVG
fig.write_image("titanic_scatter.png")
fig.write_image("titanic_scatter.svg", format="svg")