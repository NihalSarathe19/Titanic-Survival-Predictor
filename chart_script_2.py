import plotly.graph_objects as go
import numpy as np

# Parse the correlation matrix data
features = ["Survived", "Age", "Fare", "Sex_Male", "Pclass_1", "Pclass_2", "Pclass_3", "Family_Size", "Has_Cabin"]
correlation_data = [
    [1.000, -0.009, 0.238, -0.100, 0.329, 0.093, -0.345, 0.004, 0.039],
    [-0.009, 1.000, 0.089, 0.054, -0.067, -0.034, 0.078, -0.198, 0.098],
    [0.238, 0.089, 1.000, -0.182, 0.599, 0.202, -0.667, 0.156, 0.387],
    [-0.100, 0.054, -0.182, 1.000, -0.131, -0.078, 0.187, 0.067, -0.089],
    [0.329, -0.067, 0.599, -0.131, 1.000, -0.278, -0.627, 0.089, 0.298],
    [0.093, -0.034, 0.202, -0.078, -0.278, 1.000, -0.584, 0.067, 0.134],
    [-0.345, 0.078, -0.667, 0.187, -0.627, -0.584, 1.000, -0.123, -0.378],
    [0.004, -0.198, 0.156, 0.067, 0.089, 0.067, -0.123, 1.000, 0.045],
    [0.039, 0.098, 0.387, -0.089, 0.298, 0.134, -0.378, 0.045, 1.000]
]

# Convert to numpy array for easier handling
corr_matrix = np.array(correlation_data)

# Create text annotations for each cell
text_annotations = []
for i in range(len(features)):
    row_text = []
    for j in range(len(features)):
        row_text.append(f"{corr_matrix[i,j]:.3f}")
    text_annotations.append(row_text)

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix,
    x=features,
    y=features,
    colorscale='RdBu_r',  # Red-Blue reversed (blue for negative, red for positive)
    zmid=0,  # Center the colorscale at 0
    text=text_annotations,
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False,
    colorbar=dict(
        title="Correlation"
    )
))

# Update layout
fig.update_layout(
    title="Titanic Feature Correlations",
    xaxis_title="Features",
    yaxis_title="Features"
)

# Update axes to prevent text rotation
fig.update_xaxes(tickangle=45)
fig.update_yaxes(tickangle=0)

# Save as PNG and SVG
fig.write_image("correlation_heatmap.png")
fig.write_image("correlation_heatmap.svg", format="svg")

print("Correlation heatmap created and saved!")