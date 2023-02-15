import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

# CONSTANTS
ROOT = os.path.join("/", "home", "niche", "find_ants")
DIR_PRED = os.path.join(ROOT, "out", "pred_0.341.csv")
DIR_IMGS = os.path.join(ROOT, "data", "peptone_sucrose", "test")
os.chdir(ROOT)

# predictions
df_pred = pd.read_csv(DIR_PRED)
df_pred["p_sucrose"] = df_pred["pred_0"].round().astype(int)
df_pred["p_peptone"] = df_pred["pred_1"].round().astype(int)
df_pred["p_total"] = df_pred["p_sucrose"] + df_pred["p_peptone"]
df_pred["filename"] = df_pred["filename"].apply(lambda x: os.path.join(DIR_IMGS, x))
df_pred = df_pred.loc[:, ["filename", "sucrose", "p_sucrose", 
                          "peptone", "p_peptone", "total_foragers", "p_total"]]

# Create dash app
app = dash.Dash(__name__)


# Generate dataframe
df = pd.DataFrame(
   dict(
      x=df_pred["sucrose"],
      y=df_pred["p_sucrose"],
      images=df_pred["filename"],
   )
)

# Create scatter plot with x and y coordinates
fig = px.scatter(df, x="x", y="y", custom_data=["images"])

# Update layout and update traces
fig.update_layout(clickmode='event + select')
fig.update_traces(marker_size=20)

# Create app layout to show dash graph
app.layout = html.Div(
   [
      dcc.Graph(
         id="graph_interaction",
         figure=fig,
      ),
      html.Img(id='image', src='')
   ]
)

# html callback function to hover the data on specific coordinates
@app.callback(
   Output('image', 'src'),
   Input('graph_interaction', 'hoverData'))
def open_url(hoverData):
    if hoverData:
        return hoverData["points"][0]["customdata"][0]
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)