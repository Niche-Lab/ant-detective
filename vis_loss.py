import os
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# CONSTANTS
ROOT = os.path.join("/Users/niche/OneDrive - Virginia Tech/_03_Papers", "find_ants")
MODELNAME = "pred_0.401.csv"
DIR_PRED = os.path.join(ROOT, "out", MODELNAME)
DIR_IMGS = os.path.join(ROOT, "data", "peptone_sucrose", "test")
JITTER = 0.1
PAD = 50
os.chdir(ROOT)

# predictions
df_pred = pd.read_csv(DIR_PRED)
df_pred["p_sucrose"] = df_pred["pred_0"].round().astype(int)
df_pred["p_peptone"] = df_pred["pred_1"].round().astype(int)
df_pred["p_total"] = df_pred["p_sucrose"] + df_pred["p_peptone"]
df_pred["filename"] = df_pred["filename"].apply(lambda x: os.path.join(DIR_IMGS, x))
df_pred["err_su"] = abs(df_pred["p_sucrose"] - df_pred["sucrose"])
df_pred["err_pep"] = abs(df_pred["p_peptone"] - df_pred["peptone"])
df_pred["err_total"] = abs(df_pred["p_total"] - df_pred["total_foragers"])
df_pred = df_pred.loc[
    :,
    [
        "filename",
        "sucrose",
        "p_sucrose",
        "peptone",
        "p_peptone",
        "total_foragers",
        "p_total",
        "err_su",
        "err_pep",
        "err_total",
    ],
]
df_pred["name"] = df_pred["filename"].apply(lambda x: os.path.basename(x))

correlation = df_pred["total_foragers"].corr(df_pred["p_total"])
mae = df_pred["err_total"].mean()


fig = go.Figure(
    data=[
        go.Scatter(
            x=df_pred["total_foragers"] + JITTER * np.random.randn(len(df_pred)),
            y=df_pred["p_total"] + JITTER * np.random.randn(len(df_pred)),
            mode="markers",
            marker=dict(
                colorscale="sunset",
                color=df_pred["err_total"],
                colorbar={"title": "MAE"},
                line={"color": "#444"},
                reversescale=True,
                # change the marker size
                sizeref=45,
                size=20,
                sizemode="diameter",
                opacity=0.8,
            ),
        )
    ],
    # add grid lines
    layout=go.Layout(
        title=f"Total Foragers: r = {correlation:.2f}, MAE = {mae:.2f}",
        height=900,
        template="plotly_dark",
        dragmode="pan",
        margin=dict(l=PAD, r=PAD, b=PAD, t=PAD + 100, pad=10),
        xaxis=dict(
            title="Total Foragers",
            # range
            range=[0, 20],
            showgrid=True,
            dtick=2,
            zeroline=False,
            # showline=False,
            showticklabels=True,
            gridcolor="rgb(255, 255, 255)",
            gridwidth=2,
        ),
        yaxis=dict(
            title="Predicted Total Foragers",
            range=[0, 20],
            showgrid=True,
            dtick=2,
            # showline=False,
            # showticklabels=True,
            # gridcolor="rgb(255, 255, 255)",
            gridwidth=2,
        ),
        # change the background color
        # plot_bgcolor="rgba(20, 20, 20, .5)",
        # change the font
        font=dict(family="Courier New, monospace", size=18),
    ),
)

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

app = Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ]
)


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df_pred.iloc[num]
    img_src = app.get_asset_url(df_row["name"])

    children = [
        html.Div(
            [
                html.Img(src=img_src, style={"width": "200%"}),
                html.P(f"{img_src}", style={"color": "darkblue"}),
                # create a table
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td("Cateogries"),
                                html.Td("Truth"),
                                html.Td("Pred"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Sucrose"),
                                html.Td(df_row["sucrose"]),
                                html.Td(df_row["p_sucrose"]),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Peptone"),
                                html.Td(df_row["peptone"]),
                                html.Td(df_row["p_peptone"]),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Total Foragers"),
                                html.Td(df_row["total_foragers"]),
                                html.Td(df_row["p_total"]),
                            ]
                        ),
                    ],
                    style={"width": "100%"},
                ),
            ],
            style={"width": "300px", "white-space": "normal"},
        )
    ]

    return True, bbox, children


if __name__ == "__main__":
    # export as html
    app.run_server(debug=True)
