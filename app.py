import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


df_original = pd.read_csv("data/commercial_planning.csv", encoding="latin-1")

final_states = ["CLOSED", "COMPLETE", "CANCELED", "SUSPECTED_FRAUD"]
df_original = df_original[df_original["Order Status"].isin(final_states)]
df_original["Order Success"] = df_original["Order Status"].isin(["COMPLETE", "CLOSED"]).astype(int)

df_original["Discount Ratio"] = df_original["Order Item Discount"] / df_original["Product Price"]

df = df_original.copy()
df.drop(columns=["Order Status","Order Item Discount"], inplace=True)
df = df.reset_index(drop=True)

categorical_vars = ["Category Name", "Market", "Order Region", "Shipping Mode"]
numeric_vars = ["Days for shipment (scheduled)", "Product Price", "Discount Ratio"]

X_train, X_test, y_train, y_test = train_test_split(
    df[categorical_vars + numeric_vars],
    df["Order Success"],
    test_size=0.2,
    random_state=42,
    stratify=df["Order Success"]
)

smote = SMOTE(sampling_strategy=0.176, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_vars])

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[categorical_vars])

X_train_processed = np.hstack((X_train_cat, X_train_num))

knn_classifier = KNeighborsClassifier(n_neighbors=5)

bagging_knn = BaggingClassifier(
    estimator=knn_classifier,
    n_estimators=50,       
    max_samples=0.3,
    bootstrap=True,
    n_jobs=1             
)
bagging_knn.fit(X_train_processed, y_train)

pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(X_train_processed)

df_pca = pd.DataFrame(pca_results, columns=["PC1", "PC2"])
df_pca["Order Success"] = df["Order Success"].values

success = df_pca.loc[df_pca["Order Success"] == 1,:]
fails = df_pca.loc[df_pca["Order Success"] == 0,:]

probability_text = html.B(id="probability",children=[],style={})

fig_pca = go.Figure()
fig_pca.add_trace(go.Scatter(x=success["PC1"], y=success["PC2"], mode="markers", marker_color="green", name="Completadas"))
fig_pca.add_trace(go.Scatter(x=fails["PC1"], y=fails["PC2"], mode="markers", marker_color="red", name="Sin éxito"))
fig_pca.update_layout(title="Resultados de órdenes históricas")
fig_pca.update_layout(legend=dict(font=dict(size=9)))

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body",className="e4_body",children=[
    html.A(href="https://github.com/genagithub/proyecto-4/blob/main/README.md",children=[html.H1("Evaluación de riesgo en planificaciones comerciales",id="title",className="e4_title")]),
    html.Div(id="dashboard", className="e4_dashboard", children=[
        html.Div(className="e4_graph_div",children=[
            dcc.Graph(id="graph_pca",className="e4_graph",figure=fig_famd),
            html.Div(id="input_div", style={"display":"flex","flexWrap":"wrap","gap":"10px"}, children=[
                dcc.Input(id="input_1", type="number", placeholder="Días envío", style={"width":"75px"}),
                dcc.Input(id="input_5", type="number", placeholder="Precio Producto", style={"width":"75px"}),
                dcc.Input(id="input_6", type="number", placeholder="Ratio Descuento", style={"width":"75px"}),
                dcc.Dropdown(id="input_2", options=[{"label": i, "value": i} for i in df_original["Market"].dropna().unique()], placeholder="Mercado", style={"width":"150px"}),
                dcc.Dropdown(id="input_3", options=[{"label": i, "value": i} for i in df_original["Order Region"].dropna().unique()], placeholder="Región", style={"width":"150px"}),
                dcc.Dropdown(id="input_4", options=[{"label": i, "value": i} for i in df_original["Category Name"].dropna().unique()], placeholder="Categoría", style={"width":"150px"}),
                dcc.Dropdown(id="input_7", options=[{"label": i, "value": i} for i in df_original["Shipping Mode"].dropna().unique()], placeholder="Tipo Envío", style={"width":"150px"}),
                html.Button(id="button", className="e4_button", children="Enviar", n_clicks=0)
            ]),
            html.P(["predicción: riesgo de fracaso del ",probability_text],className="e4_predict")
        ])
    ])
])


@app.callback(
    [Output(component_id="graph_pca",component_property="figure"),
    Output(component_id="probability",component_property="children"),
    Output(component_id="probability",component_property="style")],
    [Input(component_id="button",component_property="n_clicks")],
    [State(component_id="input_1",component_property="value"),
    State(component_id="input_2",component_property="value"),
    State(component_id="input_3",component_property="value"),
    State(component_id="input_4",component_property="value"),
    State(component_id="input_5",component_property="value"),
    State(component_id="input_6",component_property="value"),
    State(component_id="input_7",component_property="value")]
)

def get_risk_prob(n_clicks, var_1, var_2, var_3, var_4, var_5, var_6, var_7):
  
    fig_update = go.Figure(fig_pca)
    prob_fail_text = "0.00%"
    style_res = {"color": "black"}

    inputs = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    if n_clicks > 0 and all(v is not None for v in inputs):
        new_object = pd.DataFrame({
            "Days for shipment (scheduled)": [float(var_1)],
            "Market": [str(var_2)],
            "Order Region": [str(var_3)],
            "Category Name": [str(var_4)],
            "Product Price": [float(var_5)],
            "Discount Ratio": [float(var_6)],
            "Shipping Mode": [str(var_7)]
        })

        obj_num_scaled = scaler.transform(new_object[numeric_vars])
        obj_cat_enc = encoder.transform(new_object[categorical_vars])

        object_processed = np.hstack((obj_cat_enc, obj_num_scaled))

        prob_fail = bagging_knn.predict_proba(object_processed)[0, 0] * 100 
        prob_fail_text = f"{prob_fail:.2f}%"

        if prob_fail <= 45:
            factor = prob_fail / 45
            hue = int(120 - (factor * 120))
            color_res = f"hsl({hue}, 100%, 45%)"
        elif 45 < prob_fail <= 50:
            color_res = "hsl(0, 100%, 45%)"
        else:
            dark_factor = (prob_fail - 50) / 50
            lightness = int(45 - (dark_factor * 25))
            color_res = f"hsl(0, 100%, {lightness}%)"

            style_res = {"color":color_res}

        obj_pca = pca.transform(object_processed)

        fig_update.add_trace(go.Scatter(
            x=[obj_pca[0, 0]],
            y=[obj_pca[0, 1]],
            mode="markers",
            marker=dict(color="blueviolet", size=15, symbol="star"),
            name="Nuevo producto"
        ))

    return fig_pca, prob_fail_text, style_res


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port)
