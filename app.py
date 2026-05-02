import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


df_original = pd.read_csv("data/smart_supply_chain.csv", encoding="latin-1")

final_states = ["CLOSED", "COMPLETE", "CANCELED", "SUSPECTED_FRAUD"]
df_original = df_original[df_original["Order Status"].isin(final_states)]
df_original["Order Success"] = df_original["Order Status"].isin(["COMPLETE", "CLOSED"]).astype(int)

df_original["Discount Ratio"] = df_original["Order Item Discount"] / df_original["Product Price"]

df = df_original.copy()
df.drop(columns=["Order Status","Order Item Discount"], inplace=True)
df = df.reset_index(drop=True)

categorical_vars = ["Category Name", "Market", "Order Region", "Shipping Mode"]
numeric_vars = ["Days for shipment (scheduled)", "Product Price", "Discount Ratio"]

scaler = StandardScaler()
df[numeric_vars] = scaler.fit_transform(df[numeric_vars])

encoder = OrdinalEncoder()
df[categorical_vars] = encoder.fit_transform(df[categorical_vars])

column = df.pop("Order Success")
df.insert(0, "Order Success", column)

X_train_columns = df.columns[1:] 

knn_classifier = KNeighborsClassifier(n_neighbors=5)

bagging_knn = BaggingClassifier(estimator=knn_classifier,
                                n_estimators=100,
                                max_samples=0.3,
                                bootstrap=True,
                                n_jobs=1)

bagging_knn.fit(df[X_train_columns], df["Order Success"])

pca = PCA(n_components=2)
pca_results = pca.fit_transform(df[numeric_vars])

df_pca = pd.DataFrame(pca_results, columns=["PC1", "PC2"])
df_pca["Order Success"] = df["Order Success"].values

success = df_pca.loc[df_pca["Order Success"] == 1,:]
fails = df_pca.loc[df_pca["Order Success"] == 0,:]

df_value_counts = df_original["Order Success"].value_counts(normalize=True)
success_prc, fails_prc = round(df_value_counts.loc[1]*100,1), round(df_value_counts.loc[0]*100,1)

probability_text = html.B(id="probability", children=[], style={})

fig_pca = go.Figure()
fig_pca.add_trace(go.Scatter(x=success["PC1"], y=success["PC2"], mode="markers", marker_color="green", name=f"Completadas ({success_prc}%)"))
fig_pca.add_trace(go.Scatter(x=fails["PC1"], y=fails["PC2"], mode="markers", marker_color="red", name=f"Sin éxito ({fails_prc}%)"))
fig_pca.update_layout(title="Resultados de órdenes históricas")
fig_pca.update_layout(legend=dict(font=dict(size=9)))

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body",className="e4_body",children=[
    html.H1("Evaluación de riesgo en ventas planificadas",id="title",className="e4_title"),
    html.Div(id="dashboard",className="e4_dashboard",children=[
        html.Div(className="e4_graph_div",children=[
            dcc.Graph(id="graph_pca",className="e4_graph",figure=fig_pca),
            html.Form(id="input_div",className="input_div",children=[
                dcc.Input(id="input_1",className="input",type="text",placeholder="Días de envío (esquema)",size="lg"),
                dcc.Input(id="input_2",className="input",type="text",placeholder="Mercado objetivo",size="lg"),
                dcc.Input(id="input_3",className="input",type="text",placeholder="Región específica",size="lg"),
                dcc.Input(id="input_4",className="input",type="text",placeholder="Categoría asignada",size="lg"),
                dcc.Input(id="input_5",className="input",type="text",placeholder="Precio del producto",size="lg"),
                dcc.Input(id="input_6",className="input",type="text",placeholder="Ratio del descuento",size="lg"),
                dcc.Input(id="input_7",className="input",type="text",placeholder="Tipo de envío",size="lg"),
                html.Button(id="button",className="button",children="Enviar",n_clicks=0)
            ]),
            html.P(["predicción: riesgo de fracaso del ",probability_text,"%"],className="e4_predict")
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
    
    if n_clicks is not None and n_clicks > 0:
        new_object = pd.DataFrame({
            "Days for shipment (scheduled)": [int(var_1)],
            "Market": [var_2],
            "Order Region": [var_3],
            "Category Name": [var_4],
            "Product Price": [float(var_5)],
            "Discount Ratio": [float(var_6)],
            "Shipping Mode": [var_7]
        })

        obj_num_scaled = scaler.transform(new_object[numeric_vars])
        df_obj_num_scaled = pd.DataFrame(obj_num_scaled, columns=numeric_vars)

        obj_pca = pca.transform(obj_num_scaled)
        df_obj_pca = pd.DataFrame(obj_pca, columns=["PC1", "PC2"])
      
        df_obj_cat = pd.DataFrame(encoder.transform(new_object[categorical_vars]), columns=categorical_vars)

        object_to_predict = pd.concat([df_obj_num_scaled, df_obj_cat], axis=1)[X_train_columns]
      
        prob_fail = bagging_knn.predict_proba(object_to_predict)[0, 0] * 100
        color_res = "red" if prob_fail > 45 else "green"

        fig_update.add_trace(go.Scatter(x=df_obj_pca["PC1"], y=df_obj_pca["PC2"], mode="markers", marker=dict(color="blueviolet", size=15, symbol="star"), name="Nuevo producto"))

    return fig_update, f"{prob_fail:.2f}", {"color": color_res}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port)
