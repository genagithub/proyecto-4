import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


df_original = pd.read_csv("data/credit_risk.csv")
df_original.dropna(subset=["person_emp_length","loan_int_rate"], inplace=True)

df_original = df_original.loc[(df_original["person_age"] < 100) | (df_original["person_emp_length"] < 100),:]

df = df_original.copy()
df.reset_index(drop=True, inplace=True)

X = df.drop(["loan_status","loan_intent","loan_grade","person_home_ownership","cb_person_default_on_file"], axis=1)
y = df["loan_status"] 

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca = PCA(n_components=2)

# reduciendo los datos estandarizados a 2 dimensiones
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

columns_pca = ["PC1_numeric","PC2_numeric"]
df_pca = pd.DataFrame(X_pca, columns=columns_pca)

categorical_cols = ["loan_intent","loan_grade","person_home_ownership","cb_person_default_on_file"]

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
one_hot_encoder.fit(df[categorical_cols])
cols_encoded = one_hot_encoder.transform(df[categorical_cols])

new_cols_names = one_hot_encoder.get_feature_names_out(categorical_cols)
df_one_hot = pd.DataFrame(cols_encoded, columns=new_cols_names)

df_processed = pd.concat([df_one_hot, df_pca, df["loan_status"]], axis=1)

knn_classifier = KNeighborsClassifier(n_neighbors=5)

bagging_knn = BaggingClassifier(estimator=knn_classifier, # clasificador base
                                n_estimators=100, # cantidad de estimadores
                                max_samples=0.3,  # número de muestras requeridas para cada estimador
                                bootstrap=True) # muestreo con reemplazo

bagging_knn.fit(df_processed[df_processed.columns[:-1]], df["loan_status"])

object = df.sample(n=1)

object_categorical_cols = object[categorical_cols]
categorical_cols_encoded = one_hot_encoder.transform(object_categorical_cols)
categorical_cols_encoded = pd.DataFrame(categorical_cols_encoded, columns=new_cols_names)

categorical_cols.append("loan_status")
object_numeric_cols = object.drop(categorical_cols, axis=1)
numeric_cols_scaled = scaler.transform(object_numeric_cols)
numeric_cols_pca = pca.transform(numeric_cols_scaled)
numeric_cols_pca = pd.DataFrame(numeric_cols_pca, columns=columns_pca)

object_processed = pd.concat([categorical_cols_encoded, numeric_cols_pca], axis=1)

predict_encoded = bagging_knn.predict(object_processed)[0]

predict_proba = bagging_knn.predict_proba(object_processed)
probability = predict_proba[0, predict_encoded]*100

if predict_encoded == 0:
    probability = 100 - probability 

probability = str(probability)

df_processed = df_processed.loc[(df_processed["PC1_numeric"] < 15),:]

default = df_processed.loc[df_processed["loan_status"] == 1,:]
non_default = df_processed.loc[df_processed["loan_status"] == 0,:]

probability_text = html.B(id="probability",children=[],style={})
colors = ("green","red")

graph_pca = go.Figure()
graph_pca.add_trace(go.Scatter(x=default["PC1_numeric"], y=default["PC2_numeric"], mode="markers", marker_color="red", name="en default"))
graph_pca.add_trace(go.Scatter(x=non_default["PC1_numeric"], y=non_default["PC2_numeric"], mode="markers", marker_color="green", name="regularizado"))
graph_pca.update_layout(title="Crédito regularizado vs. en default")
graph_pca.update_layout(legend=dict(font=dict(size=9)))

app = dash.Dash(__name__)
server = app.server

app.layout =  html.Div(id="body",className="e4_body",children=[
    html.H1("Evaluación en riesgo de crédito",id="title",className="e4_title",href="https://github.com/genagithub/proyecto-4/blob/main/evaluaci%C3%B3n_sobre_riesgos_crediticios.ipynb",target="blank"),
    html.Div(id="dashboard",className="e4_dashboard",children=[
        html.Div(className="e4_graph_div",children=[
            dcc.Graph(id="graph_pca",className="e4_graph",figure=graph_pca),
            html.Form(id="input_div",className="input_div",children=[
                dcc.Input(id="input_1",className="input",type="text",placeholder="Edad",size="7"),
                dcc.Input(id="input_2",className="input",type="text",placeholder="Ingreso anual",size="7"),
                dcc.Input(id="input_3",className="input",type="text",placeholder="Tenencia de vivienda",size="7"),
                dcc.Input(id="input_4",className="input",type="text",placeholder="Longitud empleado",size="7"),
                dcc.Input(id="input_5",className="input",type="text",placeholder="Intención del préstamo",size="7"),
                dcc.Input(id="input_6",className="input",type="text",placeholder="Grado del préstamo",size="7"),
                dcc.Input(id="input_7",className="input",type="text",placeholder="Monto total",size="7"),
                dcc.Input(id="input_8",className="input",type="text",placeholder="Interés del préstamo",size="7"),
                dcc.Input(id="input_9",className="input",type="text",placeholder="Porcentage sobre el ingreso",size="7"),
                dcc.Input(id="input_10",className="input",type="text",placeholder="Historial de incumplimientos",size="7"),
                dcc.Input(id="input_11",className="input",type="text",placeholder="Historial de crédito(años)",size="7"),
                html.Button("enviar",id="button",type="button",className="input_button",n_clicks=0)
            ]),
            html.P(["predicción: riesgo de impago de ",probability_text,"%"],className="e4_predict")
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
    State(component_id="input_7",component_property="value"),
    State(component_id="input_8",component_property="value"),
    State(component_id="input_9",component_property="value"),
    State(component_id="input_10",component_property="value"),
    State(component_id="input_11",component_property="value")]
)

def update_graph(n_clicks, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11):
    if n_clicks is not None and n_clicks > 0:
        
        object = pd.DataFrame({
            "person_age":[var_1],
            "person_income":[var_2],
            "person_home_ownership":[var_3],
            "person_emp_length":[var_4],
            "loan_intent":[var_5],
            "loan_grade":[var_6],
            "loan_amnt":[var_7],
            "loan_int_rate":[var_8],
            "loan_percent_income":[var_9],
            "cb_person_default_on_file":[var_10],
            "cb_person_cred_hist_length":[var_11]
        })
           
        categorical_cols = ["loan_intent","loan_grade","person_home_ownership","cb_person_default_on_file"]   
        object_categorical_cols = object[categorical_cols]
        categorical_cols_encoded = one_hot_encoder.transform(object_categorical_cols)
        categorical_cols_encoded = pd.DataFrame(categorical_cols_encoded, columns=new_cols_names)

        object_numeric_cols = object.drop(categorical_cols, axis=1)
        numeric_cols_scaled = scaler.transform(object_numeric_cols)
        numeric_cols_pca = pca.transform(numeric_cols_scaled)
        numeric_cols_pca = pd.DataFrame(numeric_cols_pca, columns=columns_pca)

        object_processed = pd.concat([categorical_cols_encoded, numeric_cols_pca], axis=1)

        predict_encoded = bagging_knn.predict(object_processed)[0]
        predict_proba = bagging_knn.predict_proba(object_processed)
        probability = predict_proba[0, predict_encoded]*100
        
        if predict_encoded == 0:
            probability = 100 - probability
            if probability >= 45:
                predict_encoded = 1
        
        probability = str(probability)
        probability = probability[:4]
        probability_color = {"color":colors[predict_encoded]}
        
        graph_pca.add_trace(go.Scatter(x=object_processed["PC1_numeric"], y=object_processed["PC2_numeric"], mode="markers", marker_color="blueviolet", name="nuevo préstamo"))
    
    return graph_pca, probability, probability_color


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
