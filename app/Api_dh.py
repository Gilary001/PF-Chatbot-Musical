import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from src.Api_chatbot_engine import MúsicBotCUC


# Inicializar la App con un tema moderno (Bootstrap)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
bot = MúsicBotCUC()

# --- DISEÑO DE LA INTERFAZ (Layout) ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("HistoryBot - ChatBot", className="text-center my-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            # Área del Chat
            html.Div(id="chat-history", style={
                "height": "400px",
                "overflowY": "scroll",
                "border": "1px solid #ccc",
                "padding": "15px",
                "borderRadius": "10px",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "20px",
                }),

            # Entrada de texto
            dbc.InputGroup([
                dbc.Input(id="user-input", placeholder="Escribe tu pregunta sobre música...", type="text"),
                dbc.Button("Enviar", id="send-btn", color="primary", n_clicks=0),
            ]),

            # Indicador de carga
            dcc.Loading(id="loading", type="dot", children=html.Div(id="loading-output")),

        ], width=8, className="mx-auto")
    ])
], fluid=True)


# --- LÓGICA DE INTERACCIÓN (Callbacks) ---
@app.callback(
    [Output("chat-history", "children"),
     Output("user-input", "value")],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit")],
    [State("user-input", "value"),
     State("chat-history", "children")]
)
def actualizar_chat(n_clicks, n_submit, texto_usuario, historial_previo):
    # Si no hay texto o no se ha hecho clic, no hacer nada
    if not texto_usuario:
        return historial_previo, ""

    historial_previo = historial_previo or []

    # 1. Obtener respuesta del Bot
    respuesta_completa = bot.responder(texto_usuario)

    # 2. Formatear el mensaje del Usuario
    mensaje_usuario = html.Div([
        html.P(f"👤 Tú: {texto_usuario}", style={"fontWeight": "bold", "color": "#2c3e50"}),
    ], style={"textAlign": "right", "marginBottom": "10px"})

    # 3. Formatear el mensaje del Bot
    mensaje_bot = html.Div([
        html.P(f"🤖 Bot: {respuesta_completa}", style={"color": "#1a5276"}),
        html.Hr()
    ], style={"textAlign": "left", "marginBottom": "20px"})

    # Agregar a la lista del historial
    historial_previo.append(mensaje_usuario)
    historial_previo.append(mensaje_bot)

    return historial_previo, ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)