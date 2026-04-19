import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─── Path setup ───────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import app.config  # ← descomentar si tenés API key

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ─── Inicialización del RAG (Capa 1) ──────────────────────────
print("Inicializando RAG...")
from data.corpus_canciones import get_collection
from src.rag_utils import chunking_por_estrofa, cargar_modelo
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "notebooks")
PKL_PATH   = os.path.join(BASE, "emb_por_estrofa.pkl")
FAISS_PATH = os.path.join(BASE, "faiss_index_A.bin")

collection   = get_collection()
cursor       = collection.find({}, {"_id": 0, "artista": 1, "nombre": 1,
                                     "letra": 1, "genero": 1, "anio": 1, "titulo": 1})
df_canciones = pd.DataFrame(list(cursor))
datos        = df_canciones.to_dict("records")
fragmentos   = chunking_por_estrofa(datos)

modelo_emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Cargar índice FAISS desde archivo si existe, si no construirlo
if os.path.exists(FAISS_PATH):
    print(f"Cargando índice FAISS desde {FAISS_PATH}...")
    indice = faiss.read_index(FAISS_PATH)
    print(f"Índice FAISS cargado: {indice.ntotal} vectores")
else:
    print("Construyendo índice FAISS desde embeddings...")
    from src.rag_utils import generar_o_cargar_embeddings, crear_indice_faiss
    emb    = generar_o_cargar_embeddings(fragmentos, PKL_PATH.replace(".pkl", ""), modelo_emb)
    indice = crear_indice_faiss(emb)

cargar_modelo()
print("RAG listo.")

# ─── Inicialización del Chatbot (Capa 2 — Fine-Tuning + Motor) ─
print("Inicializando HistoryBot...")
from src.chatbot_engine import HistoryBotEngine

bot = HistoryBotEngine(
    indice_faiss=indice,
    chunks=fragmentos,
    modelo_emb=modelo_emb
)
print("HistoryBot listo.")

# ─── App Dash ─────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap"
    ],
    title="HistoryBot — Musical Historian"
)

COLORS = {
    "bg":       "#0D0D0D",
    "surface":  "#1A1A1A",
    "card":     "#222222",
    "accent":   "#C9A84C",
    "accent2":  "#8B5E3C",
    "text":     "#F0EAD6",
    "muted":    "#888888",
    "user_msg": "#2A2A2A",
    "bot_msg":  "#1E1E1E",
}

app.layout = html.Div([
    dcc.Store(id="store-historial", data=[]),

    html.Div([
        html.Div([
            html.Span("♪", style={"fontSize": "2rem", "color": COLORS["accent"], "marginRight": "12px"}),
            html.H1("HistoryBot", style={
                "fontFamily": "Playfair Display, serif",
                "color": COLORS["accent"],
                "fontSize": "2rem",
                "margin": 0,
                "letterSpacing": "2px"
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.P("Musical Historian · English Song Lyrics · 1990s–2020s", style={
            "fontFamily": "DM Sans, sans-serif",
            "color": COLORS["muted"],
            "margin": "4px 0 0 0",
            "fontSize": "0.85rem",
            "letterSpacing": "1px"
        })
    ], style={
        "backgroundColor": COLORS["surface"],
        "padding": "20px 32px",
        "borderBottom": f"1px solid {COLORS['accent2']}",
    }),

    html.Div([
        html.Div([
            html.Div(id="ventana-chat", children=[
                html.Div([
                    html.Span("♪", style={"fontSize": "1.5rem", "color": COLORS["accent"]}),
                    html.P(
                        "Hello! I'm HistoryBot, your musical historian. Ask me about songs, artists, or how music evolved across the decades. I specialize in English-language music from the 1990s to the 2020s.",
                        style={"fontFamily": "DM Sans, sans-serif", "color": COLORS["text"],
                               "margin": "8px 0 0 0", "lineHeight": "1.6"}
                    )
                ], style={
                    "backgroundColor": COLORS["bot_msg"],
                    "border": f"1px solid {COLORS['accent2']}",
                    "borderRadius": "12px",
                    "padding": "16px 20px",
                    "marginBottom": "12px"
                })
            ], style={
                "flex": 1,
                "overflowY": "auto",
                "padding": "20px",
                "display": "flex",
                "flexDirection": "column",
            }),

            html.Div(id="decada-badge", style={"padding": "0 20px 8px"}),

            html.Div([
                dcc.Input(
                    id="input-pregunta",
                    type="text",
                    placeholder="Ask about a song, artist, or musical era...",
                    debounce=False,
                    n_submit=0,
                    style={
                        "flex": 1,
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['accent2']}",
                        "borderRadius": "8px",
                        "color": COLORS["text"],
                        "fontFamily": "DM Sans, sans-serif",
                        "fontSize": "1.1rem",  # ← más grande
                        "padding": "14px 16px",  # ← más padding
                        "outline": "none",
                        "height": "50px",  # ← altura fija
                    }
                ),
                html.Button("Send", id="btn-enviar", n_clicks=0, style={
                    "backgroundColor": COLORS["accent"],
                    "color": "#0D0D0D",
                    "border": "none",
                    "borderRadius": "8px",
                    "fontFamily": "DM Sans, sans-serif",
                    "fontWeight": "500",
                    "fontSize": "0.95rem",
                    "padding": "12px 24px",
                    "cursor": "pointer",
                    "marginLeft": "8px",
                    "letterSpacing": "0.5px"
                }),
                html.Button("Clear", id="btn-limpiar", n_clicks=0, style={
                    "backgroundColor": "transparent",
                    "color": COLORS["muted"],
                    "border": f"1px solid {COLORS['muted']}",
                    "borderRadius": "8px",
                    "fontFamily": "DM Sans, sans-serif",
                    "fontSize": "0.9rem",
                    "padding": "12px 16px",
                    "cursor": "pointer",
                    "marginLeft": "8px",
                }),
            ], style={"display": "flex", "padding": "12px 20px 20px", "alignItems": "center"}),

        ], style={
            "display": "flex",
            "flexDirection": "column",
            "backgroundColor": COLORS["surface"],
            "borderRadius": "16px",
            "border": f"1px solid {COLORS['accent2']}",
            "height": "70vh",
            "flex": 2,
            "marginRight": "20px",
            "overflow": "hidden"
        }),

        html.Div([
            html.H3("Sources", style={
                "fontFamily": "Playfair Display, serif",
                "color": COLORS["accent"],
                "fontSize": "1.1rem",
                "marginBottom": "16px",
                "paddingBottom": "8px",
                "borderBottom": f"1px solid {COLORS['accent2']}"
            }),
            html.Div(id="panel-chunks", children=[
                html.P("Sources will appear here after your first question.",
                       style={"color": COLORS["muted"], "fontFamily": "DM Sans, sans-serif",
                              "fontSize": "0.85rem"})
            ], style={"overflowY": "auto", "flex": 1})
        ], style={
            "backgroundColor": COLORS["surface"],
            "borderRadius": "16px",
            "border": f"1px solid {COLORS['accent2']}",
            "padding": "20px",
            "flex": 1,
            "display": "flex",
            "flexDirection": "column",
            "height": "70vh",
            "overflow": "hidden"
        }),

    ], style={
        "display": "flex",
        "padding": "24px 32px",
        "flex": 1,
    }),

], style={
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "display": "flex",
    "flexDirection": "column",
    "fontFamily": "DM Sans, sans-serif"
})


@app.callback(
    Output("ventana-chat", "children"),
    Output("panel-chunks", "children"),
    Output("decada-badge", "children"),
    Output("store-historial", "data"),
    Output("input-pregunta", "value"),
    Input("btn-enviar", "n_clicks"),
    Input("input-pregunta", "n_submit"),
    Input("btn-limpiar", "n_clicks"),
    State("input-pregunta", "value"),
    State("ventana-chat", "children"),
    State("store-historial", "data"),
    prevent_initial_call=True
)
def manejar_interaccion(n_enviar, n_submit, n_limpiar, pregunta, mensajes_actuales, historial):
    ctx = dash.callback_context
    if not ctx.triggered:
        return mensajes_actuales, dash.no_update, dash.no_update, historial, ""

    trigger = ctx.triggered[0]["prop_id"]

    if "btn-limpiar" in trigger:
        bot.limpiar_historial()
        return [], [html.P("Sources will appear here after your first question.",
                           style={"color": COLORS["muted"], "fontFamily": "DM Sans, sans-serif",
                                  "fontSize": "0.85rem"})], "", [], ""

    if not pregunta or not pregunta.strip():
        return mensajes_actuales, dash.no_update, dash.no_update, historial, ""

    msg_usuario = html.Div([
        html.P(pregunta, style={
            "fontFamily": "DM Sans, sans-serif",
            "color": COLORS["text"],
            "margin": 0,
            "lineHeight": "1.6"
        })
    ], style={
        "backgroundColor": COLORS["user_msg"],
        "border": "1px solid #333",
        "borderRadius": "12px",
        "padding": "12px 16px",
        "marginBottom": "8px",
        "marginLeft": "40px",
        "alignSelf": "flex-end"
    })

    respuesta, decada_detectada, chunks_filtrados = bot.responder(pregunta)

    msg_bot = html.Div([
        html.Span("♪ ", style={"color": COLORS["accent"], "fontWeight": "bold"}),
        html.Span(respuesta, style={
            "fontFamily": "DM Sans, sans-serif",
            "color": COLORS["text"],
            "lineHeight": "1.6"
        })
    ], style={
        "backgroundColor": COLORS["bot_msg"],
        "border": f"1px solid {COLORS['accent2']}",
        "borderRadius": "12px",
        "padding": "12px 16px",
        "marginBottom": "12px",
        "marginRight": "40px"
    })

    nuevos_mensajes = list(mensajes_actuales) + [msg_usuario, msg_bot]

    badge = html.Span(
        f"🎵 Detected era: {decada_detectada}" if decada_detectada else "",
        style={
            "backgroundColor": COLORS["accent2"],
            "color": COLORS["text"],
            "fontFamily": "DM Sans, sans-serif",
            "fontSize": "0.8rem",
            "padding": "4px 10px",
            "borderRadius": "12px",
        }
    ) if decada_detectada else ""

    if chunks_filtrados:
        items_chunks = []
        for item in chunks_filtrados[:3]:
            texto = item["chunk"]["texto_ia"]
            lineas = texto.split("\n")
            ficha  = lineas[0] if lineas else ""
            letra  = lineas[1][:120] + "..." if len(lineas) > 1 else ""
            items_chunks.append(html.Div([
                html.P(ficha, style={
                    "color": COLORS["accent"],
                    "fontFamily": "DM Sans, sans-serif",
                    "fontSize": "0.78rem",
                    "fontWeight": "500",
                    "margin": "0 0 4px 0"
                }),
                html.P(letra, style={
                    "color": COLORS["muted"],
                    "fontFamily": "DM Sans, sans-serif",
                    "fontSize": "0.75rem",
                    "margin": 0,
                    "lineHeight": "1.4"
                }),
                html.P(f"Score: {item['score']:.3f}", style={
                    "color": COLORS["accent2"],
                    "fontSize": "0.7rem",
                    "margin": "4px 0 0 0"
                })
            ], style={
                "backgroundColor": COLORS["card"],
                "borderRadius": "8px",
                "padding": "10px 12px",
                "marginBottom": "8px",
                "border": "1px solid #333"
            }))
        panel_chunks = items_chunks
    else:
        panel_chunks = [html.P("No sources found.", style={"color": COLORS["muted"],
                                                            "fontFamily": "DM Sans, sans-serif",
                                                            "fontSize": "0.85rem"})]

    return nuevos_mensajes, panel_chunks, badge, historial, ""


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)