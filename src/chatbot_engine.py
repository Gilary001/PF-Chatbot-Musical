import os
import re
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline


# ─── Ruta LOCAL al checkpoint fine-tuneado ─────────────────────
RUTA_MODELO_CLASIFICADOR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "clasificador_decadas", "checkpoint-234")
)


# ─── PROMPT EN ESPAÑOL ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are HistoryBot, a music historian specializing in English-language music.

Your knowledge comes exclusively from a real corpus of song lyrics
from the 1990s to the 2020s.

Strict rules:
- You ALWAYS answer in English.

- The songs can be in English, so your explanation is in English.

- In the exact moment the user writes to you in Spanish, tell them "Please write in English", don't give them a response..

- Always mention the artist, song title, and year when citing examples.

- You don't make up songs.

- You don't invent information outside the corpus.

- You can only talk about music history and lyrics.

If the question isn't about music or lyrics, answer:

"That's outside my area as a music historian."
"""


def anio_a_decada(anio):
    try:
        anio = int(anio)
        if 1990 <= anio <= 1999: return "90s"
        if 2000 <= anio <= 2009: return "2000s"
        if 2010 <= anio <= 2019: return "2010s"
        if 2020 <= anio <= 2029: return "2020s"
    except:
        pass
    return None


def _extraer_anio_de_chunk(chunk):
    texto = chunk.get("texto_ia", "")
    match = re.search(r'Year:\s*(\d{4})', texto)
    if match:
        return int(match.group(1))

    if "metadata" in chunk:
        return chunk["metadata"].get("anio")

    return None


class HistoryBotEngine:

    def __init__(self, indice_faiss, chunks, modelo_emb, max_historial=5):
        self.indice_faiss = indice_faiss
        self.chunks = chunks
        self.modelo_emb = modelo_emb
        self.max_historial = max_historial
        self.historial = []

        print("Cargando clasificador fine-tuneado...")
        self._cargar_clasificador()

        print("Cargando generador...")
        self._cargar_generador()

        print("HistoryBot listo.")

    # ───────────── CLASIFICADOR FINE-TUNED ─────────────
    def _cargar_clasificador(self):

        self.tokenizer_clf = AutoTokenizer.from_pretrained(
            RUTA_MODELO_CLASIFICADOR,
            local_files_only=True
        )

        self.modelo_clf = AutoModelForSequenceClassification.from_pretrained(
            RUTA_MODELO_CLASIFICADOR,
            local_files_only=True
        )

        self.clasificador = hf_pipeline(
            "text-classification",
            model=self.modelo_clf,
            tokenizer=self.tokenizer_clf
        )

    def _detectar_decada(self, texto):
        try:
            resultado = self.clasificador(texto[:512])[0]["label"]
            return resultado
        except:
            return None

    # ───────────── GENERADOR (Flan desde rag_utils) ─────────────
    def _cargar_generador(self):
        from src.rag_utils import cargar_modelo
        cargar_modelo()
        self.modo_generacion = "local"

    def _generar_respuesta(self, contexto, pregunta):
        from src.rag_utils import generar_con_flan_t5
        prompt = f"What is the answer to '{pregunta}' based on: {contexto[:200]}" # ← cambiá esta línea
        return generar_con_flan_t5(contexto[:200], prompt)
    # ───────────── FILTRO POR DÉCADA ─────────────
    def _filtrar_por_decada(self, chunks_relevantes, decada):
        if decada is None:
            return chunks_relevantes

        filtrados = []

        for item in chunks_relevantes:
            anio = _extraer_anio_de_chunk(item["chunk"])
            if anio_a_decada(anio) == decada:
                filtrados.append(item)

        return filtrados if len(filtrados) >= 2 else chunks_relevantes

    def _es_pregunta_general(self, pregunta):
        texto = pregunta.lower()
        palabras_generales = [
            "How it evolved",
            "What themes dominated",
            "What characterizes it",
            "How it changed",
            "Differences between"
        ]
        return any(p in texto for p in palabras_generales)

    def _es_pregunta_musical(self, pregunta):
        texto = pregunta.lower()

        palabras_clave = [
            "song", "songs", "artist", "album",
            "rock", "pop", "hip hop",
            "90s", "2000s", "2010s", "2020s",
            "lyrics", "theme", "genre",
            "decade", "era"
        ]

        return any(p in texto for p in palabras_clave)

    def _es_saludo_inicial(self, pregunta):
        texto = pregunta.lower().strip()

        saludos_puros = [
            "hello",
            "hi",
            "Good Morning",
            "Good Afternoon",
            "Good Night"
        ]

        return texto in saludos_puros

    # ───────────── PIPELINE COMPLETO ─────────────
    def responder(self, pregunta):

        from src.rag_utils import buscar_chunks_relevantes

        # 0. Saludo inicial
        if self._es_saludo_inicial(pregunta) and len(self.historial) == 0:
            respuesta = "Hi 🙂 I'm your music historian. What decade or artist are you interested in?"
            self.historial.append(f"Usuario: {pregunta}")
            self.historial.append(f"HistoryBot: {respuesta}")
            return respuesta, None, []

        # 0.1. Si NO es musical → conversación ligera
        if not self._es_pregunta_musical(pregunta):

            texto = pregunta.lower().strip()

            if texto in ["ok", "okay", "good", "perfect"]:
                respuesta = "Perfect 🙂 When you want to talk about music, tell me the decade or artist you're interested in."

            elif texto in ["?", "what?", "like?", "how?"]:
                respuesta = "Could you be a little more specific about your question regarding music?"

            elif "how are you" in texto or "how are you?" in texto:
                respuesta = "Great 🙂 Always ready to talk about music. What era are you interested in discussing?"

            else:
                respuesta = "I can help you with music history, artists, decades, or lyric analysis. What are you interested in?"

            self.historial.append(f"Usuario: {pregunta}")
            self.historial.append(f"HistoryBot: {respuesta}")
            return respuesta, None, []


        # 1️⃣ Clasificador fine-tuneado
        decada_detectada = self._detectar_decada(pregunta)

        # 2️⃣ RAG
        chunks_relevantes = buscar_chunks_relevantes(
            pregunta=pregunta,
            indice_FAISS=self.indice_faiss,
            chunks=self.chunks,
            modelo=self.modelo_emb,
            top_k=5
        )

        # 3️⃣ Filtro por década
        chunks_filtrados = self._filtrar_por_decada(
            chunks_relevantes,
            decada_detectada
        )

        # 4️⃣ Construir contexto
        contexto = "\n\n".join([
            item["chunk"]["texto_ia"]
            for item in chunks_filtrados[:3]
        ])

        # 5️⃣ Generación
        respuesta = self._generar_respuesta(contexto, pregunta)

        # 6️⃣ Memoria
        self.historial.append(f"Usuario: {pregunta}")
        self.historial.append(f"HistoryBot: {respuesta}")

        return respuesta, decada_detectada, chunks_filtrados

    def limpiar_historial(self):
        self.historial = []
        print("Historial limpiado.")