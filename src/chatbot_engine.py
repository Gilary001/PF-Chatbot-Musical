import os
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline

# ─── Ruta al modelo fine-tuneado ──────────────────────────────
RUTA_MODELO_CLASIFICADOR = os.path.join(
    os.path.dirname(__file__), "..", "models", "clasificador_decadas", "checkpoint-234"
)

# ─── Personalidad del Historiador Musical ─────────────────────
# 🟢 Adaptado — patrón de self.sistema del notebook 03_Agentes_Conversacionales_CUC
SYSTEM_PROMPT = """
You are HistoryBot, an expert musical historian specialized in English-language music.
Your knowledge comes exclusively from a corpus of song lyrics spanning the 1990s to the 2020s.

Your personality:
- You speak with academic authority but remain approachable and engaging
- You contextualize songs within their historical and cultural moment
- You explain how themes, vocabulary, and styles evolved across decades
- You always cite the specific song, artist, and year when referencing your corpus

Your capabilities:
- Classify any song lyric into its probable decade
- Compare lyrical themes across different eras
- Identify which themes dominated each decade based on real songs in your database
- Recommend songs from a specific era that match a given theme

Your limitations:
- You only answer questions about music history and song lyrics
- If asked about something outside your corpus or outside music history, respond:
  "That falls outside my expertise as a musical historian. I can only speak to what the songs in my collection tell us."
- You do not invent songs or lyrics. Every reference comes from your actual database.

When answering, always mention: artist, song title, and year when citing specific songs.
"""

# ─── Mapeo de décadas ─────────────────────────────────────────
def anio_a_decada(anio):
    try:
        anio = int(anio)
        if 1990 <= anio <= 1999: return "90s"
        if 2000 <= anio <= 2009: return "2000s"
        if 2010 <= anio <= 2019: return "2010s"
        if 2020 <= anio <= 2029: return "2020s"
    except: pass
    return None


# ─── Extraer año de un chunk ──────────────────────────────────
def _extraer_anio_de_chunk(chunk):
    """
    Extrae el año del chunk.
    Formato de rag_utils:
    'Song: titulo | Artist: artista | Genre: genero | Year: anio\nLyrics: ...'
    """
    texto = chunk.get("texto_ia", "")
    try:
        # Busca el patrón "Year: XXXX" en el texto
        import re
        match = re.search(r'Year:\s*(\d{4})', texto)
        if match:
            return int(match.group(1))
    except:
        pass

    # Intenta desde metadata si existe
    if "metadata" in chunk:
        return chunk["metadata"].get("anio")

    return None


# ─── Clase principal del Chatbot ──────────────────────────────
# 🟢 Adaptado — patrón de ChatbotLocal y AgenteRAGConversacional del notebook 03
class HistoryBotEngine:
    """
    Chatbot Historiador Musical.
    Combina RAG + clasificador de décadas + memoria conversacional.
    """

    def __init__(self, indice_faiss, chunks, modelo_emb, max_historial=5):
        self.indice_faiss  = indice_faiss
        self.chunks        = chunks
        self.modelo_emb    = modelo_emb
        self.max_historial = max_historial
        self.historial     = []
        self.sistema       = SYSTEM_PROMPT

        print("Cargando clasificador de décadas...")
        self._cargar_clasificador()

        print("Cargando generador Flan-T5...")
        self._cargar_generador()

        print("HistoryBot listo.")

    def _cargar_clasificador(self):
        try:
            model     = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO_CLASIFICADOR)
            tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO_CLASIFICADOR)
            self.clasificador = hf_pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el clasificador: {e}")
            self.clasificador = None

    def _detectar_decada(self, texto):
        if self.clasificador is None:
            return None
        try:
            return self.clasificador(texto[:512])[0]["label"]
        except:
            return None

    def _cargar_generador(self):
        """🟢 Adaptado — patrón _detectar_api() de ChatbotAPI notebook 03."""
        api = self._detectar_api()
        if api:
            self.modo_generacion = api
            print(f"Generador: API {api}")
        else:
            from src.rag_utils import cargar_modelo
            cargar_modelo()
            self.modo_generacion = "local"
            print("Generador: Flan-T5 local")

    def _detectar_api(self):
        """🟢 Directo — patrón de ChatbotAPI notebook 03."""
        if os.environ.get("OPENAI_API_KEY"):    return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"): return "claude"
        if os.environ.get("GOOGLE_API_KEY"):    return "gemini"
        return None

    def _filtrar_por_decada(self, chunks_relevantes, decada):
        if decada is None:
            return chunks_relevantes

        filtrados = []
        for item in chunks_relevantes:
            chunk = item["chunk"]
            anio  = _extraer_anio_de_chunk(chunk)
            if anio_a_decada(anio) == decada:
                filtrados.append(item)

        return filtrados if len(filtrados) >= 2 else chunks_relevantes

    def _generar_respuesta(self, contexto, pregunta):
        """🟢 Adaptado — patrón _llamar_* de ChatbotAPI notebook 03."""
        historial_texto = "\n".join(self.historial[-self.max_historial:])

        if self.modo_generacion == "local":
            from src.rag_utils import generar_con_flan_t5
            prompt = f"Based on these songs: {contexto[:500]}. Answer: {pregunta}"
            return generar_con_flan_t5(contexto[:500], prompt)

        elif self.modo_generacion == "claude":
            from anthropic import Anthropic
            client = Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=self.sistema,
                messages=[{"role": "user", "content":
                    f"History:\n{historial_texto}\n\nContext:\n{contexto}\n\nQuestion: {pregunta}"}]
            )
            return msg.content[0].text

        elif self.modo_generacion == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.sistema},
                    {"role": "user", "content":
                        f"History:\n{historial_texto}\n\nContext:\n{contexto}\n\nQuestion: {pregunta}"}
                ],
                max_tokens=400, temperature=0.7
            )
            return resp.choices[0].message.content

        elif self.modo_generacion == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"{self.sistema}\n\nHistory:\n{historial_texto}\n\nContext:\n{contexto}\n\nQuestion: {pregunta}"
            return model.generate_content(prompt).text

        return "No hay generador disponible."

    def responder(self, pregunta):
        """
        Pipeline completo:
        1. Detectar década con clasificador
        2. Buscar chunks con RAG
        3. Filtrar por década
        4. Generar respuesta con historial
        5. Actualizar memoria
        🟢 Adaptado — patrón responder() de AgenteRAGConversacional notebook 03
        """
        from src.rag_utils import buscar_chunks_relevantes

        # 1. Detectar década
        decada_detectada = self._detectar_decada(pregunta)

        # 2. Buscar chunks
        chunks_relevantes = buscar_chunks_relevantes(
            pregunta=pregunta,
            indice_FAISS=self.indice_faiss,
            chunks=self.chunks,
            modelo=self.modelo_emb,
            top_k=5
        )

        # 3. Filtrar por década
        chunks_filtrados = self._filtrar_por_decada(chunks_relevantes, decada_detectada)

        # 4. Construir contexto con texto_ia (formato de rag_utils)
        contexto = "\n\n".join([
            item["chunk"]["texto_ia"] for item in chunks_filtrados
        ])

        # 5. Generar respuesta
        respuesta = self._generar_respuesta(contexto, pregunta)

        # 6. Actualizar historial — patrón directo notebook 03
        self.historial.append(f"Usuario: {pregunta}")
        self.historial.append(f"HistoryBot: {respuesta}")

        return respuesta, decada_detectada, chunks_filtrados

    def limpiar_historial(self):
        self.historial = []
        print("Historial limpiado.")