import os
import pickle
import faiss
import torch
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# ============================================================
# CONFIGURACIÓN Y RUTAS
# ============================================================
CACHE_DIR = "../notebooks"
MODELO_FINETUNED_PATH = "../models/clasificador_decadas/checkpoint-234"
FAISS_PATH = os.path.join(CACHE_DIR, "faiss_index_A.bin")
CHUNKS_PATH = os.path.join(CACHE_DIR, "emb_por_estrofa.pkl")

# API KEY (Asegúrate de que sea válida)
os.environ["GROQ_API_KEY"] = "key_api"


class MúsicBotCUC:
    def __init__(self):
        print("🛠️ Inicializando MúsicBot...")
        try:
            # 1. Componentes de Clasificación y Embeddings
            self.tk_class = AutoTokenizer.from_pretrained(MODELO_FINETUNED_PATH)
            self.mod_class = AutoModelForSequenceClassification.from_pretrained(MODELO_FINETUNED_PATH)
            self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

            # 2. Base de Datos Vectorial
            self.indice = faiss.read_index(FAISS_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)

            # 3. Clientes de Generación
            self.client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            # Memoria Conversacional (Límite de 5 turnos = 10 mensajes)
            self.historial = []
            self.max_mensajes = 10

            print(f"✅ Sistema listo con {len(self.chunks)} canciones.")
        except Exception as e:
            print(f"❌ Error crítico: {e}")

    def clasificar_intencion(self, texto):
        inputs = self.tk_class(texto, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.mod_class(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        return {0: "Letras", 1: "Artista", 2: "Género"}.get(pred_id, "General")

    def buscar_contexto(self, texto):
        # Generar vector y buscar en FAISS
        v_pregunta = self.embedder.encode([texto]).astype("float32")
        faiss.normalize_L2(v_pregunta)
        distancias, indices = self.indice.search(v_pregunta, k=5)

        contexto_lista = []
        for idx in indices[0]:
            if idx != -1:
                item = self.chunks[int(idx)]
                txt = item.get("texto_ia") or item.get("texto") or str(item)
                contexto_lista.append(txt)
        return "\n---\n".join(contexto_lista)

    def responder(self, usuario_input):
        # A. Clasificación e Intención
        intencion = self.clasificar_intencion(usuario_input)

        # B. Detección de seguimiento (Si es pregunta corta, usamos memoria, no FAISS)
        es_seguimiento = any(p in usuario_input.lower() for p in ["quien", "año", "genero", "esa", "artista"])
        contexto = "" if es_seguimiento else self.buscar_contexto(usuario_input)

        # C. Construcción del Prompt para Groq
        system_prompt = (
            "Eres MúsicBot eres un experto en musica principalmente en pop, rock y hip hop." 
            "Responde basado EXCLUSIVAMENTE en el contexto proporcionado no inventes nada. "
            "Si la respuesta no está en el contexto o en el historial reciente, di que no sabes."
            "NUNCA uses conocimiento externo para inventar canciones o artistas.\n"
            "Si el contexto no tiene la respuesta decí: "
            "'No tengo esa información en mi corpus.'\n"
            "Sé conciso. Máximo 3 párrafos."
        )

        # Preparamos los mensajes: System + Historial + Pregunta Actual
        mensajes_groq = [{"role": "system", "content": system_prompt}]

        # Inyectar memoria
        for h in self.historial:
            mensajes_groq.append(h)

        # Pregunta actual con contexto RAG (si existe)
        contenido_usuario = f"CONTEXTO RAG: {contexto}\n\nPREGUNTA: {usuario_input}"
        mensajes_groq.append({"role": "user", "content": contenido_usuario})

        # D. Generación con Groq (Llama 4 Scout)
        try:
            chat = self.client_groq.chat.completions.create(
                messages=mensajes_groq,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                top_p=0.2
            )
            respuesta = chat.choices[0].message.content
        except Exception as e:
            respuesta = f"Error en API: {e}"

        # E. Actualizar Historial (Mantener solo últimos X mensajes)
        self.historial.append({"role": "user", "content": usuario_input})
        self.historial.append({"role": "assistant", "content": respuesta})
        if len(self.historial) > self.max_mensajes:
            self.historial = self.historial[-self.max_mensajes:]

        return f"[Intención: {intencion}] {respuesta}"


# ============================================================
# BUCLE PRINCIPAL
# ============================================================
if __name__ == "__main__":
    bot = MúsicBotCUC()
    while True:
        u = input("Usuario: ")
        if u.lower() in ["salir", "exit"]: break
        print(f"Bot: {bot.responder(u)}\n")