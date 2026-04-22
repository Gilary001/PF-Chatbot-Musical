from pymongo import MongoClient
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline
import torch


# ---Conexión de MongoDB---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "dbx_Canciones"
COL_NAME  = "canciones02"

# ---Conexión a la base de datos---
def get_collection():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    return db[COL_NAME]

#                                          ---------------------- Chunking ------------------------

# --Función que hace los chunking por estrofa estrategia A--
def chunking_por_estrofa(lista_canciones):
    todos_los_pedazos = []

    for cancion in lista_canciones:
        # Extraemos los datos
        titulo = cancion.get("titulo") or cancion.get("nombre") or "Sin Título"
        artista = cancion.get("artista", "Desconocido")
        genero = cancion.get("genero", "N/A")
        anio = cancion.get("anio", "N/A")
        letra = str(cancion.get("letra", ""))

        # Intenta cortar por estrofas primero
        estrofas = letra.split('\n\n')

        for estrofa in estrofas:
            texto_limpio = estrofa.strip() #Elimina espacios en blanco al inicio o al final

            if len(texto_limpio.split()) > 5:
                # Asegura que si la estrofa es demasiado larga (> 1000 caracteres) el sistema la divida cada 800
                if len(texto_limpio) > 1000:
                    for i in range(0, len(texto_limpio), 800):
                        sub_texto = texto_limpio[i: i + 800]
                        info_ia = (
                            f"Song: {titulo} | Artist: {artista} | Genre: {genero} | Year: {anio}\n"
                            f"Lyrics: {sub_texto}"
                        )
                        todos_los_pedazos.append({"texto_ia": info_ia, "longitud": len(info_ia)})
                else:
                    info_ia = (
                        f"Song: {titulo} | Artist: {artista} | Genre: {genero} | Year: {anio}\n"
                        f"Lyrics: {texto_limpio}"
                    )
                    todos_los_pedazos.append({"texto_ia": info_ia, "longitud": len(info_ia)})

    return todos_los_pedazos

# --Función que hace los chunking por canción completa estrategia B--
def chunking_cancion_completa(lista_canciones):
    todos_los_chunks = []
    for c in lista_canciones:
        letra = str(c.get("letra", "")).strip()
        if not letra:
            continue

        # Creamos el mismo formato de texto que la Estrategia A
        texto_unido = (
            f"Song: {c.get('titulo')} | Artist: {c.get('artista')} "
            f"| Genre: {c.get('genero')} | Year: {c.get('anio')}\n"
            f"Lyrics: {letra}"
        )

        diccionario_cancion = {
            "texto_ia": texto_unido,
            "longitud": len(texto_unido), # Agregamos longitud para que las métricas funcionen igual
            "metadata": {
                "titulo": c.get("titulo"),
                "artista": c.get("artista"),
                "genero": c.get("genero"),
                "anio": c.get("anio")
            }
        }
        todos_los_chunks.append(diccionario_cancion)
    return todos_los_chunks

# --Funcion que calcula los resultados(Metricas) de estrategia A vs B--
def mostrar_metricas_chunking(lista_chunks, nombre_estrategia):
    #Por si surge algun error
    if not lista_chunks:
        print(f"La lista de {nombre_estrategia} está vacía.")
        return

    # Extrae solo las longitudes del texto
    longitudes = [len(str(c.get("texto") or c.get("texto_ia", ""))) for c in lista_chunks]

    total = len(lista_chunks)
    promedio = sum(longitudes) / total
    v_min = min(longitudes)
    v_max = max(longitudes)

    # Tomamos el primer ejemplo para mostrarlo
    ejemplo = str(lista_chunks[0].get("texto") or lista_chunks[0].get("texto_ia", ""))

    print(f"--- MÉTRICAS ESTRATEGIA: {nombre_estrategia} ---")
    print(f"Total chunks: {total}")
    print(f"Tamaño promedio: {promedio:.0f} caracteres")
    print(f"Min/Max: {v_min}/{v_max} caracteres")
    print(f"Ejemplo: '{ejemplo[:100]}...'")
    print("-" * 40 + "\n")


#                       -------------------------------Embeddings------------------------------
# --Funcion que va a generar por primera vez los embeddings y en caso de ya tenerlos solo cargarlos--
def generar_o_cargar_embeddings(lista_chunks, nombre_archivo, modelo):
    ruta_archivo = f"{nombre_archivo}.pkl"

    # REVISA SI EL ARCHIVO YA EXISTE
    if os.path.exists(ruta_archivo):
        print(f"Cargando desde caché: {ruta_archivo} (Ahorrando tiempo...)")
        with open(ruta_archivo, "rb") as f:
            datos_completos = pickle.load(f)

        # Extrae los vectores
        vectores = np.array([d["vector"] for d in datos_completos]).astype('float32')
        return vectores, datos_completos

    # SI NO EXISTE, LO GENERA
    print(f"Generando {len(lista_chunks)} embeddings por primera vez...")
    solo_textos = [ch["texto_ia"] for ch in lista_chunks]
    vectores = modelo.encode(solo_textos, show_progress_bar=True)

    # Crea Texto + Vector
    datos_completos = []
    for i, chunk in enumerate(lista_chunks):
        datos_completos.append({
            "texto_ia": chunk["texto_ia"],
            "vector": vectores[i]
        })

    # GUARDAR PARA LA PRÓXIMA VEZ
    with open(ruta_archivo, "wb") as f:
        pickle.dump(datos_completos, f)

    print(f"Proceso terminado y guardado en: {ruta_archivo}")
    return vectores, datos_completos

#                       ----------------------Creacion de Indice FAISS----------------------
# --Funcion para encontrar la similitud--
def crear_indice_faiss(embeddings_datos):
    # Obtener la dimensión
    dimension = embeddings_datos.shape[1]

    # Crear el índice 'FlatL2'
    indice = faiss.IndexFlatL2(dimension) # L2 es un índice que mide la "Distancia Euclidiana"

    # Normaliza los vectores (Esto es para que la búsqueda sea por 'Similitud Coseno')
    # Ayuda a que se encuentre mejor los significados aunque las frases tengan largos distintos
    embeddings_norm = embeddings_datos.copy().astype('float32')
    faiss.normalize_L2(embeddings_norm)

    # Agregar los vectores al índice
    indice.add(embeddings_norm)

    print(f"Índice FAISS creado: {indice.ntotal} vectores, dimensión {dimension}")
    return indice

#                                       -------------Busqueda Semantica---------------
def buscar_chunks_relevantes(pregunta, indice_FAISS, chunks, modelo, top_k=5):
    # Proceso de búsqueda
    embedding_pregunta = modelo.encode([pregunta]).astype('float32')
    faiss.normalize_L2(embedding_pregunta)
    distancias, indices = indice_FAISS.search(embedding_pregunta, top_k)

    print(f"Resultados para: '{pregunta}'\n")
    print("=" * 60)

    resultados = []

    # Bucle para procesar e IMPRIMIR los resultados
    for i, (dist, idx) in enumerate(zip(distancias[0], indices[0])):
        # Extraemos la información del chunk
        chunk_info = chunks[idx]
        score = 1 - dist

        # Guardamos en la lista de resultados
        resultados.append({
            "chunk": chunk_info,
            "score": score,
            "indice": idx
        })

        # --- Resultados ---
        info_completa = chunk_info['texto_ia']

        print(f"Resultado #{i + 1} (Similitud: {score:.4f})")

        # Separa la ficha técnica de la letra
        lineas = info_completa.split('\n')
        print(f"{lineas[0]}")  # Song: ... | Artist: ...

        # Si hay letra, mostramos si no, evitamos error
        if len(lineas) > 1:
            print(f"{lineas[1][:150]}...")

        print("-" * 50)

    return resultados


#                                    ------- Generación de Respuestas-------
# Variables globales
tokenizer_local = None
model_local = None

# Funcion para cargar el modelo
def cargar_modelo():
    global tokenizer_local, model_local #Es una instrucción de "permiso"
    if model_local is None:
        model_id = "google/flan-t5-base"
        print(f"Cargando {model_id}...")
        try:
            tokenizer_local = AutoTokenizer.from_pretrained(model_id) #carga la herramienta que convierte palabras en listas de números
            model_local = AutoModelForSeq2SeqLM.from_pretrained(model_id) #Es el cerebro
            print("Modelo Base listo.")
        except Exception as e:
            print(f"Error crítico en la carga: {e}")


# --- Generador el modelo ---
# Funcion del modelo configuración etc...
def generar_con_flan_t5(contexto, pregunta):
    prompt = f"Using this text: {contexto}. Answer this: {pregunta}"

    inputs = tokenizer_local(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model_local.generate(
            **inputs,
            max_new_tokens=100,      ## Cuántas palabras nuevas puede escribir
            do_sample=True,          # Permite que la IA "elija" entre varias palabras probables.
            temperature=0.6,         # Creatividad
            top_p=0.4,
            repetition_penalty=1.5   # Evita que el modelo se quede atrapado diciendo la misma palabra una y otra vez
        )

    respuesta = tokenizer_local.decode(outputs[0], skip_special_tokens=True) #skip_special... Limpia la respuesta de etiquetas técnicas
    return respuesta

#                                       ----------Sistema RAG Completo--------------
def rag_completo(pregunta, indice_faiss, chunks, modelo_emb, top_k=3, modelo="local"):
    print(f"\n{'=' * 60}")
    print(f"BUSCANDO: {pregunta}")
    print(f"{'=' * 60}")

    # Búsqueda Semántica
    # Esta función usa el modelo de embeddings, el índice FAISS y el modelo paraphrase-multilingual-MiniLM-L12-v2
    resultados_busqueda = buscar_chunks_relevantes(pregunta, indice_faiss, chunks, modelo_emb, top_k)

    if not resultados_busqueda:
        return "No se encontraron fragmentos de canciones relacionados."

    # Construcción del Contexto
    # Une los textos de los chunks recuperados para que la IA los lea
    contexto = "\n\n".join([res['chunk']['texto_ia'] for res in resultados_busqueda])

    print(f"\nGenerando respuesta con modelo {modelo}...")

    # Generación de la respuesta
    if modelo == "local":
        respuesta = generar_con_flan_t5(contexto, pregunta)
    else:
        respuesta = "Modelo no reconocido."

    print(f"\nRESPUESTA FINAL:")
    print(f"{respuesta}")
    print(f"{'=' * 60}")

    return respuesta


def sin_rag(pregunta):
    prompt = f"Question: {pregunta} Answer:"

    inputs = tokenizer_local(prompt, return_tensors="pt")
    outputs = model_local.generate(**inputs, max_new_tokens=50)

    return tokenizer_local.decode(outputs[0], skip_special_tokens=True)































