import json
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from datasets import Dataset
warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"
DECADAS = ["90s", "2000s", "2010s", "2020s"]

# ─── Etiquetado ───────────────────────────────────────────────
def anio_a_decada(anio):
    try:
        anio = int(anio)
        if 1990 <= anio <= 1999: return "90s"
        if 2000 <= anio <= 2009: return "2000s"
        if 2010 <= anio <= 2019: return "2010s"
        if 2020 <= anio <= 2029: return "2020s"
    except: pass
    return None

# ─── Balanceo ─────────────────────────────────────────────────
def balancear_dataset(df, max_por_clase=900):
    partes = []
    for decada in DECADAS:
        subset = df[df["decada"] == decada]
        partes.append(subset.sample(n=min(len(subset), max_por_clase), random_state=42))
    return pd.concat(partes).reset_index(drop=True)

# ─── Preparar corpus ──────────────────────────────────────────
def preparar_corpus(collection):
    cursor = collection.find({}, {"_id": 0, "artista": 1, "titulo": 1,
                                   "letra": 1, "genero": 1, "anio": 1})
    df = pd.DataFrame(list(cursor))
    df.columns = df.columns.str.strip()
    df["decada"] = df["anio"].apply(anio_a_decada)
    df = df[df["decada"].notna()].copy()
    df = df[df["letra"].str.len() > 50].copy()
    return df


# ─── Split ────────────────────────────────────────────────────
def hacer_split(df, max_canciones=5000):
    decadas = sorted(df["decada"].unique())
    label2id = {d: i for i, d in enumerate(decadas)}
    id2label = {i: d for d, i in label2id.items()}
    df["label"] = df["decada"].map(label2id)

    df_sample = df.sample(n=min(max_canciones, len(df)), random_state=42)
    train_df, temp_df = train_test_split(df_sample, test_size=0.30,
                                          random_state=42, stratify=df_sample["label"])
    val_df, test_df   = train_test_split(temp_df,   test_size=0.50,
                                          random_state=42, stratify=temp_df["label"])
    return train_df, val_df, test_df, label2id, id2label


# ─── Tokenización ─────────────────────────────────────────────
def tokenizar_datasets(train_df, val_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenizar(batch):
        return tokenizer(batch["letra"], truncation=True,
                         padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(train_df[["letra", "label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["letra",  "label"]].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[["letra", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(tokenizar, batched=True)
    val_ds   = val_ds.map(tokenizar,   batched=True)
    test_ds  = test_ds.map(tokenizar,  batched=True)

    return train_ds, val_ds, test_ds, tokenizer


# ─── Guardar métricas ─────────────────────────────────────────
def guardar_metricas(train_df, val_df, test_df,
                     acc_ft, f1_ft, acc_zs, f1_zs,
                     ruta=r"C:\PF-Chatbot-Musical\resultados"):
    os.makedirs(ruta, exist_ok=True)

    metricas = {
        "modelo_base": MODEL_NAME,
        "tarea": "clasificacion_decada",
        "corpus": "canciones_ingles_mongodb",
        "decadas": DECADAS,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "finetuned":  {"accuracy": float(acc_ft), "f1_macro": float(f1_ft)},
        "zero_shot":  {"modelo": "cross-encoder/nli-MiniLM2-L6-H768",
                       "accuracy": float(acc_zs), "f1_macro": float(f1_zs)},
        "ganancia":   {"accuracy": float(acc_ft - acc_zs),
                       "f1_macro": float(f1_ft  - f1_zs)}
    }

    with open(os.path.join(ruta, "metricas.json"), "w") as f:
        json.dump(metricas, f, indent=2)

    print(f"Guardado en {os.path.join(ruta, 'metricas.json')}")