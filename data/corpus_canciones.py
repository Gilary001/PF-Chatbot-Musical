from pymongo import MongoClient

# ---Conexión de MongoDB---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "dbx_Canciones"
COL_NAME  = "canciones02"

# ---Conexión a la base de datos---
def get_collection():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    return db[COL_NAME]