import chromadb

BASE_DIR = '../data'
CLIENT = chromadb.PersistentClient(path=f"{BASE_DIR}/embeddings/voiyage-2")