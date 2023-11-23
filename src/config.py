import os

CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "Data") # data directory
STATIC_DIR = os.path.join(ROOT_DIR, "static")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "query-search")
VECTOR_FIELD_NAME = "fast-bge-small-en"
TEXT_FIELD_NAME = "description"
