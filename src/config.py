import os

CODE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "Data") # data directory
STATIC_DIR = os.path.join(ROOT_DIR, "static")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
# QDRANT_URL = "https://66f1a69f-f196-4be1-a28f-bc1a0ad2e267.us-east4-0.gcp.cloud.qdrant.io:6333"
# QDRANT_API_KEY = "Sixe5-rGSnummTkLdNPvJ8Czkx-VXpNWbqRzBZW6uIUmroNpJtmeRw"

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "query-search")
VECTOR_FIELD_NAME = "fast-bge-small-en"
TEXT_FIELD_NAME = "description"
