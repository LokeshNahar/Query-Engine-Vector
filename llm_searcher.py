from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import QDRANT_URL
# class sentencetransformer searcher
class LLMSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(QDRANT_URL)

    # searching the text query
    def search(self, text: str):
        # Convert text query into vector
        # df = pd.read_csv('bigBasketProducts.csv')
        rnge = 0
        vector = self.model.encode(text).tolist()

        # Define a filter for ratings
        query_filter = Filter(
            should=[
            FieldCondition(
                key="rating",
                range=Range(
                    gt=None,
                    gte=rnge,
                    lt=None,
                    lte=None,
                ),
            ),
        ],
        )
        #  retrieve top 'limit' queries
        limit = 10
        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=("fast-bge-small-en",vector),
            query_filter=None,  # for filtered query
            limit=limit  # 15 the most closest results is enough
        )

        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # extracting and returning the list of top 'limit' queries
        payloads = [hit.payload for hit in search_result]
        return payloads
    
