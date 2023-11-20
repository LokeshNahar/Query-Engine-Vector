from fastapi import FastAPI
from llm_searcher import LLMSearcher
from fastapi.middleware.cors import CORSMiddleware
from config import COLLECTION_NAME
import uvicorn

# Create a FastAPI application instance
app = FastAPI()

# Adding CORS middleware to handle Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with specific origins in a production environment)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create a searcher instance
llm_searcher = LLMSearcher(collection_name=COLLECTION_NAME)


# Define a route for the root endpoint ("/") with a query parameter named 'query'
@app.get("/")
def search_startup(q: str):
    # Return a JSON response with the search result obtained from LLMSearcher
    return {
        "result": llm_searcher.search(text=q)
    }


if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn with specified host and port
    uvicorn.run(app, host="0.0.0.0", port=6553)