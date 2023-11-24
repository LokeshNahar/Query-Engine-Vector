import streamlit as st
import requests
import pandas as pd
import time
from config import QDRANT_API_KEY,QDRANT_URL,COLLECTION_NAME
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class ProductSearchApp:
    def __init__(self):
        self.intro_text = "Made with 😍 by Lokesh Nahar, \n⭐Indian Institute of Technology Guwahati \n⭐Email - n.lokesh@iitg.ac.in\n⭐Mobile - +91-7300190532\n⭐Linkedin - lokesh-nahar-\n\t~Chaabi Hiring'23"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    def run(self):
        # Streamlit UI
        st.title("Product Search App")
        st.text(self.intro_text)

        # User input for the search query
        st.text("Welcome to The App")
        search_query = st.text_input("Enter your search query:")
        add_selectbox = st.sidebar.text_area("Please Provide Feedback")

        # Search button
        if st.button("Search"):
            self.perform_search(search_query)

    def perform_search(self, search_query):
        # Make a request to the FastAPI backend
        # response = requests.get(f"{self.backend_endpoint}/?q={search_query}")
        # Convert text query into vector
        vector = self.model.encode(search_query).tolist()
        qdrant_client = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)
        # Use `vector` for search for closest vectors in the collection
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector, # If you don't want any filters for now
            limit=15  # 15 the most closest results is enough
        )
        st.write(search_result)
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        st.balloons()
        with st.spinner('Wait for it...\nHappy Shopping $$$'):
            time.sleep(1)
        st.success('Done!')
        #st.snow()
        response = {"result":payloads, "status_code":200, "text":"Ok"}
        # Check if the request was successful
        if response.status_code == 200:
            self.display_search_results(response)
        else:
            # Display an error message if the request fails
            st.error(f"Error: {response.status_code} - {response.text}")

    def display_search_results(self, response):
        # Display the search results
        results = response.json()["result"]
        df = pd.DataFrame(results)
        
        st.write("Search Results:           ....     ---slide-right--->")
        # display Dataframe
        st.dataframe(df.style.background_gradient(cmap="Purples", axis=0), 5000, 1000, hide_index=False, column_config={
            "rating": st.column_config.NumberColumn(
                "Ratings",
                help="Ratings of the product",
                format="%d ⭐",
            )
        })

if __name__ == "__main__":
    
    # Create an instance of the ProductSearchApp class
    app = ProductSearchApp()
    
    # Run the app
    app.run()

