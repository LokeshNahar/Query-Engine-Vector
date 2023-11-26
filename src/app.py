import streamlit as st
import requests
import pandas as pd
import time

# Define the FastAPI backend endpoint
backend_endpoint = "http://127.0.0.1:8000"


class ProductSearchApp:
    def __init__(self, backend_endpoint):
        self.backend_endpoint = backend_endpoint
        self.intro_text = "Made with 😍 by Lokesh Nahar, \n⭐Indian Institute of Technology Guwahati \n⭐Email - n.lokesh@iitg.ac.in\n⭐Mobile - +91-7300190532\n⭐Linkedin - lokesh-nahar-\n\t~Chaabi Hiring'23"

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
            if not search_query:
                st.warning("Please enter a query first!.")
                return
            else:
                self.perform_search(search_query)

    def perform_search(self, search_query):
        # Make a request to the FastAPI backend
        response = requests.get(f"{self.backend_endpoint}/?q={search_query}")
        st.balloons()
        with st.spinner('Wait for it...\nHappy Shopping $$$'):
            time.sleep(0.5)
        st.success('Done!')
        #st.snow()

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
        
        st.success("Search Results: -    -    -   -  -  -  -  -  -  - - - - - ---slide-right--->")
        # display Dataframe
        st.dataframe(df.style.background_gradient(cmap="Purples", axis=0), 5000, 1000, hide_index=False, column_config={
            "rating": st.column_config.NumberColumn(
                "Ratings",
                help="Ratings of the product",
                format="%f ⭐"
            )
        })

if __name__ == "__main__":
    
    # Create an instance of the ProductSearchApp class
    app = ProductSearchApp(backend_endpoint)
    
    # Run the app
    app.run()

