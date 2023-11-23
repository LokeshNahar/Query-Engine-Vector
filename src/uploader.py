#import the necessary libraries
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os,re
import nltk,ssl
from qdrant_client import QdrantClient, models
from config import DATA_DIR,COLLECTION_NAME,QDRANT_URL,VECTOR_FIELD_NAME,TEXT_FIELD_NAME

# Set file paths and other configuration parameters
csv_file_path = os.path.join(DATA_DIR, "bigBasketProducts.csv")
npy_file_path = os.path.join(DATA_DIR, "bb_chaabi_vectors.npy")
collection_name = COLLECTION_NAME
qdrant_url = QDRANT_URL
vector_field_name = VECTOR_FIELD_NAME
text_field_name = TEXT_FIELD_NAME

# Handle SSL certificate verification for older Python versions to simply download stopwords module from nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Class for preprocessing a DataFrame
class DataFramePreprocessor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        # Load the CSV file into a DataFrame
        self.df = pd.read_csv(csv_file_path)

    # Remove special characters and numbers
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))     
        text = text.lower()                              # Convert to lowercase
        return text
    
    #removing stopwords from text
    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))     # list of all the stopwords
        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]     
        return ' '.join(words)

    # Preprocess the DataFrame
    def preprocess_dataframe(self):
        # cleaning all the fields text
        self.df['description'] = self.df['description'].apply(self.clean_text)
        self.df['description'] = self.df['description'].apply(self.remove_stopwords)
        self.df['product'] = self.df['product'].apply(self.clean_text)
        self.df['category'] = self.df['category'].apply(self.clean_text)
        self.df['sub_category'] = self.df['sub_category'].apply(self.clean_text)
        self.df['brand'] = self.df['brand'].apply(self.clean_text)
        self.df['type'] = self.df['type'].apply(self.clean_text)
        # replace Null values to NA
        self.df.fillna("NA", inplace=True)
        # changing type of all the collumns to be str
        self.df = self.df.astype(str)

# Class for uploading embeddings to Qdrant
class QdrantUploader:
    def __init__(self, csv_file_path, npy_file_path):
        self.csv_file_path = csv_file_path
        self.npy_file_path = npy_file_path

    def generate_bert_embeddings(self):
        preprocessor = DataFramePreprocessor(self.csv_file_path)
        preprocessor.preprocess_dataframe()
        # Concatenate multiple columns into a single string for each row
        preprocessor.df['combined_text'] = preprocessor.df.apply(lambda row: f"{row.product} {row.category} {row.sub_category} {row.type} {row.brand} {row.description}", axis=1)

        # Tokenizer and model initialization
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = TFBertModel.from_pretrained("bert-base-cased")

        # Tokenize and pad the sequences
        max_length = 128  # You can adjust this based on your requirements
        tokenized_inputs = tokenizer(
            list(preprocessor.df['combined_text']),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        # Generate embeddings
        batch_size = 32  # You can adjust this based on your GPU memory
        embeddings = []
        for i in tqdm(range(0, len(preprocessor.df), batch_size)):
            batch_inputs = {key: val[i:i + batch_size] for key, val in tokenized_inputs.items()}
            outputs = model(**batch_inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings.append(last_hidden_states)

        # Concatenate embeddings for all batches
        embeddings = tf.concat(embeddings, axis=0)
        # Convert embeddings to numpy array
        embeddings_np = embeddings.numpy()
        # vectors will have shape (num_samples, hidden_size)
        vectors = tf.reduce_mean(embeddings_np, axis=1)

        # Save the generated embeddings to a npy file
        np.save(self.npy_file_path, vectors, allow_pickle=False)
        print(f"Embeddings saved to {self.npy_file_path}")


    # Generate sentence embeddings using Sentence Transformers
    def generate_embeddings(self):
        model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")
        # model = SentenceTransformer("paraphrase-albert-small-v2", device = "cuda")

        preprocessor = DataFramePreprocessor(self.csv_file_path)
        preprocessor.preprocess_dataframe()

        # Concatenate relevant columns and encode using the Sentence Transformer model
        vectors = model.encode([
            str(row.product) + ". " + str(row.category) + ". " + str(row.sub_category) + ". " + str(row.type) + ". " +
            str(row.brand) + ". " + str(row.description) for row in preprocessor.df.itertuples()
        ], show_progress_bar=True)

        # Save the generated embeddings to a npy file
        np.save(self.npy_file_path, vectors, allow_pickle=False)
        print(f"Embeddings saved to {self.npy_file_path}")

    def upload_embeddings(self, collection_name, qdrant_url, vector_field_name, text_field_name):
        client = QdrantClient(url=qdrant_url)
        preprocessor = DataFramePreprocessor(self.csv_file_path)
        # preprocessor.preprocess_dataframe()

        # Handle missing values in the DataFrame
        df = preprocessor.df
        df.fillna({"rating" : 0},inplace=True)
        df.fillna("NA", inplace=True)
        payload = df.to_dict('records')

        # Load saved embeddings and upload to Qdrant
        vectors = np.load(self.npy_file_path)

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                vector_field_name: models.VectorParams(
                    size=vectors.shape[1],
                    distance=models.Distance.COSINE
                )
            },
            # quantization to reduce the memory usage
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        )
        # Upload vectors and associated metadata to the Qdrant collection
        client.upload_collection(
            collection_name=collection_name,
            vectors={
                vector_field_name: vectors
            },
            payload=payload,
            ids=None,               # Vector ids will be assigned automatically
            batch_size=256          # How many vectors will be uploaded in a single request?
        )

    # Delete the current Qdrant collection
    def delete_current_collections(self,collection_name,qdrant_url):
        client = QdrantClient(
            url=qdrant_url
        )
        client.delete_collection(collection_name=collection_name)

    def list_all_collections(self,qdrant_url):
        client = QdrantClient(
            url=qdrant_url
        )
        client.get_collections()



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process data, generate embeddings, and upload to Qdrant")
    parser.add_argument("--generate_embeddings", action="store_true", help="Generate embeddings and save to file")
    parser.add_argument("--delete_collection", action="store_true", help="delete the current collection")
    parser.add_argument("--generate_bert_embeddings", action="store_true", help="Generate bert embeddings and save to file")
    args = parser.parse_args()
    uploader = QdrantUploader(csv_file_path, npy_file_path)
    
    # Check if the --generate_embeddings flag is provided
    if args.generate_embeddings:
        uploader.generate_embeddings()

    # Check if the --generate_embeddings flag is provided
    if args.generate_bert_embeddings:
        uploader.generate_bert_embeddings()

    # Check if the --generate_embeddings flag is provided    
    if args.delete_collection:
        uploader.delete_current_collections(collection_name,qdrant_url)

    # uploader.list_all_collections(qdrant_url)

    # upload the vectors saved in the npy file in the directory
    uploader.upload_embeddings(collection_name, qdrant_url, vector_field_name, text_field_name)
