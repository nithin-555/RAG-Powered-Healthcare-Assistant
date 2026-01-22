import os
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = 'data/faiss_index.bin'
METADATA_FILE = 'data/metadata.pkl'
DATA_FILE = 'data/medquad.csv'

def create_embeddings(csv_path=DATA_FILE):
    """
    Loads CSV, generates embeddings, and saves FAISS index.
    """
    if not os.path.exists(csv_path):
        print(f"Data file {csv_path} not found.")
        return False

    print("Loading data...")
    df = pd.read_csv(csv_path)
    # Filter out empty answers
    df = df.dropna(subset=['Answer', 'Question'])
    
    # Combine Focus and Question for better embedding context
    df['combined_text'] = "Focus: " + df['Focus'].astype(str) + ". Question: " + df['Question'].astype(str)
    sentences = df['combined_text'].tolist()
    
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Generating embeddings for {len(sentences)} items...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save Index
    faiss.write_index(index, INDEX_FILE)
    
    # Save Metadata (original dataframe)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(df, f)
        
    print("Indexing complete.")
    return True

def load_resources():
    """
    Loads the FAISS index, metadata, and model.
    """
    if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
        return None, None, None
        
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'rb') as f:
            df = pickle.load(f)
        model = SentenceTransformer(MODEL_NAME)
        return index, df, model
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None

if __name__ == "__main__":
    create_embeddings()
