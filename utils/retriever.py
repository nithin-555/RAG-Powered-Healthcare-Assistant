import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from utils.embeddings import load_resources

# Initialize resources (lazy loaded)
_index = None
_df = None
_model = None
_cross_encoder = None

def get_resources():
    global _index, _df, _model, _cross_encoder
    if _index is None:
        print("Loading FAISS resources...")
        _index, _df, _model = load_resources()
    
    if _cross_encoder is None:
        print("Loading CrossEncoder...")
        # Using a lightweight cross-encoder for reranking
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    return _index, _df, _model, _cross_encoder

def retrieve(query, top_k=10, rerank_top_k=3):
    """
    Two-stage retrieval:
    1. FAISS vector search to get top_k candidates.
    2. Cross-Encoder reranking to get best rerank_top_k results.
    """
    index, df, model, cross_encoder = get_resources()
    
    if index is None or df is None:
        return []

    # 1. Vector Search
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx < len(df):
            item = df.iloc[idx].to_dict()
            item['initial_score'] = float(distances[0][i])
            candidates.append(item)
            
    if not candidates:
        return []

    # 2. Reranking
    pairs = [[query, c['Question'] + " " + c['Answer']] for c in candidates]
    scores = cross_encoder.predict(pairs)
    
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = float(scores[i])
        
    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked[:rerank_top_k]
