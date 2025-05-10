import unicodedata
import re

def normalize_string(s):
    # Step 1: Convert to lowercase
    s = s.lower()
    
    # Step 2: Strip leading/trailing whitespace
    s = s.strip()
    
    # Step 3: Normalize Unicode characters (e.g., accented characters)
    s = unicodedata.normalize('NFKD', s)  # Normalize to decomposed form
    
    # Step 4: Remove non-printable characters (e.g., invisible characters)
    s = re.sub(r'[^\x20-\x7E]', '', s)  # Keep only printable characters
    
    return s

import pandas as pd
def get_CSVcolumn(csv_file,column_name):


    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the questions into a list
    column_list = df[column_name].tolist()
    
    # Return the list
    return column_list


from sentence_transformers import SentenceTransformer
import torch
import pandas as pd


def get_top_matching_chunks(query, k):
    
    # Load the model only once (move it outside this function if calling often)
    # model = SentenceTransformer("all-MiniLM-L6-v2") 
    model = SentenceTransformer("all-mpnet-base-v2") # around 7% better 

    # Load CSV with both chunk_id and chunk_text
    df = pd.read_csv("chunks.csv")  # assumes columns: chunk_id, chunk_text

    # Get lists
    chunk_ids = df["chunk_id"].tolist()
    chunk_texts = df["chunk_text"].tolist()

    # Encode chunks and query
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1)
    similarities = cos(chunk_embeddings, query_embedding)

    # Get top-k indices
    top_indices = torch.topk(similarities, k=k).indices.tolist()


    # Map indices back to chunk_ids
    top_chunk_ids = [chunk_ids[i] for i in top_indices]

    return top_chunk_ids