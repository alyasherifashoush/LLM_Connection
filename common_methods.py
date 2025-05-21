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

########################################################################################################################
import pandas as pd
def get_CSVcolumn(csv_file,column_name):


    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the questions into a list
    column_list = df[column_name].tolist()
    
    # Return the list
    return column_list

#################################################################################################################################################################

from sentence_transformers import SentenceTransformer #This line imports the SentenceTransformer class from the sentence-transformers library. This library provides pre-trained and fine-tuned sentence embedding models.
import torch #This line imports the torch library, which is a popular deep learning framework. It provides functionalities for tensor computations and building neural networks. When you import torch, you are importing pyTorch which is the python library for the torch framework. The names are used interchangeably.
import pandas as pd #This line imports the pandas library, which is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrames and Series to work with structured data.


def get_top_matching_chunks(query, k):
    
    # Load the model only once (move it outside this function if calling often)
    # model = SentenceTransformer("all-MiniLM-L6-v2") 
    model = SentenceTransformer("all-mpnet-base-v2") # around 7% better 

    # This line uses the read_csv function from the pandas library to load data from a CSV file named "chunks.csv" into a DataFrame called df
    df = pd.read_csv("chunks.csv") 

    # Extract data from indicated columns from the DataFrame into Python lists
    chunk_ids = df["chunk_id"].tolist()
    chunk_texts = df["chunk_text"].tolist()

    # Encode chunks and query, then convert to tensor
    # Without 'convert_to_tensor=True', chunk_embeddings and query_embedding would be lists of NumPy arrays, not PyTorch tensors.
    # We need to be using PyTorch tensors for the cosine similarity computation.
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True) 
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity

    #This line creates a CosineSimilarity object from the torch.nn module
    # dim=0 compares vectors column by column, while dim=1 compares vectors row by row.
    cos = torch.nn.CosineSimilarity(dim=1) 

    #This line calculates the cosine similarity between the embedding of the query and the embeddings of each of the chunk_texts
    # It is using the forward method of the cos object (which is implicitly called when you treat the object like a function) to perform the actual cosine similarity calculation.
    similarities = cos(chunk_embeddings, query_embedding)

    # Get top-k indices
    # It returns a named tuple containing two tensors: values (the top k similarity scores) and indices (the indices of these top k scores in the original similarities tensor).
    # .indices extracts the tensor of indices
    # .tolist() converts the PyTorch tensor of indices into a Python list.
    top_indices = torch.topk(similarities, k=k).indices.tolist()

    # Map indices back to chunk_ids
    top_chunk_ids = [chunk_ids[i] for i in top_indices]

    return top_chunk_ids