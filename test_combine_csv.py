import pandas as pd

# List the filenames of your CSVs
csv_files = [
    "RAG_Output_Answers_Qwen3-0.6B.csv",
    "RAG_Output_Answers_Qwen3-1.7B.csv",
    "RAG_Output_Answers_Qwen3-4B.csv",
    "RAG_Output_Answers_Qwen3-8B.csv",
   
]

# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Verify columns are identical
expected_columns = [
    "Generation Model", "Question Index", "Question", "Type","Source_QID","Original Chunk", 
    "Chunks Retrieved", "Generated Answer", "Reference Answer", 
    "Generation Prompt Used", "Encoding Used"
]
assert combined_df.columns.tolist() == expected_columns, "Column mismatch detected!"

# Write to a single combined CSV
combined_df.to_csv("combined_generation_outputs.csv", index=False)

print("✅ Combined CSV saved as 'combined_generation_outputs.csv'")
