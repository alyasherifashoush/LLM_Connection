import pandas as pd

# ====== USER INPUT =========================================================
csv_files = [
    "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
    "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-1.7B.csv",
    "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
    "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-8B.csv", 
]
prompt_id = "Basic_RAG_Prompt"  # ✅ Set the prompt name manually
# ============================================================================

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
output_filename = f"combined_size_outputs_{prompt_id}.csv"
combined_df.to_csv(output_filename, index=False)


print(f"✅ Combined CSV saved as '{output_filename}'")
