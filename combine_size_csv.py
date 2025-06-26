import pandas as pd

# ====== USER INPUT =========================================================
# csv_files = [
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-1.7B.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-8B.csv", 
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-14B.csv",
# ]

# csv_files = [
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-1.7B.csv",
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-8B.csv", 
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Qwen3-14B.csv",
# ]

# csv_files = [
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Llama-3.2-1B-Instruct.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Llama-3.2-3B-Instruct.csv",
# ]

csv_files = [
    "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Llama-3.2-1B-Instruct.csv",
    "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Llama-3.2-3B-Instruct.csv",
]

# csv_files = [
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-mini-128k-instruct.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-small-128k-instruct.csv",
#     "RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-medium-128k-instruct.csv",
# ]

# csv_files = [
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-mini-128k-instruct.csv",
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-small-128k-instruct.csv",
#     "NEW_RQ1_RAG_Answers_Basic_RAG_Prompt_Phi-3-medium-128k-instruct.csv",
# ]


# family_name = "Qwen3"  # ✅ Set the family name manually
family_name = "Llama-3.2"  # ✅ Set the family name manually
# family_name = "Phi-3"  # ✅ Set the family name manually

prompt_id = "Basic_RAG_Prompt"  # ✅ Set the prompt name manually
# ============================================================================

# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Verify columns are identical
expected_columns = [
    "Generation Model", "Question Index", "Question", "Type","Source_QID","Original Chunk", 
    "Chunks Retrieved", "Generated Answer", "Reference Answer", 
    "Generation Prompt Used", "Encoding Used","Enable Thinking"
]
assert combined_df.columns.tolist() == expected_columns, "Column mismatch detected!" 

# Write to a single combined CSV
# output_filename = f"combined_size_{family_name}_outputs_{prompt_id}.csv"
# output_filename = f"combined_size_{family_name}_outputs_{prompt_id}_added_model.csv"
output_filename = f"NEW_combined_size_{family_name}_outputs_{prompt_id}.csv"
combined_df.to_csv(output_filename, index=False)


print(f"✅ Combined CSV saved as '{output_filename}'")
