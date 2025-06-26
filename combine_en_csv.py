import pandas as pd


# ======  Testing enabling thinking with different model sizes=========================================================

# # USER INPUT
# csv_files = [
#      "RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#      "RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-1.7B.csv",
#      "RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
#      "RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-8B.csv",
#      "RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-14B.csv",

# ]

# family_name = "Qwen3"  # ✅ Set the family name manually
# prompt_id = "Basic_RAG_Prompt"  # ✅ Set the prompt name manually

# =====================================================================================================================



# ======  Testing enabling thinking with different model sizes=========================================================

# # USER INPUT
# csv_files = [
#      "NEW_RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#      "NEW_RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-1.7B.csv",
#      "NEW_RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
#      "NEW_RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-8B.csv",
#      "NEW_RQ3_size_RAG_Answers_Basic_RAG_Prompt_Qwen3-14B.csv",

# ]

# family_name = "Qwen3"  # ✅ Set the family name manually
# prompt_id = "Basic_RAG_Prompt"  # ✅ Set the prompt name manually

# =====================================================================================================================

# # ======  Testing enabling thinking with different prompts=========================================================

# # USER INPUT
# csv_files = [
#      "RQ3_prompt_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#      "RQ3_prompt_RAG_Answers_COT_Prompt_Qwen3-0.6B.csv",
#      "RQ3_prompt_RAG_Answers_No_Context_Prompt_Qwen3-0.6B.csv",
# ]


# model_name = "Qwen3-0.6B" # ✅ Set the model name manually

# # =====================================================================================================================


# # ======  Testing enabling thinking with different prompts=========================================================

# USER INPUT
csv_files = [
     "NEW_RQ3_prompt_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
     "NEW_RQ3_prompt_RAG_Answers_COT_Prompt_Qwen3-4B.csv",
     "NEW_RQ3_prompt_RAG_Answers_No_Context_Prompt_Qwen3-4B.csv",
]


model_name = "Qwen3-4B" # ✅ Set the model name manually

# # =====================================================================================================================

# =========================== COMMON PART=================================================================
# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Verify columns are identical
expected_columns = [
    "Generation Model", "Question Index", "Question", "Type","Source_QID","Original Chunk", 
    "Chunks Retrieved", "Generated Answer", "Reference Answer", 
    "Generation Prompt Used", "Encoding Used", "Enable Thinking"
]
assert combined_df.columns.tolist() == expected_columns, "Column mismatch detected!" 
# # ============================ =================================================================



# ======  Testing enabling thinking with different model sizes=========================================================

# output_filename = f"combined_en_size_{family_name}_outputs_{prompt_id}.csv"

#========================================================================================================================

# ======  Testing enabling thinking with different model sizes=========================================================

# output_filename = f"NEW_combined_en_size_{family_name}_outputs_{prompt_id}.csv"

#========================================================================================================================



# ======  Testing enabling thinking with different prompts ========================================================

# output_filename = f"combined_en_prompt_outputs_{model_name}.csv"

#========================================================================================================================

# ======  Testing enabling thinking with different prompts ========================================================

output_filename = f"NEW_combined_en_prompt_outputs_{model_name}.csv"

#========================================================================================================================
combined_df.to_csv(output_filename, index=False)
print(f"✅ Combined CSV saved as '{output_filename}'")
