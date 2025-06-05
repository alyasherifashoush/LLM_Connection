import pandas as pd

# # ====== USER INPUT =======================================================
# csv_files = [
#     "RQ2_RAG_Answers_Basic_RAG_Prompt_Qwen3-0.6B.csv",
#     "RQ2_RAG_Answers_COT_Prompt_Qwen3-0.6B.csv",
#     "RQ2_RAG_Answers_No_Context_Prompt_Qwen3-0.6B.csv",
# ]
# model_name = "Qwen3-0.6B"  # ✅ Set the model name manually
# # ============================================================================

# # ====== USER INPUT =======================================================
# csv_files = [
#     "RQ2_RAG_Answers_Basic_RAG_Prompt_Qwen3-4B.csv",
#     "RQ2_RAG_Answers_COT_Prompt_Qwen3-4B.csv",
#     "RQ2_RAG_Answers_No_Context_Prompt_Qwen3-4B.csv",
# ]
# model_name = "Qwen3-4B"  # ✅ Set the model name manually
# # ============================================================================


# # ====== USER INPUT =======================================================
# csv_files = [
#     "RQ2_RAG_Answers_Basic_RAG_Prompt_Llama-3.2-3B-Instruct.csv",
#     "RQ2_RAG_Answers_COT_Prompt_Llama-3.2-3B-Instruct.csv",
#     "RQ2_RAG_Answers_No_Context_Prompt_Llama-3.2-3B-Instruct.csv",
# ]
# model_name = "Llama-3.2-3B-Instruct"  # ✅ Set the model name manually
# # ============================================================================

# # ====== USER INPUT =======================================================
# csv_files = [
#     "RQ2_RAG_Answers_Basic_RAG_Prompt_Phi-3-mini-4k-instruct.csv",
#     "RQ2_RAG_Answers_COT_Prompt_Phi-3-mini-4k-instruct.csv",
#     "RQ2_RAG_Answers_No_Context_Prompt_Phi-3-mini-4k-instruct.csv",
# ]
# model_name = "Phi-3-mini-4k-instruct"  # ✅ Set the model name manually
# # ============================================================================


# ====== USER INPUT =======================================================
csv_files = [
    "RQ2_RAG_Answers_Basic_RAG_Prompt_Phi-3-mini-128k-instruct.csv",
    "RQ2_RAG_Answers_COT_Prompt_Phi-3-mini-128k-instruct.csv",
    "RQ2_RAG_Answers_No_Context_Prompt_Phi-3-mini-128k-instruct.csv",
]
model_name = "Phi-3-mini-128k-instruct"  # ✅ Set the model name manually
# ============================================================================

# Read and concatenate all CSVs
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Verify columns are identical
expected_columns = [
    "Generation Model", "Question Index", "Question", "Type", "Source_QID",
    "Original Chunk", "Chunks Retrieved", "Generated Answer",
    "Reference Answer", "Generation Prompt Used", "Encoding Used", "Enable Thinking"
]
assert combined_df.columns.tolist() == expected_columns, "❌ Column mismatch detected!"

# Generate output filename
output_filename = f"combined_prompt_outputs_{model_name}.csv"
combined_df.to_csv(output_filename, index=False)

print(f"✅ Combined CSV saved as '{output_filename}'")
