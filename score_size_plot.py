# import matplotlib.pyplot as plt
# import numpy as np # Library for numerical operations, particularly with arrays

# # Data
# models = [
#     "Qwen3-0.6B",
#     "Qwen3-1.7B",
#     "Qwen3-4B",
#     "Qwen3-8B"
# ]
# correctness = [3.67, 3.91, 4.14, 4.21]
# completeness_ref = [3.06, 3.5, 3.64, 3.62]
# faithfulness = [3.98, 4.15, 4.35, 4.48]
# completeness_q = [4.19, 4.43, 4.49, 4.53]

# # Calculate average score per model
# avg_scores = np.mean([
#     correctness,
#     completeness_ref,
#     faithfulness,
#     completeness_q
# ], axis=0)

# # Plot
# plt.figure(figsize=(10, 6)) # Width and height of the figure in inches
# plt.plot(models, correctness, marker='o', label='Correctness') # x axis is models, y-axis is scores and each point is marked with 'o'
# plt.plot(models, completeness_ref, marker='o', label='Completeness (Ref)')
# plt.plot(models, faithfulness, marker='o', label='Faithfulness')
# plt.plot(models, completeness_q, marker='o', label='Coverage (Question)')
# plt.plot(models, avg_scores, marker='o', linestyle='--', color='black', linewidth=2, label='Average Score') # Line for the average score across all metrics


# plt.xlabel('Model Size')
# plt.ylabel('Score (1 to 5)')
# plt.title('LLM-as-Judge Scores by Model Size')
# plt.ylim(1, 5) # Sets the limits for the y-axis
# plt.grid(True) # Adds a grid to the plot, making it easier to read the values on the axes.
# plt.legend()
# plt.tight_layout() # Adjusts the plot to provide reasonable spacing between elements, preventing labels and titles from overlapping.

# # # Display plot
# # plt.show()
# # Save plot as PNG
# plt.savefig('graph_score_size.png', dpi=300)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CHANGE ACCORDING TO FILE YOU WANT TO PLOT
# df = pd.read_csv('size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v3.csv')
# df = pd.read_csv('size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2_added_model.csv')
# df = pd.read_csv('size_performance_Llama-3.2_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('size_en_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('size_performance_Phi-3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW_size_performance_Llama-3.2_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW_size_performance_Phi-3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW_size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW_size_en_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW2_size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW2_size_en_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW2_size_performance_Phi-3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW2_size_performance_Llama-3.2_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW3_size_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW3_size_performance_Phi-3_summary_Basic_RAG_Prompt_prompt_v2.csv')
# df = pd.read_csv('NEW3_size_en_performance_Qwen3_summary_Basic_RAG_Prompt_prompt_v2.csv')
df = pd.read_csv('NEW3_size_performance_Llama-3.2_summary_Basic_RAG_Prompt_prompt_v2.csv')

# Clean model names for x-axis labels (optional)
df['Model'] = df['Generation Model'].apply(lambda x: x.split('/')[-1])

# # Enforce desired model order
# desired_order = [
#     "Phi-3-mini-128k-instruct",
#     "Phi-3-small-128k-instruct",
#     "Phi-3-medium-128k-instruct"
# ]
# df = df[df['Model'].isin(desired_order)]
# df['Model'] = pd.Categorical(df['Model'], categories=desired_order, ordered=True)
# df = df.sort_values('Model')


# Rename columns for display
metric_mapping = {
    'correctness': 'Correctness',
    'completeness_reference': 'Completeness (Ref)',
    'faithfulness': 'Faithfulness',
    'completeness_question': 'Coverage (Question)'
}
df.rename(columns=metric_mapping, inplace=True)

# Extract models and metrics
models = df['Model'].tolist()
metric_names = list(metric_mapping.values())
metric_values = [df[metric].tolist() for metric in metric_names]

# Compute average score per model
avg_scores = np.mean(metric_values, axis=0)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each metric line
for name, values in zip(metric_names, metric_values):
    plt.plot(models, values, marker='o', label=name)

# Plot average line
plt.plot(models, avg_scores, marker='o', linestyle='--', color='black', linewidth=2, label='Average Score')

# Axis labels and formatting
plt.xlabel('Model Size')
plt.ylabel('Score (1 to 5)')
plt.title('LLM-as-Judge Scores by Model Size')
plt.ylim(1, 5)
plt.margins(x=0.1)
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()

# CHANGE OUTPUT FILE NAME
# Save plot
# plt.savefig('size_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('size_Qwen3_graph_Basic_RAG_Prompt_prompt_v3.png', dpi=300) 
# plt.savefig('size_Qwen3_graph_Basic_RAG_Prompt_prompt_v2_added_model.png', dpi=300) 
# plt.savefig('size_Llama-3.2_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300) 
# plt.savefig('size_en_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300) 
# plt.savefig('size_Phi-3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300) 
# plt.savefig('NEW_size_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW_size_Llama-3.2_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW_size_Phi-3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW_size_en_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300) 
# plt.savefig('NEW2_size_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW2_size_en_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# # plt.savefig('NEW2_size_Phi-3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW2_size_Llama-3.2_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW3_size_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW3_size_Phi-3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
# plt.savefig('NEW3_size_en_Qwen3_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)
plt.savefig('NEW3_size_Llama-3.2_graph_Basic_RAG_Prompt_prompt_v2.png', dpi=300)

print("Done!")