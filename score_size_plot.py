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
df = pd.read_csv('size_performance_summary_Basic_RAG_Prompt_prompt_v3.csv')

# Clean model names for x-axis labels (optional)
df['Model'] = df['Generation Model'].apply(lambda x: x.split('/')[-1])

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
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig('size_graph_Basic_RAG_Prompt_prompt_v3.png', dpi=300) # CHANGE OUTPUT FILE NAME

print("Done!")