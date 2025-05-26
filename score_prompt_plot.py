# import matplotlib.pyplot as plt # Library which provides functions for creating plots
# import numpy as np # Library for numerical operations, particularly with arrays

# # Data
# prompts = ["Basic_RAG_Prompt", "COT_Prompt", "No_Context_Prompt"]
# metrics = ["correctness", "completeness_reference", "faithfulness", "completeness_question"]

# # Scores per prompt (in the same order as `prompts`)
# scores = {
#     "correctness": [3.66, 3.91, 2.7],
#     "completeness_reference": [3.06, 3.32, 1.98],
#     "faithfulness": [4.0, 4.01, 3.11],
#     "completeness_question": [4.19, 4.38, 3.48]
# }

# # Set up bar width and positions
# x = np.arange(len(prompts))  # Creates a NumPy array x with values [0, 1, 2], representing the x-axis positions for the prompt categories
# width = 0.2  # Width of each bar

# # Create the plot with width and height
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot bars for each metric, the offset is adjusted to create the grouped bar chart effect.
# ax.bar(x - 1.5*width, scores["correctness"], width, label="Correctness")
# ax.bar(x - 0.5*width, scores["completeness_reference"], width, label="Completeness (Ref)")
# ax.bar(x + 0.5*width, scores["faithfulness"], width, label="Faithfulness")
# ax.bar(x + 1.5*width, scores["completeness_question"], width, label="Coverage (Question)")

# # Formatting
# ax.set_xlabel('Generation Prompt Used')
# ax.set_ylabel('Score (1 to 5)')
# ax.set_title('LLM-as-Judge Scores by Prompt')
# ax.set_xticks(x) # Sets the positions of the x-axis ticks according to the x array created earlier
# ax.set_xticklabels(prompts, rotation=15)
# ax.set_ylim(1, 5)
# ax.legend()
# ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# plt.tight_layout()

# # Save and/or show
# plt.savefig("prompt_comparison_bargraph.png", dpi=300)
# # plt.show()  # Uncomment this if running locally with GUI support


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd  # Library to read CSV files

# # CHANGE ACCORDING TO FILE YOU WANT TO PLOT
# csv_path = "prompt_performance_summary_Qwen3-0.6B_prompt_v2.csv"
# df = pd.read_csv(csv_path)

# # Extract data
# prompts = df["prompt"].tolist()
# metrics = ["correctness", "completeness_reference", "faithfulness", "completeness_question"]
# x = np.arange(len(prompts))
# width = 0.2

# # Set up the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot each metric as a group of bars
# for i, metric in enumerate(metrics):
#     offset = (i - 1.5) * width  # Adjust the offset for grouping
#     ax.bar(x + offset, df[metric], width, label=metric.replace("_", " ").capitalize())

# # Formatting
# ax.set_xlabel('Generation Prompt Used')
# ax.set_ylabel('Score (1 to 5)')
# ax.set_title('LLM-as-Judge Scores by Prompt')
# ax.set_xticks(x)
# ax.set_xticklabels(prompts, rotation=15)
# ax.set_ylim(1, 5)
# ax.legend()
# ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# plt.tight_layout()


# plt.savefig("prompt_bargraph_Qwen3-0.6B_prompt_v2.png", dpi=300) # CHANGE OUTPUT FILE NAME

# print("Done!")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Library to read CSV files


# CHANGE ACCORDING TO FILE YOU WANT TO PLOT
# csv_path = "prompt_performance_summary_Qwen3-0.6B_prompt_v2.csv"
csv_path = "prompt_en_performance_summary_Qwen3-0.6B_prompt_v2.csv"

df = pd.read_csv(csv_path)

# Reorder prompts manually
desired_order = ["No_Context_Prompt", "Basic_RAG_Prompt", "COT_Prompt"]
df = df.set_index("Generation Prompt Used").loc[desired_order].reset_index()

# Add average row
average_row = {
    "Generation Prompt Used": "Average",
}
metrics = ["correctness", "completeness_reference", "faithfulness", "completeness_question"]

# Add column for average score per prompt
df["Average Score"] = df[metrics].mean(axis=1)

# Prepare data for plotting
prompts = df["Generation Prompt Used"].tolist()
x = np.arange(len(prompts))
width = 0.18

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each metric as a group of bars
for i, metric in enumerate(metrics + ["Average Score"]):
    offset = (i - 2) * width  # Center the bars
    label = metric.replace("_", " ").capitalize()
    ax.bar(x + offset, df[metric], width, label=label)

# Formatting
ax.set_xlabel('Generation Prompt Used',labelpad=10)
ax.set_ylabel('Score (1 to 5)')
ax.set_title('LLM-as-Judge Scores by Prompt')
ax.set_xticks(x)
ax.set_xticklabels(prompts, rotation=0)
ax.set_ylim(1, 5)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# CHANGE OUTPUT FILE NAME ACCORDING TO YOU ARE TESTING
# plt.savefig("prompt_bargraph_Qwen3-0.6B_prompt_v2.png", dpi=300) 
plt.savefig("prompt_en_bargraph_Qwen3-0.6B_prompt_v2.png", dpi=300) 

print("Done!")
