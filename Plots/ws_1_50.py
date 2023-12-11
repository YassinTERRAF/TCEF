import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for each dataset
grid_data = {
    "GTCC": [73.55, 76.04, 78.00, 79.62, 80.88, 82.15, 83.96, 84.47, 84.82, 85.62, 
    86.13, 87.79, 89.05, 89.82, 90.71, 90.76,90.75,90.74],
    "MFCC": [70.08, 73.54, 75.52, 77.47, 79.08, 79.90, 81.47, 82.59, 83.53, 84.04,
     84.33, 86.31, 87.77, 88.96, 89.72, 89.78,89.76,89.74],
    "PNCC": [61.22, 65.05, 67.85, 69.79, 71.82, 73.71, 74.07, 75.14, 76.26, 77.73, 
    79.01, 80.55, 82.57, 83.46, 84.74, 84.77,84.73,84.76]
}

grid_nr_data = {
    "GTCC": [56.69, 58.86, 61.72, 63.41, 64.63, 66.25, 67.93, 69.49, 70.06, 71.24, 
    72.10, 73.45, 75.48, 77.22, 78.28, 78.34,78.23,78.25],
    "MFCC": [43.10,47.08,49.54,52.28,53.96,54.90,57.03,57.75,58.73,59.39, 
    61.62,  64.10, 66.24,  67.63, 67.97, 68.08,68.0,67.99],
    "PNCC": [44.97, 49.01, 50.72, 52.89, 56.02, 57.33, 58.35, 59.18 , 59.89, 61.09, 
    63.51, 64.84, 66.55, 67.80, 68.75, 68.86, 68.83, 68.8]
}

ravdess_data = {
    "GTCC": [61.03, 62.77, 66.11, 69.61, 71.72, 73.11, 75.20, 76.77, 77.56, 78.78, 
    84.08, 88.03, 91.60, 92.68, 94.28, 94.7, 94.65, 94.67],
    "MFCC": [57.37, 62.03, 65.74, 69.40, 71.72, 73.38, 75.94,77.49, 78.33, 79.41, 
    85.39, 88.72, 91.63, 93.21, 94.10, 94.53,  94.55 , 94.5],
    "PNCC": [52.16, 56.26, 59.42, 64.12, 67.03, 69.22, 71.09, 71.32,72.03,75.53, 
     79.60, 84.85, 87.22,  91.03, 92.24, 92.23, 92.20,92.22]
}

# Completing the RAVDESS-NR dataset data
ravdess_nr_data = {
    "GTCC": [48.09, 50.02, 53.07, 56.92, 58.52, 60.16,63.27, 65.55, 66.28, 68.01, 
    72.86, 79.69, 81.06, 83.79, 84.97, 85.03,84.94,84.93],
    "MFCC": [41.01, 46.66, 48.38, 53.26, 56.72, 57.93, 59.41, 62.88, 64.09, 66.93, 
    71.84, 74.78, 78.80, 82.70,83.01, 83.07, 83.17,83.05],
    "PNCC": [40.44, 44.37, 46.59, 50.36, 53.19, 56.33, 56.34, 58.66, 61.01, 63.33, 
   68.79, 73.26, 77.03, 80.31, 81.55, 81.52,81.8,81.7]
}




# Creating DataFrames with interpolation
grid_df = pd.DataFrame(grid_data, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40,45,50]).interpolate()
grid_nr_df = pd.DataFrame(grid_nr_data, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40,45,50]).interpolate()
ravdess_df = pd.DataFrame(ravdess_data, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40,45,50]).interpolate()
ravdess_nr_df = pd.DataFrame(ravdess_nr_data, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40,45,50]).interpolate()

# Define the layout of subplots
fig, axs = plt.subplots(2, 2, figsize=(25, 15))  # Adjust the figsize as needed
label_fontweight = 'bold'

# Customizing x-axis ticks
x_ticks = [1] + list(range(5, 55, 5))

# Plotting each dataset
sns.lineplot(data=grid_df, markers=True, dashes=False, ax=axs[0, 0])
axs[0, 0].set_title("GRID Dataset", fontsize=20,fontweight=label_fontweight)
axs[0, 0].set_xticks(x_ticks)
axs[0, 0].grid(True)
axs[0, 0].set_ylim(40, 100)  # Set y-axis limits

sns.lineplot(data=grid_nr_df, markers=True, dashes=False, ax=axs[0, 1])
axs[0, 1].set_title("GRID-NR Dataset", fontsize=20,fontweight=label_fontweight)
axs[0, 1].set_xticks(x_ticks)
axs[0, 1].grid(True)
axs[0, 0].set_ylim(40, 100)  # Set y-axis limits

sns.lineplot(data=ravdess_df, markers=True, dashes=False, ax=axs[1, 0])
axs[1, 0].set_title("RAVDESS Dataset", fontsize=20,fontweight=label_fontweight)
axs[1, 0].set_xticks(x_ticks)
axs[1, 0].grid(True)
axs[0, 0].set_ylim(40, 100)  # Set y-axis limits

sns.lineplot(data=ravdess_nr_df, markers=True, dashes=False, ax=axs[1, 1])
axs[1, 1].set_title("RAVDESS-NR Dataset", fontsize=20,fontweight=label_fontweight)
axs[1, 1].set_xticks(x_ticks)
axs[1, 1].grid(True)
axs[0, 0].set_ylim(40, 100)  # Set y-axis limits

# Common settings for all subplots
for ax in axs.flat:
    ax.set_xlabel("Window Size", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.legend(fontsize=18)

plt.tight_layout()

# Increase space between the rows
plt.subplots_adjust(hspace=0.3)  # Adjust this value as needed for better spacing
plt.subplots_adjust(wspace=0.3)

plt.savefig("1_50.png")
plt.close()