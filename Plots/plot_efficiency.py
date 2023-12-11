import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

legend_properties = {'weight':'bold','size':30}

# Common approach names
approaches = [
    "TCEF with derivatives",
    "Conventional Features with TCEF with Conventional Features Derivatives",
    "Conventional Features with TCEF with TCEF Features Derivatives", "Conventional Features with TCEF with Conventional Features and TCEF Derivatives"
]

# Data for GRID dataset
data_GRID = {
    'Feature Technique': ['GTCC'] * 4 + ['MFCC'] * 4 + ['PNCC'] * 4,
    'Approach': approaches * 3,
    'Average Converge Time (s)': [
        267 * 14,  384 * 24, 384 * 21, 466 * 25,
        267 * 18,  384 * 20, 384 * 22, 466 * 24,
        267 * 15,  384 * 20, 384 * 20, 466 * 22
    ]
}

# Data for GRID-NR dataset
data_GRID_NR = {
    'Feature Technique': ['GTCC'] * 4 + ['MFCC'] * 4 + ['PNCC'] * 4,
    'Approach': approaches * 3,
    'Average Converge Time (s)': [
       267 * 19, 384 * 18, 384 * 15, 466 * 16,
       267 * 15, 384 * 17, 384 * 18, 466 * 20,
       267 * 17, 384 * 20, 384 * 19, 466 * 22
    ]
}

# Data for RAVDESS dataset
data_RAVDESS = {
    'Feature Technique': ['GTCC'] * 4 + ['MFCC'] * 4 + ['PNCC'] * 4,
    'Approach': approaches * 3,
    'Average Converge Time (s)': [
         24 * 19,  27 * 30, 23 * 30, 49 * 22,
         24 * 20,  30 * 25, 30 * 27, 49 * 24,
         24 * 17,  30 * 34, 30 * 23, 49 * 25
    ]
}

# Data for RAVDESS-NR dataset
data_RAVDESS_NR = {
    'Feature Technique': ['GTCC'] * 4 + ['MFCC'] * 4 + ['PNCC'] * 4,
    'Approach': approaches * 3,
    'Average Converge Time (s)': [
      24 * 15,   30 * 22, 30 * 24, 49 * 23,
       24 * 19,  30 * 24, 30 * 19, 49 * 17,
       24 * 19,  30 * 24, 30 * 21, 49 * 23
    ]
}

# Convert to DataFrames and convert time to minutes
df_GRID = pd.DataFrame(data_GRID)
df_GRID_NR = pd.DataFrame(data_GRID_NR)
df_RAVDESS = pd.DataFrame(data_RAVDESS)
df_RAVDESS_NR = pd.DataFrame(data_RAVDESS_NR)

for df in [df_GRID, df_GRID_NR, df_RAVDESS, df_RAVDESS_NR]:
    df['Average Converge Time (min)'] = df['Average Converge Time (s)'] / 60

# Set the style of the plot
sns.set(style="whitegrid")

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(60, 40))

# Define font size for titles and labels
title_fontsize = 40
label_fontsize = 35
label_fontweight = 'bold'

# Create the bar plots with enhanced readability settings
plot_params = {
    'x': 'Feature Technique',
    'y': 'Average Converge Time (min)',
    'hue': 'Approach',
    'palette': 'tab10'
}

sns.barplot(ax=axes[0, 0], data=df_GRID, **plot_params)
axes[0, 0].set_title('GRID Dataset', fontsize=title_fontsize,fontweight=label_fontweight)
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Average Converge Time (min)', fontsize=label_fontsize)
axes[0, 0].tick_params(labelsize=label_fontsize)

sns.barplot(ax=axes[0, 1], data=df_GRID_NR, **plot_params)
axes[0, 1].set_title('GRID-NR Dataset', fontsize=title_fontsize,fontweight=label_fontweight)
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Average Converge Time (min)', fontsize=label_fontsize)
axes[0, 1].tick_params(labelsize=label_fontsize)

sns.barplot(ax=axes[1, 0], data=df_RAVDESS, **plot_params)
axes[1, 0].set_title('RAVDESS Dataset', fontsize=title_fontsize,fontweight=label_fontweight)
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Average Converge Time (min)', fontsize=label_fontsize)
axes[1, 0].tick_params(labelsize=label_fontsize)

sns.barplot(ax=axes[1, 1], data=df_RAVDESS_NR, **plot_params)
axes[1, 1].set_title('RAVDESS-NR Dataset', fontsize=title_fontsize,fontweight=label_fontweight)
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Average Converge Time (min)', fontsize=label_fontsize)
axes[1, 1].tick_params(labelsize=label_fontsize)

# Fine-tuning the layout
plt.tight_layout()

# Increase space between the rows
plt.subplots_adjust(hspace=0.3)  # Adjust this value as needed for better spacing
plt.subplots_adjust(wspace=0.3)

# Place the legend
handles, labels = axes[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=28,prop=legend_properties)

# Remove the legends from individual plots
for ax in axes.flat:
    ax.get_legend().remove()

# Improve layout to prevent clipping of ylabel and save the plot
plt.subplots_adjust(bottom=0.15)
plt.savefig("effeciency.png", dpi=300, bbox_inches='tight')