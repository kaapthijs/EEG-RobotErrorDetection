import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Prepare the data with AUC
data = [
    ("Task1", "P10", 0.360, 0.222, 0.498, 0.264),
    ("Task1", "P11", 0.497, 0.289, 0.705, 0.502),
    ("Task1", "P13", 0.420, 0.311, 0.529, 0.447),
    ("Task1", "P14", 0.536, 0.322, 0.749, 0.621),
    ("Task1", "P15", 0.479, 0.311, 0.648, 0.457),
    ("Task1", "P18", 0.531, 0.406, 0.656, 0.521),
    ("Task1", "P1", 0.513, 0.444, 0.581, 0.556),
    ("Task1", "P21", 0.614, 0.528, 0.700, 0.608),
    ("Task1", "P24", 0.639, 0.511, 0.767, 0.739),
    ("Task1", "P25", 0.436, 0.267, 0.605, 0.352),
    ("Task1", "P26", 0.544, 0.267, 0.822, 0.607),
    ("Task1", "P2", 0.437, 0.200, 0.674, 0.467),
    ("Task1", "P3", 0.398, 0.244, 0.551, 0.322),
    ("Task1", "P5", 0.578, 0.522, 0.633, 0.633),
    ("Task1", "P6", 0.464, 0.356, 0.573, 0.456),
    ("Task1", "P8", 0.592, 0.367, 0.817, 0.713),

    ("Task2", "P10", 0.414, 0.317, 0.511, 0.414),
    ("Task2", "P11", 0.442, 0.417, 0.467, 0.428),
    ("Task2", "P13", 0.580, 0.506, 0.654, 0.591),
    ("Task2", "P14", 0.659, 0.597, 0.722, 0.723),
    ("Task2", "P15", 0.581, 0.583, 0.578, 0.622),
    ("Task2", "P18", 0.494, 0.422, 0.565, 0.444),
    ("Task2", "P1", 0.787, 0.667, 0.908, 0.967),
    ("Task2", "P21", 0.592, 0.489, 0.695, 0.635),
    ("Task2", "P24", 0.871, 0.883, 0.859, 0.962),
    ("Task2", "P25", 0.666, 0.544, 0.788, 0.697),
    ("Task2", "P26", 0.497, 0.289, 0.705, 0.533),
    ("Task2", "P2", 0.387, 0.156, 0.619, 0.346),
    ("Task2", "P3", 0.730, 0.622, 0.838, 0.791),
    ("Task2", "P5", 0.886, 0.833, 0.938, 0.947),
    ("Task2", "P6", 0.445, 0.250, 0.640, 0.362),
    ("Task2", "P8", 0.437, 0.000, 0.875, 0.042),

    ("Task3_Slight", "P10", 0.590, 0.529, 0.652, 0.623),
    ("Task3_Slight", "P11", 0.559, 0.446, 0.672, 0.559),
    ("Task3_Slight", "P13", 0.527, 0.399, 0.656, 0.545),
    ("Task3_Slight", "P14", 0.637, 0.553, 0.721, 0.699),
    ("Task3_Slight", "P15", 0.570, 0.467, 0.673, 0.559),
    ("Task3_Slight", "P18", 0.551, 0.413, 0.689, 0.560),
    ("Task3_Slight", "P1", 0.568, 0.478, 0.658, 0.601),
    ("Task3_Slight", "P21", 0.548, 0.439, 0.656, 0.591),
    ("Task3_Slight", "P24", 0.526, 0.388, 0.664, 0.539),
    ("Task3_Slight", "P25", 0.639, 0.626, 0.652, 0.672),
    ("Task3_Slight", "P26", 0.503, 0.368, 0.638, 0.480),
    ("Task3_Slight", "P2", 0.534, 0.444, 0.623, 0.510),
    ("Task3_Slight", "P3", 0.657, 0.599, 0.715, 0.722),
    ("Task3_Slight", "P5", 0.628, 0.532, 0.724, 0.655),
    ("Task3_Slight", "P6", 0.453, 0.295, 0.610, 0.440),
    ("Task3_Slight", "P8", 0.669, 0.610, 0.728, 0.731),

    ("Task3_Severe", "P10", 0.436, 0.067, 0.806, 0.419),
    ("Task3_Severe", "P11", 0.591, 0.378, 0.804, 0.585),
    ("Task3_Severe", "P13", 0.511, 0.222, 0.800, 0.461),
    ("Task3_Severe", "P14", 0.611, 0.333, 0.889, 0.721),
    ("Task3_Severe", "P15", 0.469, 0.067, 0.872, 0.311),
    ("Task3_Severe", "P18", 0.540, 0.200, 0.881, 0.565),
    ("Task3_Severe", "P1", 0.560, 0.267, 0.853, 0.638),
    ("Task3_Severe", "P21", 0.512, 0.144, 0.880, 0.551),
    ("Task3_Severe", "P24", 0.597, 0.333, 0.860, 0.664),
    ("Task3_Severe", "P25", 0.503, 0.100, 0.905, 0.630),
    ("Task3_Severe", "P26", 0.538, 0.200, 0.875, 0.588),
    ("Task3_Severe", "P2", 0.572, 0.356, 0.788, 0.626),
    ("Task3_Severe", "P3", 0.470, 0.033, 0.907, 0.607),
    ("Task3_Severe", "P5", 0.731, 0.533, 0.929, 0.888),
    ("Task3_Severe", "P6", 0.523, 0.167, 0.880, 0.609),
    ("Task3_Severe", "P8", 0.488, 0.133, 0.843, 0.561),
]

# Set a global font size multiplier
plt.rcParams.update({'font.size': 30}) # Global font size increase

# Extract unique participants and sort numerically
participants = sorted(set(p for _, p, *_, in data), key=lambda x: int(x[1:]))
participant_indices = {p: i for i, p in enumerate(participants)}

# Organize data by metric and task
metrics = ['Balanced Accuracy', 'AUC', 'TPR', 'TNR']
# Renamed tasks
tasks = ['Reaching', 'Block', 'Grid_Slight', 'Grid_Severe']
# Map original task names to new ones for data processing
original_tasks = ['Task1', 'Task2', 'Task3_Slight', 'Task3_Severe']

colors = ["#00a2ff", "#ffd900", "#2bd800", "#ff0000"]  # Distinct colors for each task

# Initialize data storage
metric_data = {metric: np.zeros((len(participants), len(tasks))) for metric in metrics}
task_means = {metric: [] for metric in metrics}

# Process data
for task, part, bal, tpr_val, tnr_val, auc_val in data:
    p_idx = participant_indices[part]
    
    # Map original task name to its new index
    if task == 'Task1':
        t_idx = tasks.index('Reaching')
    elif task == 'Task2':
        t_idx = tasks.index('Block')
    elif task == 'Task3_Slight':
        t_idx = tasks.index('Grid_Slight')
    elif task == 'Task3_Severe':
        t_idx = tasks.index('Grid_Severe')
    else:
        continue # Skip if task name is unexpected
    
    metric_data['Balanced Accuracy'][p_idx, t_idx] = bal
    metric_data['AUC'][p_idx, t_idx] = auc_val
    metric_data['TPR'][p_idx, t_idx] = tpr_val
    metric_data['TNR'][p_idx, t_idx] = tnr_val

# Calculate means for each metric-task combination
for metric in metrics:
    for t_idx in range(len(tasks)):
        task_values = metric_data[metric][:, t_idx]
        non_zero = task_values[task_values != 0]  # Exclude uninitialized zeros
        task_means[metric].append(np.mean(non_zero) if len(non_zero) > 0 else 0)

# Set up plot with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(20, 30)) 
x = np.arange(len(participants))
width = 0.2  # Bar width

# Define parameters for mean labels positioned at the top right
mean_label_x_pos = 0.98 # X-position as a fraction of subplot width (0.98 for far right)
mean_label_y_start = 0.95 # Starting Y-position for the first label (top)
mean_label_y_spacing = 0.1 # Vertical spacing between labels (as a fraction of subplot height)

# Plot each metric
for ax_idx, metric in enumerate(metrics):
    ax = axes[ax_idx]
    # Plot bars for each task
    for t_idx, task in enumerate(tasks):
        offset = width * (t_idx - 1.5)  # Position offset for each task
        values = metric_data[metric][:, t_idx] * 100  # Convert to percentage
        ax.bar(x + offset, values, width, color=colors[t_idx], label=task, edgecolor='black')
        
        # Add mean line with increased linewidth
        mean_val = task_means[metric][t_idx] * 100
        # Line extends nearly to the edge where the text will start
        ax.axhline(mean_val, color=colors[t_idx], linestyle='--', linewidth=3, alpha=0.7, xmax=0.95)
        
    # Add mean labels to the top right of the subplot, stacked vertically, with transparent white background
    for t_idx, task in enumerate(tasks):
        mean_val = task_means[metric][t_idx] * 100
        # Calculate y-position based on starting point and spacing, using ax.transAxes
        y_position_for_label = mean_label_y_start - (t_idx * mean_label_y_spacing)
        
        ax.text(mean_label_x_pos, y_position_for_label, f'{mean_val:.1f}%', 
                color=colors[t_idx], va='top', ha='right', fontsize=27, 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3', edgecolor='none')) # Added transparent white background
    
    ax.set_ylabel('Accuracy [%]', fontsize=30) 
    ax.set_title(f'{metric}', fontsize=36)
    ax.grid(axis='y', alpha=0.4)
    
    # Set y-axis limits for specific metrics
    if metric in ['Balanced Accuracy', 'AUC', 'TNR']:
        ax.set_ylim(40, 100)
    else:  # TPR
        ax.set_ylim(0, 100)
    
    # Set x-axis tick labels for all subplots (participant numbers)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='center', fontsize=22) 

# Add legend and adjust layout
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=4, fontsize=28) 
plt.tight_layout(rect=[0, 0, 1, 0.95]) 

# Save as PNG
plt.savefig('task_metrics_comparison_with_auc_larger_text.png', dpi=150, bbox_inches='tight')
