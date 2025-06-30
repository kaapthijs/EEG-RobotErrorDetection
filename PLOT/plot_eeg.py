import os
import glob
import numpy as np
import pandas as pd
import mne
from mne import create_info
from mne.preprocessing import ICA, create_eog_epochs
from asrpy import ASR
from scipy.stats import kurtosis
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import signal
from mne.time_frequency import tfr_morlet
import matplotlib.gridspec as gridspec # Import GridSpec

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

CHANNEL_MAPPING = {
    'eeg_ch1': 'Fz', 'eeg_ch2': 'C3', 'eeg_ch3': 'Cz', 'eeg_ch4': 'C4',
    'eeg_ch5': 'Pz', 'eeg_ch6': 'F3', 'eeg_ch7': 'Oz', 'eeg_ch8': 'F4'
}

TASK_NAMES = {
    1: 'REACHING',
    2: 'BLOCK',
    3: 'GRID'
}

def load_all_data(participant_filter=None):
    """Load data from all participants and tasks."""
    base_dir = './DATA'
    all_data = {}
    participants = []
    tasks = []
    
    pattern = os.path.join(base_dir, 'participant_*_task*_merged.csv')
    files = glob.glob(pattern)
    print(f"Found {len(files)} data files")
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        parts = filename.replace('.csv', '').split('_')
        participant = int(parts[1])
        
        if participant_filter and participant not in participant_filter:
            continue

        task = int(parts[2].replace('task', ''))
        
        print(f"Loading Participant {participant}, Task {task}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            eeg_cols = list(CHANNEL_MAPPING.keys())
            eeg_data = df[eeg_cols].to_numpy().T * 1e-6
            ch_names = [CHANNEL_MAPPING[c] for c in eeg_cols]
            
            info = create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg'] * len(ch_names))
            raw = mne.io.RawArray(eeg_data, info)
            raw.set_montage('standard_1020')
            
            events, event_id = extract_events(df, task)
            
            key = f"p{participant}_t{task}"
            all_data[key] = {
                'raw': raw, 'events': events, 'event_id': event_id,
                'participant': participant, 'task': task, 'dataframe': df
            }
            participants.append(participant)
            tasks.append(task)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
            
    return all_data, list(set(participants)), list(set(tasks))

def extract_events(df, task):
    """Extract events based on task type with proper error severity handling"""
    events = []
    event_id = {}
    
    if task == 1:
        event_id = {'non_error': 0, 'error': 1}
        for idx, row in df.iterrows():
            if row.get('correctness') == 'correct':
                events.append((idx, 0, 0))
            elif row.get('correctness') == 'error':
                events.append((idx, 0, 1))
                
    elif task == 2:
        event_id = {'non_error': 0, 'error': 1}
        for idx, row in df.iterrows():
            if row.get('action') == 'correct':
                events.append((idx, 0, 0))
            elif row.get('action') == 'drop':
                events.append((idx, 0, 1))
                
    elif task == 3:
        event_id = {'non_error': 0, 'slight_error': 1, 'severe_error': 2}
        for idx, row in df.iterrows():
            if row.get('correctness') == 'correct':
                events.append((idx, 0, 0))
            elif row.get('correctness') == 'incorrect':
                if row.get('punishment') == 'punishment':
                    events.append((idx, 0, 2))
                else:
                    events.append((idx, 0, 1))
            elif row.get('punishment') == 'punishment':
                events.append((idx, 0, 2))
    
    return np.array(events), event_id

def advanced_preprocess_data(raw):
    """Applies an advanced preprocessing pipeline with ASR and ICA."""
    report = {}
    raw_copy = raw.copy()
    
    raw_copy.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', phase='zero', verbose=False)

    sfreq = raw_copy.info['sfreq']
    asr = ASR(sfreq=sfreq, cutoff=20)
    asr.fit(raw_copy)
    raw_copy = asr.transform(raw_copy)
    report['asr_applied'] = True
    report['asr_cutoff'] = 20
    
    n_eeg_channels = len(mne.pick_types(raw_copy.info, eeg=True, exclude='bads'))
    if n_eeg_channels > 1:
        ica = ICA(n_components=n_eeg_channels - 1, max_iter='auto', random_state=97)
        ica.fit(raw_copy)
        
        eog_epochs = create_eog_epochs(raw_copy, ch_name=['Fz', 'F3', 'F4'], verbose=False)
        
        eog_indices = []
        if len(eog_epochs) > 0:
            eog_ch_name = eog_epochs.ch_names[0]
            eog_indices, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=eog_ch_name, verbose=False)
        
        ica.exclude = eog_indices
        report['ica_excluded_indices'] = eog_indices
        print(f"    ICA: Found and marked {len(eog_indices)} EOG-related components for removal.")
        
        if eog_indices:
            ica.apply(raw_copy, verbose=False)
    else:
        report['ica_excluded_indices'] = []
        print("    Skipping ICA: Not enough EEG channels available.")

    raw_copy.filter(l_freq=1.0, h_freq=20.0, fir_design='firwin', phase='zero', verbose=False)

    eeg_picks = mne.pick_types(raw_copy.info, eeg=True, exclude='bads')
    if len(eeg_picks) > 1:
        kurt_vals = kurtosis(raw_copy.get_data(picks=eeg_picks), axis=1)
        kurt_threshold = np.percentile(kurt_vals, 90)
        bad_ch_indices = np.where(kurt_vals >= kurt_threshold)[0]
        
        bad_ch_names = [raw_copy.ch_names[i] for i in eeg_picks[bad_ch_indices]]
        report['interpolated_channels'] = bad_ch_names
        
        if bad_ch_names:
            raw_copy.info['bads'] = bad_ch_names
            print(f"    Interpolating bad channels: {bad_ch_names}")
            raw_copy.interpolate_bads(reset_bads=True, verbose=False)
    else:
        report['interpolated_channels'] = []

    raw_copy.set_eeg_reference('average', projection=False, verbose=False)
    report['reference'] = 'CAR'

    return raw_copy, report

def create_epochs_from_data(data_dict, apply_preprocessing=True):
    """Create epochs from loaded data with optional advanced preprocessing"""
    epochs_dict = {}
    
    for key, data in data_dict.items():
        raw = data['raw']
        events = data['events']
        event_id = data['event_id']
        task = data['task']
        
        if len(events) == 0:
            print(f"No events found for {key}")
            continue
        
        if apply_preprocessing:
            print(f"    Advanced preprocessing for {key}...")
            raw_preprocessed, rejection_report = advanced_preprocess_data(raw)
        else:
            print(f"    Basic preprocessing for {key}...")
            raw_preprocessed = raw.copy()
            raw_preprocessed.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', phase='zero', verbose=False)
            raw_preprocessed.set_eeg_reference('average', projection=False, verbose=False)
            rejection_report = {'basic_preprocessing': True}
        
        tmax = 0.5 if task == 2 else 1.0
        
        try:
            epochs = mne.Epochs(raw_preprocessed, events, event_id=event_id,  
                                 tmin=-0.2, tmax=tmax,
                                 preload=True, baseline=(-0.2, 0),  
                                 picks='eeg', reject_by_annotation=False, verbose=False)
            
            epochs_dict[key] = {
                'epochs': epochs, 'participant': data['participant'],
                'task': data['task'], 'event_id': event_id,
                'rejection_report': rejection_report
            }
        except Exception as e:
            print(f"Error creating epochs for {key}: {e}")
            continue
    
    return epochs_dict

def compute_grand_averages(epochs_dict):
    """Compute grand averages across all participants for each task"""
    grand_averages = {}
    
    # Organize epochs by task
    task_epochs = {1: [], 2: [], 3: []}
    
    for key, data in epochs_dict.items():
        task = data['task']
        epochs = data['epochs']
        
        if task in task_epochs:
            task_epochs[task].append(epochs)
    
    # Compute grand averages for each task
    for task, epochs_list in task_epochs.items():
        if not epochs_list:
            continue
            
        print(f"Computing grand average for Task {task} ({len(epochs_list)} participants)")
        
        # Combine all epochs for this task
        all_epochs = mne.concatenate_epochs(epochs_list)
        
        if task == 3:
            # For task 3, separate slight and severe errors
            non_error_epochs = all_epochs['non_error']
            slight_error_epochs = all_epochs['slight_error'] if 'slight_error' in all_epochs.event_id else None
            severe_error_epochs = all_epochs['severe_error'] if 'severe_error' in all_epochs.event_id else None
            
            grand_averages[f'task{task}_slight'] = {
                'non_error': non_error_epochs.average() if len(non_error_epochs) > 0 else None,
                'error': slight_error_epochs.average() if slight_error_epochs and len(slight_error_epochs) > 0 else None,
                'task_name': 'GRID-SLIGHT'
            }
            
            grand_averages[f'task{task}_severe'] = {
                'non_error': non_error_epochs.average() if len(non_error_epochs) > 0 else None,
                'error': severe_error_epochs.average() if severe_error_epochs and len(severe_error_epochs) > 0 else None,
                'task_name': 'GRID-SEVERE'
            }
        else:
            # For tasks 1 and 2
            non_error_epochs = all_epochs['non_error']
            error_epochs = all_epochs['error'] if 'error' in all_epochs.event_id else None
            
            grand_averages[f'task{task}'] = {
                'non_error': non_error_epochs.average() if len(non_error_epochs) > 0 else None,
                'error': error_epochs.average() if error_epochs and len(error_epochs) > 0 else None,
                'task_name': TASK_NAMES[task]
            }
    
    return grand_averages

def create_topographic_maps(evoked_error, evoked_non_error, times_to_plot, ax_row):
    """Create topographic maps for specified time points for the difference wave."""
    if evoked_error is None or evoked_non_error is None:
        return
    
    # Calculate difference wave
    evoked_diff = mne.combine_evoked([evoked_error, evoked_non_error], weights=[1, -1])
    
    for i, time_point in enumerate(times_to_plot):
        # Find closest time index
        time_idx = np.argmin(np.abs(evoked_diff.times - time_point))
        actual_time = evoked_diff.times[time_idx]
        
        # Plot topographic map
        im, _ = mne.viz.plot_topomap(evoked_diff.data[:, time_idx], evoked_diff.info, 
                                     axes=ax_row[i], show=False, contours=6,
                                     cmap='RdBu_r', vlim=(-5e-6, 5e-6)) # Fixed vlim for consistent color scale
        ax_row[i].set_title(f'{actual_time*1000:.0f} ms', fontsize=16, pad=1) # Increased title fontsize
        ax_row[i].tick_params(axis='both', labelsize=16) # Increased tick label size

    return evoked_diff

def create_erp_waveform(evoked_error, evoked_non_error, ax, channel='Cz'):
    """Create ERP waveform plot for specified channel"""
    if evoked_error is None or evoked_non_error is None:
        return None, None
    
    # Get channel index
    if channel not in evoked_error.ch_names:
        print(f"Warning: Channel '{channel}' not found in evoked data.")
        return None, None

    ch_idx = evoked_error.ch_names.index(channel)
    times = evoked_error.times * 1000  # Convert to ms
    
    # Get data
    error_data = evoked_error.data[ch_idx, :] * 1e6  # Convert to µV
    non_error_data = evoked_non_error.data[ch_idx, :] * 1e6
    difference_data = error_data - non_error_data
    
    # Plot waveforms
    ax.plot(times, non_error_data, 'b-', linewidth=2, label='non-error')
    ax.plot(times, error_data, 'r-', linewidth=2, label='error')
    ax.plot(times, difference_data, 'k--', linewidth=1.5, label='difference')
    
    # Formatting
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (ms)', fontsize=36) # Increased font size
    ax.set_ylabel('Amplitude (µV)', fontsize=36) # Increased font size
    ax.set_ylim([-5, 5]) # Fixed Y-axis range
    ax.set_xlim([-200, 1000]) # Fixed X-axis range
    ax.tick_params(axis='both', labelsize=36) # Increased tick label size
    # ax.legend(fontsize=10) # Removed legend as requested
    ax.grid(True, alpha=0.3)
    
    return difference_data, times

def create_channel_heatmap(evoked_error, evoked_non_error, ax):
    """Create channel x time heatmap"""
    if evoked_error is None or evoked_non_error is None:
        return
    
    # Calculate difference
    diff_data = (evoked_error.data - evoked_non_error.data) * 1e6  # Convert to µV
    times = evoked_error.times * 1000  # Convert to ms
    
    # Create heatmap
    im = ax.imshow(diff_data, aspect='auto', cmap='RdBu_r', 
                    extent=[times[0], times[-1], len(evoked_error.ch_names), 0],
                    interpolation='nearest', vmin=-5, vmax=5) # Fixed vmin/vmax for consistent color scale
    
    # Set channel labels
    ax.set_yticks(range(len(evoked_error.ch_names)))
    ax.set_yticklabels(evoked_error.ch_names, fontsize=36) # Increased font size
    ax.set_xlabel('Time (ms)', fontsize=36) # Increased font size
    ax.set_ylabel('Channels', fontsize=36) # Increased font size
    ax.set_xlim([-200, 1000]) # Fixed X-axis range
    ax.tick_params(axis='x', labelsize=36) # Increased tick label size for x-axis
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude (µV)', fontsize=36) # Increased font size
    cbar.ax.tick_params(labelsize=36) # Increased colorbar tick label size
    
    return im

def save_topomaps(evoked_error, evoked_non_error, task_name, output_dir='./plots'):
    """Saves topographic maps for a given task to a separate PNG file."""
    if evoked_error is None or evoked_non_error is None:
        print(f"Skipping topomaps for {task_name}: missing evoked data.")
        return

    fig = plt.figure(figsize=(12, 6)) # Adjust figure size for topomaps
    
    # Unified spacing
    unified_spacing = 1.0
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=unified_spacing) 
    
    times_to_plot = [0.1, 0.2, 0.3, 0.4] # Times for topographic maps (in seconds)
    
    ax_topo = []
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax_topo.append(ax)
    
    create_topographic_maps(evoked_error, evoked_non_error, times_to_plot, ax_topo)
    
    # Removed main title for the figure as requested.
    # fig.suptitle(f'{task_name} - Topographic Maps', fontsize=20, y=0.98) 

    os.makedirs(output_dir, exist_ok=True)
    filename = f"topomap_{task_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def save_erp_waveform(evoked_error, evoked_non_error, task_name, channel, output_dir='./plots'):
    """Saves ERP waveform for a given task and channel to a separate PNG file."""
    if evoked_error is None or evoked_non_error is None:
        print(f"Skipping ERP waveform for {task_name} ({channel}): missing evoked data.")
        return None, None

    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size for ERP waveform
    
    difference_data, times = create_erp_waveform(evoked_error, evoked_non_error, ax, channel=channel)
    
    # Removed main title for the figure as requested.
    # fig.suptitle(f'ERP Waveform - {task_name} ({channel})', fontsize=20, y=0.98) 

    plt.tight_layout() # Removed rect as suptitle is gone
    os.makedirs(output_dir, exist_ok=True)
    filename = f"ERP_Waveform_{channel}_{task_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")
    return difference_data, times

def save_channel_heatmap(evoked_error, evoked_non_error, task_name, channel, output_dir='./plots'):
    """Saves channel heatmap for a given task and channel to a separate PNG file."""
    if evoked_error is None or evoked_non_error is None:
        print(f"Skipping channel heatmap for {task_name} ({channel}): missing evoked data.")
        return

    fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size for heatmap

    create_channel_heatmap(evoked_error, evoked_non_error, ax)

    # Removed main title for the figure as requested.
    # fig.suptitle(f'Channel Heatmap - {task_name} ({channel})', fontsize=20, y=0.98)

    plt.tight_layout() # Removed rect as suptitle is gone
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Channel_Heatmap_{channel}_{task_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

def create_comparison_plot(all_difference_data, output_dir='./plots', channel_name='Cz', filename_suffix=''):
    """Create comparison plot across all tasks for a specific channel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) 
    
    colors = {
        'task1': 'blue',
        'task2': 'orange',
        'task3_slight': 'green',
        'task3_severe': 'red'
    }
    
    # Simplified labels
    labels = {
        'task1': 'REACHING',
        'task2': 'BLOCK',
        'task3_slight': 'GRID SLIGHT',
        'task3_severe': 'GRID SEVERE'
    }
    
    # Top plot: Raw comparison
    for task_key, task_data in all_difference_data.items():
        diff_data = task_data['data']
        times = task_data['times']
        if diff_data is not None and times is not None:
            ax1.plot(times, diff_data, color=colors[task_key], 
                     linewidth=2, label=labels[task_key])
    
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (ms)', fontsize=18) # Reduced font size by 50% (36 -> 18)
    ax1.set_ylabel('Amplitude (µV)', fontsize=18) # Reduced font size by 50% (36 -> 18)
    ax1.set_title(f'Task Comparison - Raw Difference Waves ({channel_name})', fontsize=18) # Reduced title font size by 50% (36 -> 18)
    ax1.legend(fontsize=15) # Reduced legend font size by 50% (30 -> 15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, 5]) # Fixed Y-axis range
    ax1.set_xlim([-200, 1000]) # Fixed X-axis range
    ax1.tick_params(axis='both', labelsize=18) # Reduced tick label size by 50% (36 -> 18)
    
    # Bottom plot: Normalized comparison
    for task_key, task_data in all_difference_data.items():
        diff_data = task_data['data']
        times = task_data['times']
        if diff_data is not None and times is not None:
            # Normalize to peak amplitude
            max_amp = np.max(np.abs(diff_data))
            if max_amp > 0:
                normalized_data = diff_data / max_amp
                ax2.plot(times, normalized_data, color=colors[task_key], 
                         linewidth=2, label=labels[task_key])
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time (ms)', fontsize=18) # Reduced font size by 50% (36 -> 18)
    ax2.set_ylabel('Normalized Amplitude', fontsize=18) # Reduced font size by 50% (36 -> 18)
    ax2.set_title(f'Task Comparison - Normalized Difference Waves ({channel_name})', fontsize=18) # Reduced title font size by 50% (36 -> 18)
    ax2.legend(fontsize=15) # Reduced legend font size by 50% (30 -> 15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-200, 1000]) # Fixed X-axis range
    ax2.tick_params(axis='both', labelsize=18) # Reduced tick label size by 50% (36 -> 18)
    
    plt.tight_layout()
    
    # Save comparison plot with channel-specific filename
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'task_comparison_erp{filename_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory

def main():
    """Main function to run ERP analysis and plotting"""
    # Configuration
    preprocessing = False  # Set to False for basic preprocessing only
    output_dir = './plots' 

    print("=== ERP Analysis and Plotting Pipeline ===")
    
    print("\n1. Loading data...")
    # NOTE: You need to have data files in a './DATA' directory, 
    # e.g., 'participant_1_task1_merged.csv'
    all_data, participants, tasks = load_all_data()
    print(f"Loaded data for participants {sorted(participants)} and tasks {sorted(tasks)}")
    
    print(f"\n2. Creating epochs with preprocessing={'advanced' if preprocessing else 'basic'}...")
    epochs_dict = create_epochs_from_data(all_data, apply_preprocessing=preprocessing)
    
    print("\n3. Computing grand averages...")
    grand_averages = compute_grand_averages(epochs_dict)
    
    print("\n4. Creating ERP analysis plots individually...")
    
    # Store difference data for comparison plots
    all_difference_data_cz = {}
    all_difference_data_fz = {}

    # Iterate through each task and then each channel to save individual plots
    tasks_to_plot = ['task1', 'task2', 'task3_slight', 'task3_severe']
    channels_to_plot = ['Cz', 'Fz']

    for task_key in tasks_to_plot:
        if task_key not in grand_averages:
            print(f"Skipping plots for {task_key}: No grand average data found.")
            continue
        
        task_data = grand_averages[task_key]
        evoked_error = task_data['error']
        evoked_non_error = task_data['non_error']
        task_name = task_data['task_name']

        # Save Topomaps
        save_topomaps(evoked_error, evoked_non_error, task_name, output_dir)

        for channel in channels_to_plot:
            # Save ERP Waveform
            diff_data, times = save_erp_waveform(evoked_error, evoked_non_error, task_name, channel, output_dir)
            if channel == 'Cz' and diff_data is not None:
                all_difference_data_cz[task_key] = {'data': diff_data, 'times': times}
            elif channel == 'Fz' and diff_data is not None:
                all_difference_data_fz[task_key] = {'data': diff_data, 'times': times}

            # Save Channel Heatmap
            save_channel_heatmap(evoked_error, evoked_non_error, task_name, channel, output_dir)
    
    print("\n5. Creating comparison plots...")
    # Create comparison plots for Cz and Fz separately
    if all_difference_data_cz:
        create_comparison_plot(all_difference_data_cz, output_dir, channel_name='Cz', filename_suffix='_cz')
    if all_difference_data_fz:
        create_comparison_plot(all_difference_data_fz, output_dir, channel_name='Fz', filename_suffix='_fz')
    
    print("\nPlotting complete! Check the './plots' directory for output files.")
    
    # Print summary (using Cz data as a representative, as it's typically the primary focus)
    print("\n=== SUMMARY ===")
    for task_key, data in grand_averages.items():
        if data['error'] is not None and data['non_error'] is not None:
            n_error = data['error'].nave if hasattr(data['error'], 'nave') else 'N/A'
            n_non_error = data['non_error'].nave if hasattr(data['non_error'], 'nave') else 'N/A'
            print(f"{data['task_name']}: Error epochs={n_error}, Non-error epochs={n_non_error}")

if __name__ == '__main__':
    main()

