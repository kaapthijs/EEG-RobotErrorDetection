import os
import glob
import numpy as np
import pandas as pd
import mne
from mne import create_info
from mne.preprocessing import ICA, create_eog_epochs
from asrpy import ASR
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import warnings
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split')

CHANNEL_MAPPING = {
    'eeg_ch1': 'Fz', 'eeg_ch2': 'C3', 'eeg_ch3': 'Cz', 'eeg_ch4': 'C4',
    'eeg_ch5': 'Pz', 'eeg_ch6': 'F3', 'eeg_ch7': 'Oz', 'eeg_ch8': 'F4'
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

def create_epochs_from_data(data_dict):
    """Create epochs from loaded data with advanced preprocessing"""
    epochs_dict = {}
    
    for key, data in data_dict.items():
        raw = data['raw']
        events = data['events']
        event_id = data['event_id']
        task = data['task']
        
        if len(events) == 0:
            print(f"No events found for {key}")
            continue
        
        print(f"  Advanced preprocessing for {key}...")
        raw_preprocessed, rejection_report = advanced_preprocess_data(raw)
        
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

def assess_data_quality(epochs_dict):
    """Assess data quality and identify problematic participants"""
    quality_metrics = {}
    bad_participants = set()
    
    print("\n=== Data Quality Assessment ===")
    
    for key, data in epochs_dict.items():
        epochs = data['epochs']
        participant = data['participant']
        
        epoched_data = epochs.get_data()
        max_amplitude = np.max(np.abs(epoched_data)) * 1e6
        mean_stability = np.mean(np.std(epoched_data, axis=2)) * 1e6
        mean_kurtosis = np.mean([kurtosis(ch) for ch in epoched_data.reshape(-1, epoched_data.shape[-1])])
        n_epochs = len(epochs)
        labels = epochs.events[:, -1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        quality_metrics[key] = {
            'max_amplitude_uV': max_amplitude, 'mean_stability_uV': mean_stability,
            'mean_kurtosis': mean_kurtosis, 'n_epochs': n_epochs,
            'min_class_ratio': np.min(counts) / np.sum(counts) if len(counts) > 1 else 0,
            'n_classes': len(unique_labels), 'rejection_report': data['rejection_report']
        }
        
        is_bad = False
        reasons = []
        if max_amplitude > 500:  # Relaxed threshold for participant-specific models
            is_bad = True
            reasons.append(f"extreme amplitude ({max_amplitude:.1f}µV)")
        if abs(mean_kurtosis) > 8:  # Relaxed threshold
            is_bad = True
            reasons.append(f"extreme kurtosis ({mean_kurtosis:.1f})")
        if n_epochs < 15:  # Relaxed threshold
            is_bad = True
            reasons.append(f"too few epochs ({n_epochs})")
        if len(unique_labels) < 2:
            is_bad = True
            reasons.append("missing error class")
        
        print(f"{key}: Amp={max_amplitude:.1f}µV, Stab={mean_stability:.1f}µV, Kurt={mean_kurtosis:.1f}, Epochs={n_epochs}, Classes={len(unique_labels)}")
        if is_bad:
            bad_participants.add(participant)
            print(f"  -> FLAGGED: {', '.join(reasons)}")
            
    print(f"\nBad participants identified: {sorted(list(bad_participants))}")
    return quality_metrics, bad_participants

def extract_features(epochs):
    """Extracts temporal features from epoched data using overlapping time windows."""
    windows = [(t, t + 0.1) for t in np.arange(0.0, epochs.tmax - 0.1, 0.05)]
    epoched_data = epochs.get_data()
    times = epochs.times
    
    n_epochs, n_channels, _ = epoched_data.shape
    features = np.zeros((n_epochs, n_channels * len(windows)))
    
    for i in range(n_epochs):
        feature_idx = 0
        for j in range(n_channels):
            for t_min, t_max in windows:
                time_slice = (times >= t_min) & (times < t_max)
                if np.any(time_slice):
                    features[i, feature_idx] = np.mean(epoched_data[i, j, time_slice])
                feature_idx += 1
                
    return features, epochs.events[:, -1]

def run_participant_specific_validation(features, labels, participant_id):
    """
    Performs participant-specific cross-validation with adaptive parameters.
    """
    if len(np.unique(labels)) < 2:
        return None
    
    # Adaptive CV parameters based on data size
    n_splits = min(5, len(labels) // 4)  # Ensure sufficient samples per fold
    min_class_count = np.min(np.unique(labels, return_counts=True)[1])
    n_splits = min(n_splits, min_class_count)
    
    if n_splits < 2:
        return None
    
    # Use different classifiers based on sample size
    if len(labels) < 50:
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.8)  # More regularization
    else:
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    ros = RandomOverSampler(random_state=42)
    
    fold_results = []
    
    for train_idx, test_idx in cv.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Skip fold if test set doesn't have both classes
        if len(np.unique(y_test)) < 2:
            continue
        
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        
        try:
            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc = roc_auc_score(y_test, y_proba)
            
            fold_results.append({
                'accuracy': accuracy, 'tpr': tpr, 'tnr': tnr, 'auc': auc,
                'participant': participant_id, 'n_train': len(y_train_res), 'n_test': len(y_test)
            })
            
        except Exception as e:
            print(f"    Error in fold for participant {participant_id}: {e}")
            continue
    
    if not fold_results:
        return None
    
    # Return aggregated results for this participant
    return {
        'participant': participant_id,
        'n_folds': len(fold_results),
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'tpr': np.mean([r['tpr'] for r in fold_results]),
        'tnr': np.mean([r['tnr'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'auc_std': np.std([r['auc'] for r in fold_results]),
        'fold_results': fold_results
    }

def aggregate_participant_results(participant_results):
    """
    Aggregate results across participants using different strategies.
    """
    if not participant_results:
        return None
    
    # Filter out None results
    valid_results = [r for r in participant_results if r is not None]
    if not valid_results:
        return None
    
    # Simple averaging across participants
    metrics = ['accuracy', 'tpr', 'tnr', 'auc']
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in valid_results]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
        aggregated[f'{metric}_median'] = np.median(values)
    
    # Weighted averaging by number of folds (data quality proxy)
    weights = [r['n_folds'] for r in valid_results]
    total_weight = sum(weights)
    
    for metric in metrics:
        weighted_sum = sum(r[metric] * w for r, w in zip(valid_results, weights))
        aggregated[f'{metric}_weighted'] = weighted_sum / total_weight
    
    aggregated['n_participants'] = len(valid_results)
    aggregated['participant_results'] = valid_results
    
    return aggregated

def save_results(all_results, quality_metrics, bad_participants, output_dir='./results'):
    """Save all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)): 
                return float(obj)
            if isinstance(obj, np.ndarray): 
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    # Save detailed results
    with open(os.path.join(output_dir, f'participant_specific_results_{timestamp}.json'), 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
        
    with open(os.path.join(output_dir, f'quality_metrics_{timestamp}.json'), 'w') as f:
        json.dump(quality_metrics, f, indent=2, cls=NpEncoder)


def create_boxplots(aggregated_results, output_dir='./results'):
    """Create boxplots for accuracy, TPR, and TNR across participants."""
    import matplotlib.pyplot as plt
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for plotting
    plot_data = {}
    for task_type, results in aggregated_results.items():
        if not results or not results['participant_results']:
            continue
        
        participants = [f"P{r['participant']}" for r in results['participant_results']]
        accuracies = [r['accuracy'] * 100 for r in results['participant_results']]
        tprs = [r['tpr'] * 100 for r in results['participant_results']]
        tnrs = [r['tnr'] * 100 for r in results['participant_results']]
        
        plot_data[task_type] = {
            'participants': participants,
            'accuracy': accuracies,
            'tpr': tprs,
            'tnr': tnrs
        }
    
    # Create plots for each task type
    for task_type, data in plot_data.items():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{task_type.upper()} Performance Metrics', fontsize=16)
        
        # Accuracy plot
        ax1.bar(data['participants'], data['accuracy'], alpha=0.7, color='skyblue')
        ax1.axhline(y=np.mean(data['accuracy']), color='red', linestyle='--', label=f'Mean: {np.mean(data["accuracy"]):.1f}%')
        ax1.set_title('Accuracy')
        ax1.set_ylabel('Accuracy [%]')
        ax1.set_xlabel('Participants')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # TPR plot
        ax2.bar(data['participants'], data['tpr'], alpha=0.7, color='lightgreen')
        ax2.axhline(y=np.mean(data['tpr']), color='red', linestyle='--', label=f'Mean: {np.mean(data["tpr"]):.1f}%')
        ax2.set_title('True Positive Rate (TPR)')
        ax2.set_ylabel('TPR [%]')
        ax2.set_xlabel('Participants')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # TNR plot
        ax3.bar(data['participants'], data['tnr'], alpha=0.7, color='lightcoral')
        ax3.axhline(y=np.mean(data['tnr']), color='red', linestyle='--', label=f'Mean: {np.mean(data["tnr"]):.1f}%')
        ax3.set_title('True Negative Rate (TNR)')
        ax3.set_ylabel('TNR [%]')
        ax3.set_xlabel('Participants')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_type}_performance_metrics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Boxplots saved to {output_dir}")

def main():
    """Main function to run the participant-specific analysis pipeline."""
    print("=== Participant-Specific ErrP Classification Pipeline ===")
    
    # Use all participants (remove filter to include everyone)
    PARTICIPANTS_TO_RUN = None  # Set to None to include all participants
    
    print("\n1. Loading data...")
    all_data, participants, tasks = load_all_data(participant_filter=PARTICIPANTS_TO_RUN)
    print(f"Loaded data for participants {sorted(participants)} and tasks {sorted(tasks)}")
    
    print("\n2. Creating epochs and running advanced preprocessing...")
    epochs_dict = create_epochs_from_data(all_data)
    
    print("\n3. Assessing data quality...")
    quality_metrics, bad_participants = assess_data_quality(epochs_dict)
    
    # Organize data by participant and task
    participant_data = {}
    for key, data in epochs_dict.items():
        participant = data['participant']
        task = data['task']
        
        if participant not in participant_data:
            participant_data[participant] = {}
        participant_data[participant][task] = data
    
    all_results = {'task1': [], 'task2': [], 'task3_slight': [], 'task3_severe': []}
    
    print("\n4. Running participant-specific classification...")
    
    for participant, tasks_data in participant_data.items():
        if participant in bad_participants:
            print(f"\nSkipping participant {participant} (flagged as bad quality)")
            continue
        
        print(f"\n--- Participant {participant} ---")
        
        for task, data in tasks_data.items():
            print(f"  Processing Task {task}...")
            features, labels = extract_features(data['epochs'])
            
            if task in [1, 2]:
                results = run_participant_specific_validation(features, labels, participant)
                if results:
                    all_results[f'task{task}'].append(results)
                    print(f"    Results: Acc={results['accuracy']:.3f}, AUC={results['auc']:.3f} ({results['n_folds']} folds)")
            
            elif task == 3:
                # Task 3: Non-error vs Slight-error
                mask_slight = (labels == 0) | (labels == 1)
                if np.sum(mask_slight) > 10:
                    labels_slight = (labels[mask_slight] > 0).astype(int)
                    if len(np.unique(labels_slight)) == 2:
                        results = run_participant_specific_validation(features[mask_slight], labels_slight, participant)
                        if results:
                            all_results['task3_slight'].append(results)
                            print(f"    Slight error: Acc={results['accuracy']:.3f}, AUC={results['auc']:.3f}")
                
                # Task 3: Non-error vs Severe-error
                mask_severe = (labels == 0) | (labels == 2)
                if np.sum(mask_severe) > 10:
                    labels_severe = (labels[mask_severe] > 1).astype(int)
                    if len(np.unique(labels_severe)) == 2:
                        results = run_participant_specific_validation(features[mask_severe], labels_severe, participant)
                        if results:
                            all_results['task3_severe'].append(results)
                            print(f"    Severe error: Acc={results['accuracy']:.3f}, AUC={results['auc']:.3f}")

    print("\n5. Aggregating and saving results...")

    # Aggregate results across participants
    aggregated_results = {}
    for task_type, participant_results in all_results.items():
        aggregated_results[task_type] = aggregate_participant_results(participant_results)

    save_results(aggregated_results, quality_metrics, bad_participants)

    # Create and save plots
    create_boxplots(aggregated_results)
    
    print("\n=== PARTICIPANT-SPECIFIC RESULTS SUMMARY ===")

    for task_type, results in aggregated_results.items():
        if not results:
            print(f"\n{task_type.upper()}: No valid results")
            continue
        
        print(f"\n{task_type.upper()} (n={results['n_participants']} participants):")
        print(f"  Accuracy: {results['accuracy_mean']*100:.1f}% (± {results['accuracy_std']*100:.1f}%)")
        print(f"  TPR: {results['tpr_mean']*100:.1f}% (± {results['tpr_std']*100:.1f}%)")
        print(f"  TNR: {results['tnr_mean']*100:.1f}% (± {results['tnr_std']*100:.1f}%)")
        print(f"  Balanced Accuracy: {(results['tpr_mean'] + results['tnr_mean'])/2*100:.1f}%")
        print(f"  AUC: {results['auc_mean']:.3f} (± {results['auc_std']:.3f})")
        print(f"  Weighted AUC: {results['auc_weighted']:.3f}")
        
        # Show individual participant performance
        print("  Individual participants:")
        for p_result in results['participant_results']:
            balanced_acc = (p_result['tpr'] + p_result['tnr']) / 2
            print(f"    P{p_result['participant']}: Acc={p_result['accuracy']:.3f}, TPR={p_result['tpr']:.3f}, TNR={p_result['tnr']:.3f}, Bal_Acc={balanced_acc:.3f}, AUC={p_result['auc']:.3f}")

if __name__ == '__main__':
    main()