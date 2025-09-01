
import matplotlib.pyplot as plt
import numpy as np
import random


def plot_category_scatter(result, categories):

    labels = ['Context invariant', 'Arena only', 'Wheel only', 'Context switching']
    colors = ["#000000", '#195A2C', '#7D0C81', '#083356']
    category_fields = [categories.context_invariant, categories.arena_only, categories.wheel_only, categories.context_switching]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot the data for each category with its specific color
    for i in range(len(labels)):
        # Get the boolean mask for the current category
        mask = category_fields[i]
        
        # Use the mask to select the relevant data points
        ax.scatter(result.correlations.arena[mask], result.correlations.wheel[mask], 
                c=colors[i], label=labels[i], alpha=0.7, s=50)


    ax.set_xlabel('correlation arena', fontsize=14)
    ax.set_ylabel('correlation wheel', fontsize=14)

    # Optional: Add lines to indicate zero correlation
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xticks(np.arange(-0.4, 0.41, 0.4))

    ax.set_yticks(np.arange(-0.4, 0.41, 0.4))
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_category_pie_chart(result, categories):

    counts = [
        np.sum(categories.context_invariant),
        np.sum(categories.arena_only),
        np.sum(categories.wheel_only),
        np.sum(categories.context_switching),
        np.sum(categories.non_encoding)
    ]

    labels = ['Context\ninvariant', 'Arena\nonly', 'Wheel\nonly', 'Context\nswitching', 'Non-\nencoding']
    colors = ["#000000", '#195A2C', '#7D0C81', '#083356', '#D3D3D3']

    # Create pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                                    startangle=90, pctdistance=0.85,
                                    textprops={'fontsize': 500})  # Increased from 10 to 14

    # Make percentage text bold and bigger
    for autotext in autotexts:
        autotext.set_weight('bold')
        autotext.set_color('white')
        autotext.set_fontsize(16)  # Set percentage font size explicitly

    # Make label text bigger
    for text in texts:
        text.set_fontsize(18)  # Set label font size explicitly

    ax.set_title(f'{result.metadata.subject_id} - {result.metadata.date}', fontsize=12, pad=20)
    


def plot_correlation_distributions(all_session_results, context='arena'):
    
    # Define colors
    colors = {
        'hippocampus': "#A5CB5D",  
        'striatum': "#E37A2A",
        'MOs(1)': "#660D0D",  # AV043
        'MOs(2)': "#9F3A3A",  # GB011  
        'MOs(3)': "#D26E6E"   # GB012
    }
    
    # Organize data by subject
    data_by_subject = {}
    for result in all_session_results:
        if result.spike_counts is None:
            continue
        
        subject_id = result.metadata.subject_id
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = []
        
        # Get correlations based on context
        if context == 'wheel':
            data_by_subject[subject_id].append(result.correlations.wheel)
        else:
            data_by_subject[subject_id].append(result.correlations.arena)
    
    # For MOs mice (AV043, GB011, GB012), randomly select 3 sessions
    mos_subjects = ['AV043', 'GB011', 'GB012']
    for subject in mos_subjects:
        if subject in data_by_subject:
            n_sessions = len(data_by_subject[subject])
            if n_sessions > 3:
                indices = random.sample(range(n_sessions), 3)
                data_by_subject[subject] = [
                    data_by_subject[subject][i] for i in indices
                ]
    
    # Create single figure with wider aspect ratio
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define order and labels for plotting
    plot_order = [
        ('AV043', 'MOs1', 'MOs(1)'),
        ('GB011', 'MOs2', 'MOs(2)'),
        ('GB012', 'MOs3', 'MOs(3)'),
        ('EB037', 'striatum', 'striatum'),
        ('EB036', 'hippocampus', 'hippocampus')
    ]
    
    positions = []
    labels = []
    spacing = 1.5  # Increased spacing between histograms
    
    for idx, (subject_id, label, color_key) in enumerate(plot_order):
        position = idx * spacing  # Apply spacing multiplier
        
        if subject_id in data_by_subject:
            positions.append(position)
            labels.append(label)
            
            # Pool all correlations from selected sessions
            pooled_correlations = []
            
            for corr in data_by_subject[subject_id]:
                pooled_correlations.extend(corr.flatten())
            
            pooled_correlations = np.array(pooled_correlations)
            
            # Show all correlations
            if len(pooled_correlations) > 0:
                hist, bins = np.histogram(pooled_correlations, bins=70, range=(-0.5, 0.5))
                # Normalize by total neuron count to show proportion
                hist = hist / len(pooled_correlations)
                # Scale for display width
                hist = hist / 0.1 * 0.5  # Slightly wider histograms
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                for h, b in zip(hist, bin_centers):
                    ax.barh(b, h, left=position-h/2, height=bins[1]-bins[0], 
                            color=colors[color_key], alpha=0.8, edgecolor='none')
    
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.75, (len(plot_order) - 1) * spacing + 0.75)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Pearson correlation', fontsize=18)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title(f"{context}", fontsize=18)
    
    # Add vertical lines to separate brain regions with adjusted positions
    ax.axvline(2.5 * spacing, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(3.5 * spacing, color='gray', linestyle=':', alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()