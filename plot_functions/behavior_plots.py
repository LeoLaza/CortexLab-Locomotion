
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon


def plot_context_speed_distributions(speed_arena, speed_wheel, mask_arena, mask_wheel):

    fig, ax =plt.subplots(1,1, figsize=(6, 6))
    ax.hist(speed_arena[mask_arena], color='#195A2C', bins=50, alpha=0.9, density=True)
    ax.hist(speed_wheel[mask_wheel], color='#7D0C81', bins=50, alpha=0.7, density=True )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('speed (cm/s)', fontsize=20)
    ax.set_ylabel('probability', fontsize=20)
    ax.set_xticks(np.arange(0,50.1,25))
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()



def plot_context_preference(all_session_results):
    """
    Plot comparison of time spent in arena vs wheel context for all sessions.

    Parameters:
    all_session_results : list
        comprises results objects for each session containing behavior summary data
    """
    
    # gather occupancy data for each context across sessions
    session_data = []
    for result in all_session_results:
        if result.spike_counts is None or result.behavior.summary is None: # only include sessions with neural data
            continue
            
        # determine color based on subject_id
        if result.metadata.subject_id == 'EB036':
            color = "#A5CB5D"  # hippocampus
        elif result.metadata.subject_id == 'EB037':
            color = "#E37A2A"  # striatum
        elif result.metadata.subject_id == 'AV043':
            color = "#660D0D"  # MOs(1)
        elif result.metadata.subject_id == 'GB011':
            color = "#9F3A3A"  # MOs(2)
        elif result.metadata.subject_id == 'GB012':
            color = "#D26E6E"  # MOs(3)
        else:
            color = "#1F1F1F"  # default/fallback
            
        session_data.append({
            'arena': result.behavior.summary.occupancy.arena,
            'wheel': result.behavior.summary.occupancy.wheel,
            'color': color
        })
    
    # convert to arrays to calculate mean occupancy across sessions
    occupancy_arena = np.array([s['arena'] for s in session_data])
    occupancy_wheel = np.array([s['wheel'] for s in session_data])
    mean_arena = np.mean(occupancy_arena)
    mean_wheel = np.mean(occupancy_wheel)
    
    # create figure
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # plot occupancy within each context for each session
    for i, session in enumerate(session_data):
        jitter = np.random.uniform(-0.05, 0.05, 2)
        
        # plot dot for occupancy in each context
        ax.scatter([0 + jitter[0], 1 + jitter[1]], 
                  [session['arena'], session['wheel']], 
                  color=session['color'], alpha=0.7, s=70, zorder=3)
        
        # connect dots of the same session with a line
        ax.plot([0 + jitter[0], 1 + jitter[1]], 
                [session['arena'], session['wheel']], 
                '--', color=session['color'], alpha=0.2, linewidth=1)
    
    # mean lines with arena/wheel colors
    ax.plot([-0.2, 0.2], [mean_arena, mean_arena], 
            color='#195A2C', linewidth=3, solid_capstyle='round', zorder=5, ls='dotted')
    ax.plot([0.8, 1.2], [mean_wheel, mean_wheel], 
            color='#7D0C81', linewidth=3, solid_capstyle='round', zorder=5, ls='dotted')
    
    # axis formatting
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['arena', 'wheel'], fontsize=18)
    xcolors = ["#195A2C", "#7D0C81"]
    for xtick, xcolor in zip(ax.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax.set_ylabel('time spent (%)', fontsize=18)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=16)
    ax.axhline(50, color='gray', linestyle='dotted', linewidth=1.5, alpha=0.5)
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()


def plot_reliability_occupation(all_session_results):
    """
    Side by side plot of correlation between reliability and time spent in each context across sessions.

    Parameters:
    all_session_results : list
        comprises results objects for each session containing behavior summary and correlation data
    """

    # gather reliability and occupancy data for each context across sessions
    reliability_arena= [r.correlations.reliability_arena for r in all_session_results 
              if hasattr(r.correlations, 'reliability_arena')]

    occupancy_arena=  [r.behavior.summary.occupancy.arena for r in all_session_results 
              if hasattr(r.correlations, 'reliability_arena')]
    
    reliability_wheel= [r.correlations.reliability_wheel for r in all_session_results 
              if hasattr(r.correlations, 'reliability_wheel')]

    occupancy_wheel=  [r.behavior.summary.occupancy.wheel for r in all_session_results 
              if hasattr(r.correlations, 'reliability_wheel')]
    

    # convert occupancy from percentages to proportions
    occupancy_arena= np.divide(occupancy_arena, 100)
    occupancy_wheel= np.divide(occupancy_wheel, 100)

    # make scatter plots with correlation coefficients
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    
    ax1.scatter(reliability_arena, occupancy_arena, alpha=0.7, color= '#195A2C', zorder=2, s=80)
    r = np.corrcoef(reliability_arena, occupancy_arena)[0,1]
    ax1.text(0.05, 0.95, f'r = {r:.3f}', transform=ax1.transAxes, 
        fontsize=18, verticalalignment='top')
    ax1.set_xlabel("reliability arena", fontsize=14)
    ax1.set_ylabel("tmime proportion arena", fontsize=14)
    ax1.axhline(0.5, color='gray', linestyle='dotted', linewidth=1.5, alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(np.arange(0, 1.01, 0.5))  
    ax1.set_yticks(np.arange(0, 1.01, 0.5))
    ax1.tick_params(axis='both', labelsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.scatter(reliability_wheel, occupancy_wheel, alpha=0.7, color= '#7D0C81', zorder=2, s=80)
    r = np.corrcoef(reliability_wheel, occupancy_wheel)[0,1]
    ax2.text(0.05, 0.95, f'r = {r:.3f}', transform=ax2.transAxes, 
        fontsize=18, verticalalignment='top')
    ax2.set_xlabel("reliability wheel", fontsize=14)
    ax2.set_ylabel("time proportion wheel", fontsize=14)
    ax2.axhline(0.5, color='gray', linestyle='dotted', linewidth=1.5, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(np.arange(0, 1.01, 0.5))  
    ax2.set_yticks(np.arange(0, 1.01, 0.5))
    ax2.tick_params(axis='both', labelsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

def plot_stability_speed_distrubution_similarity(all_session_results):
    """
    Plot correlation between session stability and similarity of speed distributions across contexts.

    Parameters:
    all_session_results : list
        comprises results objects for each session containing behavior summary and correlation data
    """

    # gather stability and compute speed distribution similarity for each session
    distribution_similarities = []
    stabilities = []

    for session in all_session_results:
        
        
        if session.spike_counts is None or session.correlations is None:
            continue
        
        if np.isnan(session.correlations.stability):
            continue
        # extract speed data 
        speed_arena = session.behavior.speed_arena[session.behavior.mask_arena]
        speed_wheel = session.behavior.speed_wheel[session.behavior.mask_wheel]
        
        if len(speed_arena) == 0 or len(speed_wheel) == 0:
            continue
        
        # compute histograms
        all_speeds = np.concatenate([speed_arena, speed_wheel])
        bins = np.linspace(0, np.percentile(all_speeds, 99), 50)
        
        hist_arena, _ = np.histogram(speed_arena, bins=bins, density=True)
        hist_wheel, _ = np.histogram(speed_wheel, bins=bins, density=True)
        
        # add small epsilon to avoid zeros
        hist_arena = hist_arena + 1e-10
        hist_wheel = hist_wheel + 1e-10
        
        # normalize
        hist_arena = hist_arena / np.sum(hist_arena)
        hist_wheel = hist_wheel / np.sum(hist_wheel)
        
        # calculate jensen-shannon divergence (measure of distribution similarity)
        js_divergence = jensenshannon(hist_arena, hist_wheel)
        distribution_similarity = 1 - js_divergence
        
        # store results
        distribution_similarities.append(distribution_similarity)
        stabilities.append(session.correlations.stability)

    # create scatter plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(distribution_similarities, stabilities, 
            alpha=0.7, color="#141414", zorder=2, s=80)

    # axis formatting
    ax.set_xlabel('speed distribution similarity', fontsize=21)
    ax.set_ylabel('stability', fontsize=21)
    ax.set_xlim(0.3, 0.6)  
    ax.set_ylim(-0.2, 0.4)  
    ax.set_xticks(np.arange(0.3, 0.61, 0.3))
    ax.set_yticks(np.arange(-0.2, 0.4, 0.2))
    ax.tick_params(axis='both', labelsize=20, pad=10)

    # remove spines
    ax.spines[['right', 'top']].set_visible(False)

    #  annotate with correlation
    r = np.corrcoef(distribution_similarities, stabilities)[0,1]
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            fontsize=18, verticalalignment='top')
    

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()
   

def plot_mean_speed_comparison(all_session_results):
    """
    Plot comparison of mean speed in arena vs wheel context for all sessions.
    Parameters:
    all_session_results : list
        comprises results objects for each session containing behavior summary data
    """
    
    # gather mean speed data for each context across sessions
    session_data = []
    for result in all_session_results:
        if result.spike_counts is None or result.behavior.summary is None:
            continue
            
        # determine color based on subject_id
        if result.metadata.subject_id == 'EB036':
            color = "#A5CB5D"  # hippocampus
        elif result.metadata.subject_id == 'EB037':
            color = "#E37A2A"  # striatum
        elif result.metadata.subject_id == 'AV043':
            color = "#660D0D"  # MOs(1)
        elif result.metadata.subject_id == 'GB011':
            color = "#9F3A3A"  # MOs(2)
        elif result.metadata.subject_id == 'GB012':
            color = "#D26E6E"  # MOs(3)
        else:
            color = "#1F1F1F"  # default/fallback
            
        session_data.append({
            'arena': result.behavior.summary.mean_speed.arena,
            'wheel': result.behavior.summary.mean_speed.wheel,
            'color': color
        })
    
    # convert to arrays to calculate mean speeds across sessions
    speed_arena = np.array([s['arena'] for s in session_data])
    speed_wheel = np.array([s['wheel'] for s in session_data])
    mean_arena = np.mean(speed_arena)
    mean_wheel = np.mean(speed_wheel)
    
    # plot mean speed within each context for each session
    fig, ax = plt.subplots(figsize=(4, 5))
    
    for i, session in enumerate(session_data):
        jitter = np.random.uniform(-0.05, 0.05, 2)

        # plot dot for mean speed in each context
        ax.scatter([0 + jitter[0], 1 + jitter[1]], 
                  [session['arena'], session['wheel']], 
                  color=session['color'], alpha=0.7, s=70, zorder=3)
        
        # connect dots of the same session with a line
        ax.plot([0 + jitter[0], 1 + jitter[1]], 
                [session['arena'], session['wheel']], 
                '--', color=session['color'], alpha=0.2, linewidth=1)
        
    # mean lines with arena/wheel colors
    ax.plot([-0.2, 0.2], [mean_arena, mean_arena], 
            color='#195A2C', linewidth=3, solid_capstyle='round', zorder=5, ls='dotted')
    ax.plot([0.8, 1.2], [mean_wheel, mean_wheel], 
            color='#7D0C81', linewidth=3, solid_capstyle='round', zorder=5, ls='dotted')
    
    # axis formatting
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['arena', 'wheel'], fontsize=18)
    xcolors = ["#195A2C", "#7D0C81"]
    for xtick, xcolor in zip(ax.get_xticklabels(), xcolors):
        xtick.set_color(xcolor)
    ax.set_ylabel('mean speed (cm/s)', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_yticks(np.arange(0, 40.1, 10))
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tight_layout()


def plot_locomotion_detection(behavior, running_arena, running_wheel, w_start=0, w_end=200, 
                             onset_threshold=2):
        """
        Plot speed traces with positon masks and detected locomotion bouts.
        
        Parameters:
        behavior :  Bunch object
            single session behavioral variables containing speed and mask data
        running_arena : array
            boolean array indicating detected locomotion bouts in arena context
        running_wheel : array
            boolean array indicating detected locomotion bouts in wheel context
        w_start : int
            start index for plotting window
        w_end : int
            end index for plotting window
        """
  
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        time = np.arange(w_start, w_end) 
        
        # plot arena speed 
        ax1.plot(time, behavior.speed_arena[w_start:w_end], 'k-', linewidth=1.2, alpha=0.8, color= 'black')
        
        # plot position based arena mask
        ax1.fill_between(time, 0, 30, where=behavior.mask_arena[w_start:w_end],
                        color='#195A2C', alpha=0.2, label='in arena')
        
        # plot detected locomotion bouts arena
        locomotion_height = np.ones_like(time) * 0.8
        ax1.fill_between(time, 0, locomotion_height, 
                        where=running_arena[w_start:w_end],
                        color='#0B3D0B', alpha=0.8, label='locomotion bout')
        
        # plot onset threshold
        ax1.axhline(onset_threshold, color='grey', linestyle='dotted', alpha=0.5, linewidth=1.5)
        
        # format arena panel
        ax1.set_ylabel('arena speed (cm/s)', fontsize=11)
        ax1.set_ylim(0, max(15, behavior.speed_arena[w_start:w_end].max() * 1.1))
        ax1.set_yticks(np.arange(0, 20.1, 20))
        ax1.tick_params(axis='y', labelsize=11)
    
        
        # plot wheel speed 
        ax2.plot(time, behavior.speed_wheel[w_start:w_end], 'k-', linewidth=1.2, alpha=0.8, color= 'black')
        
        # plot position based wheel mask
        ax2.fill_between(time, 0, 40, where=behavior.mask_wheel[w_start:w_end],
                        color='#7D0C81', alpha=0.2, label='on wheel')
        
        # plot detected locomotion bouts wheel
        ax2.fill_between(time, 0, locomotion_height, 
                        where=running_wheel[w_start:w_end],
                        color='#3D053F', alpha=0.8, label='locomotion bout')
        
        # plot onset threshold 
        ax2.axhline(onset_threshold, color='grey', linestyle='dotted', alpha=0.5, linewidth=1.5)

        # format wheel panel
        ax2.set_ylabel('wheel speed (cm/s)', fontsize=11)
        ax2.set_ylim(0, max(15, behavior.speed_wheel[w_start:w_end].max()))
        ax2.set_xticks([])
        ax2.set_yticks(np.arange(0, 30.1, 30))
        ax2.tick_params(axis='y', labelsize=11)

        # set legend for each panel (might have to adjust positioning depending on data)
        ax1.legend(loc='upper right', frameon=False)
        ax2.legend(loc='upper right', frameon=False)

        # remove spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.tight_layout()