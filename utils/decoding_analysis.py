
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from types import SimpleNamespace as Bunch
import matplotlib.pyplot as plt
import numpy as np

def split_for_decoding(spike_counts, speed_arena, speed_wheel, mask_arena, mask_wheel):
    
    epsilon = 1e-8 
    # Calculate the half point of the spike counts
    halfpoint = len(spike_counts[1]) // 2

    # Create masks for training and testing sets
    mask_arena_half1 = mask_arena[:halfpoint]
    mask_wheel_half1 = mask_wheel[:halfpoint]
    

    # Get training speed data
    speed_arena_half1 = speed_arena[:halfpoint][mask_arena_half1]
    speed_wheel_half1 = speed_wheel[:halfpoint][mask_wheel_half1]

    # Calculate GLOBAL normalization parameters from training data only
    all_half1_speeds = np.concatenate([speed_arena_half1, speed_wheel_half1])
    half1_speed_mean = np.mean(all_half1_speeds)
    half1_speed_std = np.std(all_half1_speeds)
    
    # Normalize training speeds with global parameters
    speed_arena_half1 = (speed_arena_half1 - half1_speed_mean) / half1_speed_std
    speed_wheel_half1 = (speed_wheel_half1 - half1_speed_mean) / half1_speed_std

    # Get training spike counts
    spike_counts_arena_half1 = spike_counts[:, :halfpoint][:, mask_arena_half1]
    spike_counts_wheel_half1 = spike_counts[:, :halfpoint][:, mask_wheel_half1]

    all_half1_spikes = np.concatenate([spike_counts_arena_half1, spike_counts_wheel_half1], axis=1)
    half1_spike_mean = np.mean(all_half1_spikes, axis=1, keepdims=True)
    half1_spike_std = np.std(all_half1_spikes, axis=1, keepdims=True)

    spike_counts_arena_half1 = (spike_counts_arena_half1 - half1_spike_mean) / (half1_spike_std + epsilon)
    spike_counts_wheel_half1 = (spike_counts_wheel_half1 - half1_spike_mean) / (half1_spike_std + epsilon)

    # Create masks for testing sets
    mask_arena_half2 = mask_arena[halfpoint:]
    mask_wheel_half2 = mask_wheel[halfpoint:]


    # Get testing speed data
    speed_arena_half2 = speed_arena[halfpoint:][mask_arena_half2]
    speed_wheel_half2 = speed_wheel[halfpoint:][mask_wheel_half2]

    speed_arena_half2 = (speed_arena_half2 - half1_speed_mean) / half1_speed_std 
    speed_wheel_half2 = (speed_wheel_half2 - half1_speed_mean) / half1_speed_std 
    
    # Get testing spike counts
    spike_counts_arena_half2 = spike_counts[:, halfpoint:][:, mask_arena_half2]
    spike_counts_wheel_half2 = spike_counts[:, halfpoint:][:, mask_wheel_half2]

    spike_counts_arena_half2 = (spike_counts_arena_half2 - half1_spike_mean) / (half1_spike_std+ epsilon)
    spike_counts_wheel_half2 = (spike_counts_wheel_half2 - half1_spike_mean) / (half1_spike_std+ epsilon)
    

    train_data = Bunch(

        speed_arena= speed_arena_half1,
        speed_wheel= speed_wheel_half1,
        spike_counts_arena = spike_counts_arena_half1.T, # decoder expects time_points x neurons
        spike_counts_wheel = spike_counts_wheel_half1.T,
    )

    test_data = Bunch(

        speed_arena= speed_arena_half2,
        speed_wheel= speed_wheel_half2,
        spike_counts_arena = spike_counts_arena_half2.T,
        spike_counts_wheel = spike_counts_wheel_half2.T,
    )

    return train_data, test_data



def train_model(spike_counts, speed, alpha=None):
    
    
    model = RidgeCV(alphas=alpha, store_cv_values=True)
    model.fit(spike_counts, speed)

    return model

   



def compute_leaveout_analysis(train_data, test_data, weights_arena, weights_wheel, alpha=None):

    contexts = ['arena', 'wheel']
    n_iterations = len(weights_arena) - 1
    control_weights = np.arange(len(weights_wheel))
    np.random.shuffle(control_weights)

    sort_indices = {
        'arena': np.argsort(np.abs(weights_arena))[::-1],
        'wheel': np.argsort(np.abs(weights_wheel))[::-1],
        'random': control_weights
    }
    
    r2_curves = {}
    cosine_similarities = {}
    
    # Compute all R² curves and save weights for cosine similarity
    for sort_by in sort_indices.keys():
        sort_idx = sort_indices[sort_by]
        
        # Temporary storage for weights
        arena_weights = []
        wheel_weights = []
        
        for train_on in contexts:
            # Setup data
            train_spikes = getattr(train_data, f'spike_counts_{train_on}')[:, sort_idx]
            test_spikes = getattr(test_data, f'spike_counts_{train_on}')[:, sort_idx]
            train_speed = getattr(train_data, f'speed_{train_on}')
            test_speed = getattr(test_data, f'speed_{train_on}')
            
            # Compute R² curve
            scores = np.zeros(n_iterations)
            ridge = Ridge(alpha=alpha)
            
            for i in range(n_iterations):
                ridge.fit(train_spikes[:, i:], train_speed)
                scores[i] = ridge.score(test_spikes[:, i:], test_speed)
                
                # Save weights for cosine similarity
                if train_on == 'arena':
                    arena_weights.append(ridge.coef_)
                else:
                    wheel_weights.append(ridge.coef_)
            
            # Store R² curve with simple key
            r2_curves[f'{train_on}_{sort_by}'] = scores
        
        # Compute cosine similarity for this sorting
        cosine_sim = np.zeros(n_iterations)
        for i in range(n_iterations):
            cosine_sim[i] = np.dot(arena_weights[i], wheel_weights[i]) / (
                np.linalg.norm(arena_weights[i]) * np.linalg.norm(wheel_weights[i])
            )
        cosine_similarities[sort_by] = cosine_sim
    
    return Bunch(
        r2_curves=r2_curves, 
        cosine_similarities=cosine_similarities,  
    )
            
                      

