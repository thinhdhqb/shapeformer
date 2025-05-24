import numpy as np

def find_best_matching_subsequences(time_series, si):
    """Find best matching subsequences for a shapelet"""
    num_samples = len(time_series)
    distances = np.zeros(num_samples)
    best_positions = np.zeros(num_samples, dtype=int)
    
    shapelet = time_series[int(si[0]), int(si[5]), int(si[1]):int(si[2])]
    window_size = int(si[2]) - int(si[1])

    for i in range(num_samples):
        min_dist = float('inf')
        best_pos = 0
        
        # Sliding window over the time series
        for j in range(len(time_series[i, 0]) - window_size + 1):
            # Extract subsequence
            subseq = time_series[i, int(si[5]), j:j+window_size]
            
            # Calculate complexity of subsequence and shapelet
            ci_subseq = np.sum(np.square(np.diff(subseq)))
            ci_shapelet = np.sum(np.square(np.diff(shapelet)))
            
            # Calculate Euclidean distance
            ed = np.sqrt(np.sum(np.square(subseq - shapelet)))
            
            # Calculate complexity-invariant distance
            ci_factor = max(ci_subseq, ci_shapelet) / min(ci_subseq, ci_shapelet)
            dist = ed * ci_factor
            
            if dist < min_dist:
                min_dist = dist
                best_pos = j
        
        distances[i] = min_dist
        best_positions[i] = best_pos
        # print(f"Sample {i}: Best position {best_pos}, Distance {min_dist}")
    
    return distances, best_positions

def augment(time_series, shapelets_info, noise_std=0.5, num_copies=2):
    """ Augment time series using adaptive noise based on shapelet distances """
    num_samples = len(time_series)
    ts_length = time_series.shape[2]
    augmented_series = np.tile(time_series, (num_copies, 1, 1))
    
    # Noise mask (1 for non-matching subsequences)
    noise_mask = np.ones((num_copies * num_samples, time_series.shape[1], ts_length))
    
    # Set noise mask for best matching subsequences
    for si in shapelets_info:
        # Extract shapelet
        window_size = int(si[2]) - int(si[1])
        
        # Find best matching subsequences and their distances
        distances, positions = find_best_matching_subsequences(time_series, si)
        
        # Normalize distances to [0, 1] range
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        # Update noise mask for each copy
        for copy in range(num_copies):
            for i in range(num_samples):
                ts_idx = i + (copy * num_samples)
                start_pos = positions[i]
                end_pos = start_pos + window_size
                
                # Set mask values for matching subsequence (smaller distance = smaller noise)
                noise_mask[ts_idx, int(si[5]), start_pos:end_pos] = distances[i]
    
    # Apply noise
    for copy in range(num_copies):
        for i in range(num_samples):
            ts_idx = i + (copy * num_samples)
            noise = np.random.normal(0, noise_std, size=time_series.shape[1:])
            print(f"Noise mask for sample {i}, copy {copy}: {noise_mask[ts_idx]}")
            augmented_series[ts_idx] += noise * noise_mask[ts_idx]
    
    return augmented_series