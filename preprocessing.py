import numpy as np

def normalize_in_range (lb, ub, data):
    # map to {0,1}
    normalization = (data - np.min(data))/ (np.max(data) - np.min(data))
    # scale to {0, ub - lb}
    scale = normalization * (ub - lb)
    # shift to {lb ,ub}
    shift = scale + lb
    return shift

def normalized_value_to_original_range(lb, ub, normalized_data, min_org_dataset, max_org_dataset):
    # map back to original range
    original_data = ((normalized_data - lb) * (max_org_dataset - min_org_dataset) / (ub - lb)) + min_org_dataset
    return original_data 
