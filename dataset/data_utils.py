import numpy as np
import torch


# before 1/(1+exp(-6*(x+1))), after 1/(1+exp(-12*(-x+0.5)))
def compute_time_vector(labels, fps, TT=2, TA=1):
    """
    Compute time vector reflecting time in seconds before or after anomaly range.

    Parameters:
        labels (list or np.ndarray): Binary vector of frame labels (1 for anomalous, 0 otherwise).
        fps (int): Frames per second of the video.
        TT (float): Time-to-anomalous range in seconds (priority).
        TA (float): Time-after-anomalous range in seconds.

    Returns:
        np.ndarray: Time vector for each frame.
    """
    num_frames = len(labels)
    labels = np.array(labels)
    default_value = max(TT, TA) * 2
    time_vector = torch.zeros(num_frames, dtype=float)

    # Get anomaly start and end indices
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) == 0:
        return time_vector  # No anomalies, return all zeros

    # Define maximum frame thresholds for TT and TA
    TT_frames = int(TT * fps)
    TA_frames = int(TA * fps)

    # Iterate through each frame
    for i in range(num_frames):
        if labels[i] == 1:
            time_vector[i] = 0  # Anomalous frame, set to 0
        else:
            # Find distances to the start and end of anomaly ranges
            distances_to_anomalies = anomaly_indices - i

            # Time-to-closest-anomaly-range (TT priority)
            closest_to_anomaly = distances_to_anomalies[distances_to_anomalies > 0]  # After the frame
            if len(closest_to_anomaly) > 0 and closest_to_anomaly[0] <= TT_frames:
                time_vector[i] = -closest_to_anomaly[0] / fps
                continue

            # Time-after-anomaly-range (TA range)
            closest_after_anomaly = distances_to_anomalies[distances_to_anomalies < 0]  # Before the frame
            if len(closest_after_anomaly) > 0 and abs(closest_after_anomaly[-1]) <= TA_frames:
                time_vector[i] = -closest_after_anomaly[-1] / fps
                continue

            # Outside both TT and TA
            time_vector[i] = -100.

    return time_vector


def smooth_labels(labels, time_vector, before_limit=2, after_limit=1):
    xb = before_limit / 2
    xa = after_limit / 2
    kb = 12 / before_limit # 6 for before_limit=2
    ka = 12 / after_limit # 12 for after_limit=1
    sigmoid_before = lambda x: (1 / (1 + torch.exp(-kb * (x + xb)))).float()
    sigmoid_after = lambda x: (1 / (1 + torch.exp(-ka * (-x + xa)))).float()

    before_mask = (time_vector >= -before_limit) & (time_vector < 0)
    after_mask = (time_vector > 0) & (time_vector <= after_limit)

    target_anomaly = (labels == 1).float()
    target_anomaly[before_mask] = sigmoid_before(time_vector[before_mask])
    target_anomaly[after_mask] = sigmoid_after(time_vector[after_mask])
    target_safe = 1 - target_anomaly
    smoothed_target = torch.stack((target_safe, target_anomaly), dim=-1)
    return smoothed_target
