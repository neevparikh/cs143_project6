import numpy as np 

class PoseNormalizer:
    ''' Normalizes the pose as described in the Everybody Dance Now paper '''

    def __init__(self, source, target, epsilon=0.7):
        self.s_left = source["left"]
        self.s_right = source["right"]
        self.t_left = target["left"]
        self.t_right = target["right"]
        self.epsilon = epsilon

    def _compute_translation(self):
        b = 0
        return b

    def _compute_scale(self):
        s = 0
        return s

    def _compute_statistics(self, ankle_series, ankle_name):
        self.statistics = {}
        med = self._get_median_ankle_position(ankle_series)
        mx = self._get_max_ankle_position(ankle_series)
        self.statistics[ankle_name] = {
            "med": med,
            "max": mx
        }
        self.statistics[ankle_name]["min"] = self._get_min_ankle_position(ankle_series, med, mx)
    
    def _get_median_ankle_position(self, ankle_series):
        return np.median(ankle_series, overwrite_input=False)
    
    def _get_min_ankle_position(self, ankle_series, med, mx):
        dist_mx_med = np.abs(mx - med)
        cluster = [t for t in ankle_series if (t < med) and (np.abs(np.abs() - dist_mx_med) < self.epsilon)]
        mn = max(cluster)
        return mn

    def _get_max_ankle_position(self, ankle_series):
        return np.amax(ankle_series)

    def transform_pose(self, frame):
        return frame 
