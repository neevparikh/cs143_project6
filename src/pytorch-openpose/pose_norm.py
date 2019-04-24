import numpy as np 

class PoseNormalizer:
    ''' Normalizes the pose as described in the Everybody Dance Now paper '''

    def __init__(self, source, target, epsilon=0.7):
        """
            source :: dict<ndarray> :: dict of source left ankle array and source right ankle array
            target :: dict<ndarray> :: dict of target left ankle array and target right ankle array
            epsilon :: float [0, 1] :: value for the clustering in calculating the min, paper suggests 0.7  
        """

        self.s_left = source["left"]
        self.s_right = source["right"]
        self.t_left = target["left"]
        self.t_right = target["right"]
        self.epsilon = epsilon

        self._compute_statistics(np.append(self.s_left, self.s_right), "source")
        self._compute_statistics(np.append(self.t_left, self.t_right), "target")

    def _compute_translation(self, source):
        """ b = t_min + (avg_frame_pos_source - s_min) / (s_max - s_min) * (t_max - t_min) - f_source """

        # NOTE: f_source assumed 0 as we don't know what it is yet
        avg_source = (source["left"] + source["right"]) / 2
        t_min = self.statistics["target"]["min"]
        t_max = self.statistics["target"]["max"]
        s_min = self.statistics["source"]["min"]
        s_max = self.statistics["source"]["max"]

        return t_min + (avg_source - s_min) / (s_max - s_min) * (t_max - t_min) 

    def _compute_scale(self, source):
        """ s = t_far / s_far + (a_source - s_min) / (s_max - s_min) * (t_close / s_close - t_far / s_far) """
        avg_source = (source["left"] + source["right"]) / 2
        t_far = self.statistics["target"]["far"]
        t_close = self.statistics["target"]["close"]
        s_far = self.statistics["source"]["far"]
        s_close = self.statistics["source"]["close"]
        t_min = self.statistics["target"]["min"]
        t_max = self.statistics["target"]["max"]
        s_min = self.statistics["source"]["min"]
        s_max = self.statistics["source"]["max"]

        return (t_far / s_far) + (avg_source - s_min) / (s_max - s_min) * ((t_close / s_close) - (t_far / s_far))

    def _compute_statistics(self, ankle_series, ankle_name):
        self.statistics = {}
        med = self._get_median_ankle_position(ankle_series)
        mx = self._get_max_ankle_position(ankle_series)
        self.statistics[ankle_name] = {
            "med": med,
            "max": mx
        }
        mn = self._get_min_ankle_position(ankle_series, med, mx)
        self.statistics[ankle_name]["min"] = mn
        self.statistics[ankle_name]["close"], self.statistics[ankle_name]["far"] = self._get_close_far_position(ankle_series, mx, mn)
    
    def _get_median_ankle_position(self, ankle_series):
        return np.median(ankle_series, overwrite_input=False)
    
    def _get_min_ankle_position(self, ankle_series, med, mx):
        dist_mx_med = np.abs(mx - med)
        print(ankle_series, med, mx)
        print(dist_mx_med)
        cluster = np.array([p for p in ankle_series if (p < med) and (np.abs(np.abs(p - med) - dist_mx_med) < self.epsilon)])
        print(cluster)
        return np.max(cluster)

    def _get_close_far_position(self, ankle_series, mx, mn):
        cluster_far = np.array([p for p in ankle_series if (np.abs(p - mn) < self.epsilon)])
        cluster_close = np.array([p for p in ankle_series if (np.abs(p - mx) < self.epsilon)])
        return np.max(cluster_close), np.max(cluster_far)

    def _get_max_ankle_position(self, ankle_series):
        return np.amax(ankle_series)

    def transform_pose(self, source, target):
        """
            source :: ndarray :: numpy array of all the pose estimates as returned by pose estimation of source video 
            target :: ndarray :: numpy array of all the pose estimates as returned by pose estimation of target video
            
            Returns :: normalized target in the same format 
        """

        source_ankles = {"left": source[13, 1], "right": source[10, 1]}

        b = self._compute_translation(source_ankles)
        s = self._compute_scale(source_ankles)
        target[:, 1] *= s
        target[:, 1] += b
        return target
