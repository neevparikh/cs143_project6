import numpy as np 

class PoseNormalizer:
    ''' Normalizes the pose as described in the Everybody Dance Now paper '''

    def __init__(self, source, target, epsilon=0.7, inclusion_threshold=10):
        """
            source :: dict<ndarray> :: dict of source left ankle array and source right ankle array
            target :: dict<ndarray> :: dict of target left ankle array and target right ankle array
            epsilon :: float [0, 1] :: value for the clustering in calculating the min, paper suggests 0.7  
        """

        self.inclusion_threshold = inclusion_threshold
        self.s_left, self.s_right = self._include_ground_only(source["left"], source["right"]) 
        self.t_left, self.t_right = self._include_ground_only(target["left"], target["right"]) 
        self.epsilon = epsilon
        self.statistics = {}
        self._compute_statistics(np.append(self.s_left, self.s_right), "source")
        self._compute_statistics(np.append(self.t_left, self.t_right), "target")

    
    def _include_ground_only(self, left_ankle_array, right_ankle_array):
        """ remove the frames where the leg is raised """

        num_frames = len(left_ankle_array)
        left_grounded = [] 
        right_grounded = []

        for i in range(num_frames):
            if np.abs(left_ankle_array[i] - right_ankle_array[i]) < self.inclusion_threshold:
                left_grounded.append(left_ankle_array[i])
                right_grounded.append(right_ankle_array[i])
            else:
                pass

        return np.array(left_grounded), np.array(right_grounded)

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

    def _compute_statistics(self, ankle_array, ankle_name):
        med = self._get_median_ankle_position(ankle_array)
        mx = self._get_max_ankle_position(ankle_array)
        self.statistics[ankle_name] = {
            "med": med,
            "max": mx
        }
        mn = self._get_min_ankle_position(ankle_array, med, mx)
        self.statistics[ankle_name]["min"] = mn
        self.statistics[ankle_name]["close"], self.statistics[ankle_name]["far"] = self._get_close_far_position(ankle_array, mx, mn)
    
    def _get_median_ankle_position(self, ankle_array):
        return np.median(ankle_array, overwrite_input=False)
    
    def _get_min_ankle_position(self, ankle_array, med, mx):
        dist_mx_med = np.abs(mx - med)/mx
        cluster = np.array([p for p in ankle_array if (p < med) and (np.abs(np.abs(p - med) - dist_mx_med) < self.epsilon)])
        return np.max(cluster)

    def _get_close_far_position(self, ankle_array, mx, mn):
        cluster_far = np.array([p for p in ankle_array if (np.abs(p - mn) < self.epsilon)])
        cluster_close = np.array([p for p in ankle_array if (np.abs(p - mx) < self.epsilon)])
        return np.max(cluster_close), np.max(cluster_far)

    def _get_max_ankle_position(self, ankle_array):
        return np.amax(ankle_array)

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
