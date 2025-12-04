""" """

class ContourDefiner:
    def __init__(self):
        # method used to analyze fish
        self.cv_method = ""
        
        self.going_to_mode_method = False

        # threshold used for that method
        self.thresh = 155

        # minimum fish centroid size accepted
        self.min_centroid_size = 70

        self.dist = 50

        self.sigma = 1.5
        self.truncate = 3.5

        self.white_thresh = 200

    def cv_alg_change(self, _, data):
        if "Mode" not in self.cv_method and "Mode" in data: # i.e. if going from non mode to mode
            self.going_to_mode_method = True
            
        self.cv_method = data

    def truncate_change(self, _, data):
        self.truncate = data

    def sigma_change(self, _, data):
        self.sigma = data

    def threshold_change(self, _, data):
        self.thresh = data
        
    def centroid_change(self, _, data):
        self.min_centroid_size = data

    def dist_change(self, _, data):
        self.dist = data

    def white_thresh_change(self, _, data):
        self.white_thresh = data