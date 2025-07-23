""" """

class ContourDefiner:
    def __init__(self):
        # method used to analyze fish
        self.cv_method = ""
        
        self.going_to_mode_method = False
        
        # threshold used for that method
        self.thresh = 155

        # minimum fish centroid size accepted
        self.centroid_size = 70
        
    def cv_alg_change(self, _, data):
        if "Mode" not in self.cv_method and "Mode" in data: # i.e. if going from non mode to mode
            self.going_to_mode_method = True
            
        self.cv_method = data
        
        
    def threshold_change(self, _, data):
        self.thresh = data
        
    def centroid_change(self, _, data):
        self.centroid_size = data