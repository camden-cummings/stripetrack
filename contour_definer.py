# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:02:02 2025

@author: ThymeLab
"""

class ContourDefiner:
    def __init__(self):
        # method used to analyze fish
        self.cv_method = ""

        # threshold used for that method
        self.thresh = 155

        # minimum fish centroid size accepted
        self.centroid_size = 70
        
    def cv_alg_change(self, sender_id, data):
        self.cv_method = data
        
    def threshold_change(self, sender_id, data):
        self.thresh = data
        
    def centroid_change(self, sender_id, data):
        self.centroid_size = data