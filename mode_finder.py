from multiprocessing import Pool
# import cProfile, pstats, io
# from pstats import SortKey

from strsim_for_speed.computer_vision.helpers import calc_mode


global continue_recording
global run_once

run_once = True

DESIRED_MODE_FRAMES = 2#50
FRAMES_TO_SAVE_AFTER = 2#100

class ModeFinder:
    def __init__(self, FRAME_WIDTH, FRAME_HEIGHT):
        self.async_result = None

        self.mode_noblur_img = None
        self.prev_mode_noblur_img = None

        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.FRAME_WIDTH = FRAME_WIDTH

        self.movie_deq = []

        self.found_mode = False
        self.mode_updated = False
        
    def find_mode(self, frame_counter, image):
        global run_once
        
        if len(self.movie_deq) < DESIRED_MODE_FRAMES and frame_counter % 50 == 0:
            self.movie_deq.append(image)
        elif len(self.movie_deq) >= DESIRED_MODE_FRAMES and run_once == True:
            print("start")
            pool = Pool(processes=1)
            self.async_result = pool.apply_async(calc_mode, (self.movie_deq, self.FRAME_HEIGHT, self.FRAME_WIDTH))
            run_once = False
       
        if self.async_result is not None and self.async_result.ready():
            self.prev_mode_noblur_img = self.mode_noblur_img
            self.mode_noblur_img = self.async_result.get()
            self.movie_deq.clear()
            self.mode_updated=True
            self.async_result=None
            
            if self.prev_mode_noblur_img is None and self.mode_noblur_img is not None:
                self.found_mode = False
            else:
                self.found_mode = True

            run_once = True
            