from time import time, perf_counter, get_clock_info, localtime
import datetime

class PreciseTime:
    """Timer which tries to find most precise method of timing."""
    # being paranoid and pulling some nice time code from here: https://github.com/hardbyte/python-can/pull/936
    # to double check that time stamps will be as correct as possible

    def __init__(self):
        # use this if the resolution is higher than 10us
        if get_clock_info("time").resolution > 1e-5:
            t0 = time()
            while True:
                t1, performance_counter = time(), perf_counter()
                if t1 != t0:
                    break
            self.time = t1
            self.perfcounter = performance_counter
        else:
            self.time = time()
            self.perfcounter = None  # not needed
        
        now = datetime.datetime.now()
        self.diff = int((now.hour*3600+now.minute*60+now.second) - (self.time % 86400))
        self.time += self.diff

    @staticmethod
    def formatted_time(input_time):
        l = localtime(input_time)
        return[l.tm_hour, l.tm_min, l.tm_sec]
    
    def now(self) -> float:
        """Finds current time according to best timer."""
        if self.perfcounter is None:
            return time()
        return self.time + (perf_counter() - self.perfcounter) + self.diff

