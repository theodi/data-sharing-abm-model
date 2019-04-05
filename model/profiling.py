import time
import numpy as np


class Timer(object):
    """
    Timer object follows one (set of) instructions and collates the run time, reports mean
    """
    def __init__(self, name):
        self._name = name
        self._start_epoch = None
        self._stop_epoch = None
        self._elapsed_times = []
        self._summary = None

    def start(self):
        if self._start_epoch:
            print("timer " + self._name + " already running; restarting")
        assert self._start_epoch is None, (
            "timer " + self._name + " not stopped before starting"
        )
        self._start_epoch = time.time()

    def stop(self):
        self._stop_epoch = time.time()
        self._elapsed_times.append(self._stop_epoch - self._start_epoch)
        self._start_epoch = None
        self._stop_epoch = None

    def summarize(self):
        self._summary = np.array(self._elapsed_times).mean()

    def __str__(self):
        self.summarize()
        if np.isnan(self._summary):
            return self._name + " : not called"
        else:
            return (
                self._name
                + " ("
                + str(len(self._elapsed_times))
                + ")"
                + " : "
                + str(self._summary)
            )


class LotsOfTimers(object):
    """
    An class to bundle multiple timers
    """
    def __init__(self, names=None):
        if names:
            self._timers = {n: Timer(n) for n in names}
        else:
            self._timers = {}

    def add(self, name):
        self._timers[name] = Timer(name)

    def start(self, name):
        if name not in self._timers.keys():
            print("adding timer : " + name)
            self.add(name)
        self._timers[name].start()

    def stop(self, name):
        self._timers[name].stop()

    def stop_all(self):
        for timer in self._timers.values():
            timer.stop()

    def __str__(self):
        if len(self._timers) > 0:
            return "\n".join([str(x) for x in self._timers.values()])
        else:
            return "no timers"
