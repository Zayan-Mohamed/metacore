"""Competence-drop trigger: fires when epistemic u crosses a calibrated threshold,
with hysteresis so it does not chatter around the boundary."""
import numpy as np

class CompetenceDropTrigger:
    def __init__(self, threshold, hysteresis=2):
        self.thr = threshold; self.h = hysteresis
        self._up = 0; self._down = 0; self.state = False
    @classmethod
    def calibrate(cls, id_u, false_alarm_rate=0.05, hysteresis=2):
        thr = float(np.quantile(id_u, 1.0-false_alarm_rate))
        return cls(thr, hysteresis)
    def update(self, u):
        if u > self.thr:
            self._up += 1; self._down = 0
            if self._up >= self.h: self.state = True
        else:
            self._down += 1; self._up = 0
            if self._down >= self.h: self.state = False
        return self.state
