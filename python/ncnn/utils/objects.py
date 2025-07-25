# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


class Point(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Rect(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def intersection_area(self, b):
        x1 = np.maximum(self.x, b.x)
        y1 = np.maximum(self.y, b.y)
        x2 = np.minimum(self.x + self.w, b.x + b.w)
        y2 = np.minimum(self.y + self.h, b.y + b.h)
        return np.abs(x1 - x2) * np.abs(y1 - y2)


class Detect_Object(object):
    def __init__(self, label=0, prob=0, x=0, y=0, w=0, h=0):
        self.label = label
        self.prob = prob
        self.rect = Rect(x, y, w, h)


class Face_Object(object):
    def __init__(self):
        self.prob = 0.0
        self.rect = Rect()
        self.landmark = []


class KeyPoint(object):
    def __init__(self):
        self.p = Point()
        self.prob = 0.0
