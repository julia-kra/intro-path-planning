# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPerfMonitor import IPPerfMonitor

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, LineString
from shapely import plotting

import numpy as np

class CollisionChecker(object):
    def __init__(self, scene, limits=[[0.0, 22.0], [0.0, 22.0]], statistic=None):
        self.scene = scene
        self.limits = limits
        self.counter = 0  # Zähler für Kollisionsprüfungen

    def getDim(self):
        """ Return dimension of Environment (Shapely should currently always be 2)"""
        return 2

    def getEnvironmentLimits(self):
        """ Return limits of Environment"""
        return list(self.limits)

    #@IPPerfMonitor
    def pointInCollision(self, pos):
        """ Return whether a configuration is
        inCollision -> True
        Free -> False """
        assert (len(pos) == self.getDim())
        for key, value in self.scene.items():
            self.counter += 1  # Zähler erhöhen bei jeder Kollision
            if value.intersects(Point(pos[0], pos[1])):
                return True
        return False

    #@IPPerfMonitor
    def lineInCollision(self, startPos, endPos):
        """ Check whether a line from startPos to endPos is colliding"""
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())

        p1 = np.array(startPos)
        p2 = np.array(endPos)
        p12 = p2 - p1
        k = 40

        for i in range(k):
            testPoint = p1 + (i + 1) / k * p12
            self.counter += 1  # Zähler erhöhen bei jeder Kollisionsprüfung
            if self.pointInCollision(testPoint):
                return True

        return False

    def drawObstacles(self, ax):
        for key, value in self.scene.items():
            plotting.plot_polygon(value, ax=ax, add_points=False, color='red')

