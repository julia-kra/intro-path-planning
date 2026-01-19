# coding: utf-8

"""
This code is part of a series of notebooks regarding  "Introduction to robot path planning".

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
(pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPerfMonitor import IPPerfMonitor

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, LineString
from shapely import plotting

import numpy as np


class CollisionChecker(object):
    def __init__(
        self,
        scene,
        limits=[[0.0, 22.0], [0.0, 22.0]],
        statistic=None,
        min_clearance=0.0,        # NEU: Mindestabstand (Default 0.0 = Original)
        oob_is_collision=True,    # NEU: Out-of-bounds als Kollision
        line_check_steps=40       # optional: falls du Sampling statt LineString nutzen willst
    ):
        # scene: dict {name: shapely-geometry}
        self.scene = scene
        self.limits = limits
        self.counter = 0
        self.statistic = statistic

        self.min_clearance = float(min_clearance)
        self.oob_is_collision = bool(oob_is_collision)
        self.line_check_steps = int(line_check_steps)

    def getDim(self):
        return 2

    def getEnvironmentLimits(self):
        return list(self.limits)

    def _out_of_bounds(self, pos):
        (xmin, xmax), (ymin, ymax) = self.getEnvironmentLimits()
        x, y = float(pos[0]), float(pos[1])
        return (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax)

    def pointInCollision(self, pos):
        """
        True = Kollision, False = frei
        Mit min_clearance:
          - Kollision, wenn distance(obstacle, point) <= min_clearance
        """
        assert (len(pos) == self.getDim())

        # OOB als Kollision
        if self.oob_is_collision and self._out_of_bounds(pos):
            self.counter += 1
            return True

        p = Point(float(pos[0]), float(pos[1]))
        c = self.min_clearance

        for _, geom in self.scene.items():
            self.counter += 1
            if c <= 0.0:
                if geom.intersects(p):
                    return True
            else:
                if geom.distance(p) <= c:
                    return True

        return False

    def lineInCollision(self, startPos, endPos):
        """
        Kantenprüfung mit Mindestabstand:
          - Kollision, wenn distance(obstacle, line) <= min_clearance
        Das ist deutlich robuster als punktweises Abtasten.
        """
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())

        # OOB optional
        if self.oob_is_collision:
            if self._out_of_bounds(startPos) or self._out_of_bounds(endPos):
                self.counter += 1
                return True

        c = self.min_clearance
        line = LineString([
            (float(startPos[0]), float(startPos[1])),
            (float(endPos[0]),   float(endPos[1]))
        ])

        for _, geom in self.scene.items():
            self.counter += 1
            if c <= 0.0:
                if geom.intersects(line):
                    return True
            else:
                # Abstand der GANZEN Kante zum Hindernis
                if geom.distance(line) <= c:
                    return True

        return False

    def drawObstacles(self, ax):
        for _, geom in self.scene.items():
            plotting.plot_polygon(geom, ax=ax, add_points=False, color='red')