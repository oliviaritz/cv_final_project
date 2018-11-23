#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import math
import numpy as np

# source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
def getItersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# for each horizontal line, find all of the intersections for each vertical line
def filterLines(lines):
    for line1 in lines:
        (x1, y1, x2, y2) = line1[0]
        delta_y1 = abs(y1 - y2)
        delta_x1 = abs(x1 - x2)

        if delta_y1 == 0:
            slope1 = math.inf
        else:
            slope1 = delta_x1/delta_y1

        index = 0
        for line2 in lines:
            (x3, y3, x4, y4) = line2[0]
            delta_y2 = abs(y3 - y4)
            delta_x2 = abs(x3 - x4)

            if delta_y2 == 0:
                slope2 = math.inf
            else:
                slope2 = delta_x2/delta_y2

            if slope1 < 1 and slope2 < 1: # both lines horizontal
                diff = abs(x1 - x3)
            elif slope1 >= 1 and slope2 >= 1: # both lines vertical
                diff = abs(y1 - y3)
            else:
                diff = 0
            if diff < 25 and diff is not 0:
                del lines[index]
            index = index + 1
    return lines


# for each horizontal line, find its intersection with each vertical line
def getPoints(lines):
    points = []
    for line1 in lines:
        (x1, y1, x2, y2) = line1[0]
        delta_y1 = abs(y1 - y2)
        delta_x1 = abs(x1 - x2)
        if delta_y1 == 0:
            slope1 = math.inf
        else:
            slope1 = delta_x1/delta_y1
        if slope1 < 1: # horizontal line if angle is < 45 degrees
            for line2 in lines:
                (x3, y3, x4, y4) = line2[0]
                delta_y2 = abs(y3 - y4)
                delta_x2 = abs(x3 - x4)
                if delta_y2 == 0:
                    slope2 = math.inf
                else:
                    slope2 = delta_x2/delta_y2
                if slope2 >= 1:# only looking for vertical lines
                    intersection = getItersection(((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)))
                    points.append((int(intersection[0]), int(intersection[1])))
    return points


# form a tuple of four coordinates and append it to a list of boxes
def getBoxes(points):
    m = 0
    boxes = []
    for i in range(81):
        new_row = i%9
        if new_row == 0 and i !=0:
            m += 1
        a = i%(9) + 10*m
        b = a + 1
        box = (points[a], points[b], points[a+10], points[b+10])
        boxes.append(box)
    return boxes
