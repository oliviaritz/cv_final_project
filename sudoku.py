#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import math
import numpy as np

# source: https://stackoverflow.com/questions/44449871/fine-tuning-hough-line-function-parameters-opencv
def findIntersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py


def sortLines(lines):
    delta = 100
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        (x1, y1, x2, y2) = line[0]
        if abs(x1 - x2) < delta: # horizontal
            horizontal_lines.append(line[0])
        elif abs(y1 - y2) < delta: # vertical_lines
            vertical_lines.append(line[0])
    return vertical_lines, horizontal_lines


# horizontal if type == 0
# vertical if type == 1
def filterLines(lines, type):
    last_pos = 0
    new_lines = []

    for line in lines:
        value = line[type]
        if abs(value - last_pos) > 15:
            new_lines.append(line)
            last_pos = value

    return new_lines


# for each horizontal line, find its intersection with each vertical line
def getPoints(h_lines, v_lines):
    points = []
    for i in range (0, 10):
        (x1, y1, x2, y2) = h_lines[i]
        for j in range (0, 10):
            (x3, y3, x4, y4) = v_lines[j]
            intersection = findIntersection(h_lines[i], v_lines[j])
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
