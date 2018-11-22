#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import os.path
import puzzles
import solver

blur_kernel = 5 # used for blurring images
canny_kernel = 3 # used for canny edge detection
dilate_kernel = np.ones((3,3),np.uint8)
erode_kernel = np.ones((5,5),np.uint8)
low_threshold = 50 # canny
high_threshold = 100 # canny
min_line_length = 200
max_line_gap = 100

# set up path to example sudoku puzzle images
my_path = os.path.abspath(os.path.dirname(__file__))
puzzle_path = os.path.join(my_path, "puzzles/")
board_1 = os.path.join(puzzle_path, "test1.JPG")
board_2 = os.path.join(puzzle_path, "test2.JPG")
board_3 = os.path.join(puzzle_path, "test3.JPG")
board_4 = os.path.join(puzzle_path, "test4.JPG")
board_5 = os.path.join(puzzle_path, "test5.JPG")

# read images in and convert to grayscale
board = cv2.imread(board_1)
board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

# refer to this source: https://caphuuquan.blogspot.com/2017/04/building-simple-sudoku-solver-from.html
# blur the images
board_gray = cv2.blur(board_gray, (blur_kernel, blur_kernel))

# detect edges
# First argument is our input image
# Second and third arguments are our minVal and maxVal respectively
# Third argument is aperture_size. It is the size of Sobel kernel used for find image gradients (default = 3)
# Last argument is L2gradient which specifies the equation for finding gradient magnitude
edges = cv2.Canny(board_gray, low_threshold, high_threshold, canny_kernel)
# edges = cv2.dilate(edges, dilate_kernel, iterations = 1)
# edges = cv2.erode(edges, erode_kernel, iterations = 1)

# https://stackoverflow.com/questions/19054055/python-cv2-houghlines-grid-line-detection
# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
# http://www.shogun-toolbox.org/static/notebook/current/Sudoku_recognizer.html

# apply hough transform
# image – 8-bit, single-channel binary source image. The image may be modified by the function.
# rho – Distance resolution of the accumulator in pixels.
# theta – Angle resolution of the accumulator in radians.
# threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
# minLineLength – Minimum line length. Line segments shorter than that are rejected.
# maxLineGap – Maximum allowed gap between points on the same line to link them.

# lines is a list of lists
# each list in lines only contains one element: the two endpoints of the line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, lines = None, minLineLength = min_line_length, maxLineGap = max_line_gap).tolist()
# lines = cv2.HoughLines(edges, 1, np.pi/180, 200).tolist()

num_lines = len(lines)
print('The number of lines detected in the image is: ' + str(num_lines))

# for each horizontal line, find all of the intersections for each vertical line
points = []

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

# This number should be 20
num_lines = len(lines)

'''
for line in lines:
    (x1, y1, x2, y2) = line[0]
    cv2.line(board,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('test.JPG', board)
'''

if (num_lines == 20):
    print('Redundant lines successfully filtered out...')
else:
    print(str(len(lines)) + ' lines remain after filtering out redundant lines. ' \
        'However, there should be 20 lines. Program exiting...')
    exit()

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
                points.append((x1, y3))

num_points = len(points)
# This number should be 100
if num_points == 100:
    print('Intersection points located...')
else:
    print(str(num_points) + ' points were found. ' \
        'However, there should be 100 points. Program exiting...')
    exit()

'''
for point in points:
    board[point[0], point[1]] = [0,0,255]
'''

points.sort()
boxes = [] # will contain 81 tuples of four corner coordinates

m = 0
for i in range(81):
    new_row = i%9
    if new_row == 0 and i !=0:
        m += 1
    a = i%(9) + 10*m
    b = a + 1
    box = (points[a], points[b], points[a+10], points[b+10])
    boxes.append(box)

# This number should be 81
num_boxes = len(boxes)
if num_boxes == 81:
    print('Box coordinates located...')
else:
    print(str(num_boxes) + ' boxes were found. ' \
        'However, there should be 81 boxes. Program exiting...')
    exit()

for box in boxes:
    cv2.rectangle(board, box[0], box[3], (0,255,0), 2)

cv2.imwrite('test.JPG', board)


# Example code on how to use solver.py

"""

test1_theoretical = solver.solve(puzzles.test_1)
test1_actual = solver.parse_grid(puzzles.sol_1)
test2_theoretical = solver.solve(puzzles.test_2)
test2_actual = solver.parse_grid(puzzles.sol_2)
test3_theoretical = solver.solve(puzzles.test_3)
test3_actual = solver.parse_grid(puzzles.sol_3)


if (test1_theoretical == test1_actual):
    print('Test 1 has passed')

if (test2_theoretical == test2_actual):
    print('Test 2 has passed')

if (test3_theoretical == test3_actual):
    print('Test 3 has passed')

"""
