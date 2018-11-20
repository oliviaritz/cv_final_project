#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os.path
import puzzles
import solver

blur_kernel = 5 # used for blurring images
canny_kernel = 5 # used for canny edge detection
ratio = 2
low_threshold = 30

# set up path to example sudoku puzzle images
my_path = os.path.abspath(os.path.dirname(__file__))
puzzle_path = os.path.join(my_path, "puzzles/")
board_1 = os.path.join(puzzle_path, "test1.JPG")
board_2 = os.path.join(puzzle_path, "test2.JPG")
board_3 = os.path.join(puzzle_path, "test3.JPG")

# read images in and convert to grayscale
board_1 = cv2.imread(board_1)
board_2 = cv2.imread(board_2)
board_3 = cv2.imread(board_3)

board_1_gray = cv2.cvtColor(board_1, cv2.COLOR_BGR2GRAY)
board_2_gray = cv2.cvtColor(board_2, cv2.COLOR_BGR2GRAY)
board_3_gray = cv2.cvtColor(board_3, cv2.COLOR_BGR2GRAY)


# refer to this source: https://caphuuquan.blogspot.com/2017/04/building-simple-sudoku-solver-from.html
# blur the images
board_1_gray = cv2.blur(board_1_gray, (blur_kernel, blur_kernel))
board_2_gray = cv2.blur(board_2_gray, (blur_kernel, blur_kernel))
board_3_gray = cv2.blur(board_3_gray, (blur_kernel, blur_kernel))

# detect edges
# First argument is our input image
# Second and third arguments are our minVal and maxVal respectively
# Third argument is aperture_size. It is the size of Sobel kernel used for find image gradients (default = 3)
# Last argument is L2gradient which specifies the equation for finding gradient magnitude
edges_1 = cv2.Canny(board_1_gray, low_threshold, low_threshold*ratio, canny_kernel)
edges_2 = cv2.Canny(board_2_gray, low_threshold, low_threshold*ratio, canny_kernel)
edges_3 = cv2.Canny(board_3_gray, low_threshold, low_threshold*ratio, canny_kernel)

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
lines = cv2.HoughLinesP(edges_1, 1, np.pi/180, 100, lines = None, minLineLength = 200, maxLineGap = 20).tolist()
# lines = cv2.HoughLines(edges_1, 1, np.pi/180, 200).tolist()

print('The number of lines detected is: ' + str(len(lines)))

# for each horizontal line, find all of the intersections for each vertical line
points = []

for line1 in lines:
    (x1, y1, x2, y2) = line1[0]
    # print('x1: ' + str(x1) + ' y1: ' + str(y1) + ' x2: ' + str(x2) + ' y2: ' + str(y2))
    index = 0
    for line2 in lines:
        (x3, y3, x4, y4) = line2[0]
        if y1==y2 and y3==y4: # Horizontal Lines
            diff = abs(y1-y3)
        elif x1==x2 and x3==x4: # Vertical Lines
            diff = abs(x1-x3)
        else:
            diff = 0
        if diff < 10 and diff is not 0:
            del lines[index]
        index = index + 1

# This number should be 20
print('The number of lines after filtering: ' + str(len(lines)))

points = []

for line1 in lines:
   (x1, y1, x2, y2) = line1[0]
   slope1 = abs(x1 - x2)
   if slope1 == 0: # horizontal lines
       for line2 in lines:
           (x3, y3, x4, y4) = line2[0]
           slope2 = abs(x3 - x4)
           if slope2 == 0:# only looking for vertical lines
               continue
           points.append([x1, y3])

for point in points:
    board_1[point[0], point[1]] = [0,0,255]

# This number should be 100
print('The number of points plotted is: ' + str(len(points)))

cv2.imwrite('test.JPG', board_1)
gridsize = (len(lines) - 2) / 2


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
