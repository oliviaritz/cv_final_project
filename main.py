#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os.path
import puzzles
import solver

blur_kernel = 3 # used for blurring images
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

# apply hough transform

# First parameter, Input image should be a binary image, so apply threshold edge detection before finding applying hough transform.
# Second and third parameters are r and θ(theta) accuracies respectively.
# Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line.
# Remember, number of votes depend upon number of points on the line. So it represents the minimum length of line that should be detected.

lines = cv2.HoughLines(edges_1, 2, np.pi/180, 300)
print('Number of lines: ' + str(len(lines)))

# loop through and plot lines
if (lines is not None):
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(board_1, (x1,y1), (x2,y2), (0,0,255), 2)

cv2.imwrite('test.JPG', board_1)




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
