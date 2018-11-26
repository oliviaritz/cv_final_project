#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import puzzles
import solver
import sudoku

blur_kernel = 5 # used for blurring images
bilateral_kernel = 5 # used for blurring images
canny_kernel = 5 # used for canny edge detection
low_threshold = 50 # canny
high_threshold = 100 # canny
min_line_length = 200 # hough
max_line_gap = 200 # hough
min_votes = 200  # hough

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # read images in and convert to grayscale
    # board = cv2.imread(board_2_photo)
    board_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', board_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #board_gray = cv2.bilateralFilter(board_gray, bilateral_kernel, 75, 75) # bilateral filter preserves edges
    #board_gray = cv2.blur(board_gray, (blur_kernel, blur_kernel))
    board_gray = cv2.GaussianBlur(board_gray, (blur_kernel, blur_kernel), 0)

    # detect edges
    # First argument is our input image
    # Second and third arguments are our minVal and maxVal respectively
    # Third argument is aperture_size. It is the size of Sobel kernel used for find image gradients (default = 3)
    edges = cv2.Canny(board_gray, low_threshold, high_threshold, canny_kernel)
    cv2.imwrite('edges.JPG', edges)


    # apply hough transform
    # image – 8-bit, single-channel binary source image. The image may be modified by the function.
    # rho – Distance resolution of the accumulator in pixels.
    # theta – Angle resolution of the accumulator in radians.
    # threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
    # minLineLength – Minimum line length. Line segments shorter than that are rejected.
    # maxLineGap – Maximum allowed gap between points on the same line to link them.

    # lines is a list of lists
    # each list in lines only contains one element: the two endpoints of the line
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, min_votes, lines = None, minLineLength = min_line_length, maxLineGap = max_line_gap)

    if lines is None:
        continue # try again
    else:
        lines = lines.tolist()

    num_lines = len(lines)
    print('The number of lines detected in the image is: ' + str(num_lines))

    lines = sudoku.filterLines(lines)
    num_lines = len(lines)

    # used for debugging
    '''
    for line in lines:
        (x1, y1, x2, y2) = line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite('lines_after.JPG', frame)
    '''

    # exit if there are not exactly 20 lines
    if (num_lines == 20):
        print('Redundant lines successfully filtered out...')
    else:
        continue # try again


    points = sudoku.getPoints(lines)
    num_points = len(points)

    # exit if there are not exactly 100 points
    if num_points == 100:
        print('Intersection points located...')
    else:
        continue # try again

    # used for debugging
    '''
    for point in points:
        frame[point[0], point[1]] = [0,0,255]
    '''

    # order points so that the first element is the top left corner of the board
    # and the last element is the bottom right corner of the board
    points.sort()

    boxes = sudoku.getBoxes(points)
    num_boxes = len(boxes)

    # exit if there are not exactly 81 boxes
    if num_boxes == 81:
        print('Box coordinates located...')
    else:
        continue # try again

    # if the program has made it this far break and show solution
    for box in boxes:
        cv2.rectangle(frame, box[0], box[3], (0,255,0), 2)

    cv2.imwrite('test.JPG', frame)

    cap.release()
    cv2.destroyAllWindows()
    break

cv2.imshow('frame', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    exit()





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
