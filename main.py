#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import operator
import pickle
import puzzles
import solver
import sudoku

blur_kernel = 5 # used for blurring images
bilateral_kernel = 5 # used for blurring images
canny_kernel = 5 # used for canny edge detection
dilate_kernel = np.ones((3,3), np.uint8) # used for dilating edges
low_threshold = 30 # canny
high_threshold = 90 # canny
min_line_length = 300 # hough
max_line_gap = 50 # hough
min_votes = 200  # hough


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # convert current frame to grayscale
    board_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', board_gray)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # may want to try incorporating thresholding
    # retval, thresh = cv2.threshold(board_gray, 75, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('thresh.JPG', thresh)

    #board_gray = cv2.bilateralFilter(board_gray, bilateral_kernel, 75, 75) # bilateral filter preserves edges
    #board_gray = cv2.blur(board_gray, (blur_kernel, blur_kernel))
    board_gray = cv2.GaussianBlur(board_gray, (blur_kernel, blur_kernel), 0)

    # detect edges
    # First argument is our input image
    # Second and third arguments are our minVal and maxVal respectively
    # Third argument is aperture_size. It is the size of Sobel kernel used for find image gradients (default = 3)
    edges = cv2.Canny(board_gray, low_threshold, high_threshold, canny_kernel)
    cv2.imwrite('edges.JPG', edges)

    # may want to try incorporating edge dilation
    # img_dilation = cv2.dilate(edges, dilate_kernel, iterations=1)
    # cv2.imwrite('dilate.JPG', img_dilation)

    # apply hough transform
    # image – 8-bit, single-channel binary source image. The image may be modified by the function.
    # rho – Distance resolution of the accumulator in pixels.
    # theta – Angle resolution of the accumulator in radians.
    # threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
    # minLineLength – Minimum line length. Line segments shorter than that are rejected.
    # maxLineGap – Maximum allowed gap between points on the same line to link them.
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, min_votes, lines = None, minLineLength = min_line_length, maxLineGap = max_line_gap)

    # check if any lines were detected
    if lines is None:
        continue # try again
    else:
        lines = lines.tolist()

    num_lines = len(lines)
    print('The number of lines detected in the image is: ' + str(num_lines))

    # get horizontal lines and vertical lines
    vertical_lines, horizontal_lines = sudoku.sortLines(lines)

    # sort vertical lines by y1 values
    vertical_lines.sort(key=operator.itemgetter(1))

    # sort horizontal lines by x1 values
    horizontal_lines.sort(key=operator.itemgetter(0))

    # horizontal if type == 0
    # vertical if type == 1
    vertical_lines_filtered = sudoku.filterLines(vertical_lines, 1)
    horizontal_lines_filtered = sudoku.filterLines(horizontal_lines, 0)

    # check if there are 20 lines
    if (len(vertical_lines_filtered) == 10 & len(horizontal_lines_filtered) == 10):
        print('Redundant lines successfully filtered out...')
    else:
        continue # try again

    # used for debugging
    '''
    for line in vertical_lines_filtered:
        (x1, y1, x2, y2) = line
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite('v_lines.JPG', frame)

    for line in horizontal_lines_filtered:
        (x1, y1, x2, y2) = line
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite('h_lines.JPG', frame)
    '''

    # get line intersection points
    points = sudoku.getPoints(horizontal_lines_filtered, vertical_lines_filtered)
    num_points = len(points)

    # check if there are 100 points
    if num_points == 100:
        print('Intersection points located...')
    else:
        continue # try again

    # used for debugging
    '''
    for point in points:
        frame[point[0], point[1]] = [0,0,255]

    cv2.imwrite('test.JPG', frame)
    '''

    # order points so that the first element is the top left corner of the board
    # and the last element is the bottom right corner of the board
    points.sort(key=operator.itemgetter(1))

    boxes = sudoku.getBoxes(points)
    num_boxes = len(boxes)

    # exit if there are not exactly 81 boxes
    if num_boxes == 81:
        print('Box coordinates located...')
    else:
        continue # try again

    cv2.imwrite('test.JPG', frame)

    for box in boxes:
        cv2.rectangle(frame, box[0], box[3], (0,255,0), 2)
        cv2.imshow('boxes', frame)
        cv2.waitKey(50)

    with open('boxes.txt', 'wb') as fp:
        pickle.dump(boxes, fp)

    break
