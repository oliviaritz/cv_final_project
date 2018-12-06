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
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from six.moves import urllib
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


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
    board_blur = cv2.GaussianBlur(board_gray, (blur_kernel, blur_kernel), 0)

    # detect edges
    edges = cv2.Canny(board_gray, low_threshold, high_threshold, canny_kernel)
    cv2.imwrite('edges.JPG', edges)

    # may want to try incorporating edge dilation
    # img_dilation = cv2.dilate(edges, dilate_kernel, iterations=1)
    # cv2.imwrite('dilate.JPG', img_dilation)

    # apply hough transform
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

    cv2.imwrite('frame.JPG', frame)

    for box in boxes:
        cv2.rectangle(frame, box[0], box[3], (0,255,0), 2)
        cv2.imshow('boxes', frame)
        cv2.waitKey(40)

    break

cap.release()

# *** start digit detection ***
rects = boxes
#im = frame

#load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
#im = cv2.imread("test2.jpg")

# Convert to grayscale and apply Gaussian filtering
#im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#cv2.imshow('gray', im_gray)

# Blur & Threshold
#img = cv2.GaussianBlur(board_gray, (5, 5), cv2.BORDER_DEFAULT)
img = cv2.GaussianBlur(board_gray, (5, 5), 0)
#cv2.imshow('blur', img)
im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#im_th = cv2.Canny(img, low_threshold, high_threshold, canny_kernel)
#im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations = 2)
#cv2.imshow('thresh1', im_th)

kernel = np.ones((2, 2), np.uint8)
im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel, iterations = 1)
#im_th = cv2.dilate(im_th, kernel)
im_th = cv2.erode(im_th, kernel, iterations = 1)
#im_th = cv2.dilate(im_th, kernel)
#cv2.imshow('thresh2', im_th)

# Get box coordinates
#with open('boxes2.txt', 'rb') as fp:
 #   boxes = pickle.load(fp)

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
i = 0

detected = "";
for rect in rects:
    i = i + 1
    #print(rect)
    x1 = rect[0][0]
    #print(x1)
    y1 = rect[0][1]
    #print(y1)
    x2= rect[3][0]
    y2 = rect[3][1]

    # Draw the rectangles
    cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 3)

    x_length = x2 - x1;
    y_length = y2 - y1;
    roi = im_th[y1 + (y_length // 7):y2 - (y_length // 6), x1 + (x_length // 6):x2 - (x_length // 8)]
    #roi = im_th[y1+7:y2-8, x1+9:x2-5]

    # Resize the image
    img_size = roi.size
    if(img_size > 0) :
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        #roi = cv2.erode(roi, kernel, iterations=1)
        #roi = cv2.dilate(roi, (2, 2), iterations = 1)
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=11, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm = 'L1') # 6 8 2

        #predict
        digit = "."
        if(cv2.countNonZero(roi) > 36):
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            digit = str(int(nbr[0]))
            cv2.imshow(digit, roi)
            #annotate
            cv2.putText(frame, str(int(nbr[0])), ((x1+x2)//2, (y2+y1)//2),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        #append to the result string
        detected = detected + digit

#cv2.imwrite('Annotated.jpg', im)
#cv2.imwrite('Thresh.jpg', im_th)
#cv2.imwrite('')
cv2.imshow("Resulting Image with Rectangular ROIs", frame)


if (len(detected) == 81):
	print(detected[0:8])
	print(detected[9:17])
	print(detected[18:26])
	print(detected[27:35])
	print(detected[36:44])
	print(detected[45:53])
	print(detected[54:62])
	print(detected[63:71])
	print(detected[72:80])
else:
	print('Not 81 digits detected')
	print('Using hardcoded result')
	detected = '.254...398.13962..3..752.41618....57.5.16748...45839.65372.9.6848..7159.1...35724'

result = solver.solve(detected)

if result == False:
	print('Puzzle input invalid. Unable to solve.')
	print('Using hardcoded result')
	detected = '.254...398.13962..3..752.41618....57.5.16748...45839.65372.9.6848..7159.1...35724'
	result = solver.solve(detected)

#print('result:' + str(result))


# ****** start of output ****
img = Image.open("frame.JPG")
draw = ImageDraw.Draw(img)

# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("arial.ttf", 36)

# set up dict key words
digits = '123456789'
nums = 'ABCDEFGHI'
keys = []

for n in range(9):
	for d in range(9):
		new = nums[n] + digits[d]
		keys.append(new)

# set up detected list
detected_list = []
for s in detected:
	detected_list.append(s)

#print('detected_list:' + str(detected_list))

for i in range(81):
	# shift to be centered in box
	topleftx = boxes[i][0][0]
	toprightx = boxes[i][1][0]
	difx = (toprightx - topleftx) / 4	# divide by 4 because of how digit is written
	newx = topleftx + difx

	toplefty = boxes[i][0][1]
	botlefty = boxes[i][2][1]
	dify = (botlefty - toplefty) / 8	# divide by 8 because of how digit is written
	newy = toplefty + dify

	k = keys[i]
	#print('k:' + k)
	#print('result[k]:' + result[k])


	# draw new value onto image
	if detected_list[i] != '.':
		draw.text((newx, newy),str(result[k]),(150,0,0),font=font)


img.save('test-out.jpg')
img.show('test-out.jpg')

cv2.waitKey()
