#!/usr/bin/env python3

import cv2
import numpy as np
import os.path
import puzzles
import solver

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
