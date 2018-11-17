#!/usr/bin/env python3

import solver

# The first 9 characters form the first row, the second 9 characters form the second row, and so on.
# All dots are interpreted as empty square.

grid1 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'

grid1_solution = solver.solve(grid1)
print(grid1_solution)
