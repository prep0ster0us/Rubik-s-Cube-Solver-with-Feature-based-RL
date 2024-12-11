from numpy import pi
from vpython import *
import random

# size of rubik cube's (side x side x side)
SIDE = 3 

cubeOrigin = vector(0,0,0)
cubeSize = vector(SIDE, SIDE, SIDE)

FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM = "front", "back", "left", "right", "top", "bottom"

"""
CUBE COLORS

FRONT = Red, BACK  = Orange
LEFT = White, RIGHT = Yellow
TOP/UP = Blue, BOTTOM/DOWN = Green

NOTE that the faces and moves/actions are relative to this layout of the cube.
"""

directions = [FRONT, BACK, TOP, BOTTOM, LEFT, RIGHT]

possibleMoves = {
    "Front"  : "front_clock",
    "Front'" : "front_counter",
    "Right"  : "right_clock",
    "Right'" : "right_counter",
    "Back"   : "back_clock",
    "Back'"  : "back_counter",
    "Left"   : "left_clock",
    "Left'"  : "left_counter",
    "Top"    : "top_clock",
    "Top'"   : "top_counter",
    "Bottom" : "bottom_clock",
    "Bottom'": "bottom_counter",
}

# x-axis = right (+1) and left (-1)
# y-axis = top (+1) and bottom (-1)
# z-axis = front (+1) and back (-1)
cubeAxis = {
    "front_clock"   : vector(0,0,1),
    "front_counter" : vector(0,0,1),
    "back_counter"  : vector(0,0,-1),
    "back_clock"    : vector(0,0,-1),
    "top_clock"     : vector(0,1,0),
    "top_counter"   : vector(0,1,0),
    "bottom_clock"  : vector(0,-1,0),
    "bottom_counter": vector(0,-1,0),
    "left_clock"    : vector(-1,0,0),
    "left_counter"  : vector(-1,0,0),
    "right_clock"   : vector(1,0,0),
    "right_counter" : vector(1,0,0)
}

# map moves for the solution sequence provided by kociemba
mapMoves = {
    "F"  : "front_clock",
    "F'" : "front_counter",
    "R"  : "right_clock",
    "R'" : "right_counter",
    "B"  : "back_clock",
    "B'" : "back_counter",
    "L"  : "left_clock",
    "L'" : "left_counter",
    "U"  : "top_clock",
    "U'" : "top_counter",
    "D"  : "bottom_clock",
    "D'" : "bottom_counter",
}

# map the position of each tile on the cube to the value (move) list
# needed for input to kociemba library
mapTile = {
    0  : (-1, 1.5, -1),     # U1
    1  : (0, 1.5, -1),      # U2
    2  : (1, 1.5, -1),      # U3
    3  : (-1, 1.5, 0),      # U4
    4  : (0, 1.5, 0),       # U5
    5  : (1, 1.5, 0),       # U6
    6  : (-1, 1.5, 1),      # U7
    7  : (0, 1.5, 1),       # U8
    8  : (1, 1.5, 1),       # U8

    9  : (1.5, 1, -1),      # R1
    10 : (1.5, 1, 0),       # R2
    11 : (1.5, 1, 1),       # R3
    12 : (1.5, 0, -1),      # R4
    13 : (1.5, 0, 0),       # R5
    14 : (1.5, 0, 1),       # R6
    15 : (1.5, -1, -1),     # R7
    16 : (1.5, -1, 0),      # R8
    17 : (1.5, -1, 1),      # R9

    18 : (-1, 1, 1.5),      # F1
    19 : (0, 1, 1.5),       # F2
    20 : (1, 1, 1.5),       # F3
    21 : (-1, 0, 1.5),      # F4
    22 : (0, 0, 1.5),       # F5
    23 : (1, 0, 1.5),       # F6
    24 : (-1, -1, 1.5),     # F7
    25 : (0, -1, 1.5),      # F8
    26 : (1, -1, 1.5),      # F9

    27 : (-1, -1.5, -1),    # D1
    28 : (0, -1.5, -1),     # D2
    29 : (1, -1.5, -1),     # D3
    30 : (-1, -1.5, 0),     # D4
    31 : (0, -1.5, 0),      # D5
    32 : (1, -1.5, 0),      # D6
    33 : (-1, -1.5, 1),     # D7
    34 : (0, -1.5, 1),      # D8
    35 : (1, -1.5, 1),      # D9

    36 : (-1.5, 1, -1),     # L1
    37 : (-1.5, 1, 0),      # L2
    38 : (-1.5, 1, 1),      # L3
    39 : (-1.5, 0, -1),     # L4
    40 : (-1.5, 0, 0),      # L5
    41 : (-1.5, 0, 1),      # L6
    42 : (-1.5, -1, -1),    # L7
    43 : (-1.5, -1, 0),     # L8
    44 : (-1.5, -1, 1),     # L9

    45 : (-1, 1, -1.5),     # B1
    46 : (0, 1, -1.5),      # B2
    47 : (1, 1, -1.5),      # B3
    48 : (-1, 0, -1.5),     # B4
    49 : (0, 0, -1.5),      # B5
    50 : (1, 0, -1.5),      # B6
    51 : (-1, -1, -1.5),    # B7
    52 : (0, -1, -1.5),     # B8
    53 : (1, -1, -1.5)      # B9
}