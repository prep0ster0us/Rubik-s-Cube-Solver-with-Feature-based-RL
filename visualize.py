import matplotlib.pyplot as plt
import numpy as np
from cube import RubiksCube
# visualize.py
from manim import *
from cube_3d import RubiksCube3D

def visualize_3d_solution():
    # This function will be used to run and render the 3D animation.
    scene = RubiksCube3D()
    scene.render()

if __name__ == "__main__":
    visualize_3d_solution()


def draw_cube(cube_state):
    face_positions = {
        "U": (0, 1),
        "L": (1, 0),
        "F": (1, 1),
        "R": (1, 2),
        "B": (1, 3),
        "D": (2, 1),
    }
    colors = {"W": "white", "Y": "yellow", "R": "red", "O": "orange", "G": "green", "B": "blue"}

    fig, ax = plt.subplots()
    for face, pos in face_positions.items():
        x_offset, y_offset = pos
        for i in range(cube_state[face].shape[0]):
            for j in range(cube_state[face].shape[1]):
                color = colors[cube_state[face][i, j]]
                rect = plt.Rectangle(
                    (3 * x_offset + j, -3 * y_offset - i), 1, 1, facecolor=color, edgecolor="black"
                )
                ax.add_patch(rect)

    ax.set_xlim(-1, 12)
    ax.set_ylim(-10, 2)
    ax.set_aspect("equal")
    plt.axis("off")
    plt.show()

def visualize_solution(cube, solution):
    """
    Visualize the solution step-by-step.
    cube: The RubiksCube object.
    solution: List of actions to solve the cube.
    """
    for action in solution:
        cube.rotate(action)
        draw_cube(cube.cube)
        plt.pause(0.5)

from visualize import draw_cube, visualize_solution

if __name__ == "__main__":
    # Initialize a 3x3x3 Rubik's Cube
    rubiks_cube = RubiksCube(size=3)

    # Scramble the cube
    rubiks_cube.scramble(moves=20)

    # Draw the scrambled cube
    print("Scrambled Cube:")
    draw_cube(rubiks_cube.cube)

    # Example solution (replace with your RL agent's solution later)
    solution = ["U", "U'", "F", "F'", "R", "R'"]

    print("Visualizing Solution:")
    visualize_solution(rubiks_cube, solution)
