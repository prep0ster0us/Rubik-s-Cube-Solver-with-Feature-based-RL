import numpy as np
import random

class RubiksCube:
    def __init__(self, size=3):
        self.size = size
        self.cube = self._initialize_cube()
        self.actions = ["U", "U'", "R", "R'", "F", "F'", "L", "L'", "B", "B'", "D", "D'"]

    def _initialize_cube(self):
        # Use face labels as keys
        face_labels = ['U', 'L', 'F', 'R', 'B', 'D']  # Up, Left, Front, Right, Back, Down
        return {face: np.full((self.size, self.size), color) for face, color in zip(face_labels, "WYROGB")}

    def rotate(self, action):
        """
        Perform a simplified rotation on the cube.
        Detailed implementation required for face and neighbor rotations.
        """
        print(f"Rotating {action} (Not yet implemented).")

    def scramble(self, moves=20):
        """
        Randomly scramble the cube with a specified number of moves.
        """
        for _ in range(moves):
            action = random.choice(self.actions)
            self.rotate(action)

    def is_solved(self):
        return all(np.all(self.cube[face] == self.cube[face][0, 0]) for face in self.cube)

    def cube_to_state(self):
        return tuple("".join(self.cube[face].flatten()) for face in self.cube)

    def execute_action(self, action):
        self.rotate(action)
        reward = 1 if self.is_solved() else -1
        return self.cube_to_state(), reward
