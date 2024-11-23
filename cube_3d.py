from manim import *
import random


class RubiksCube3D(ThreeDScene):
    def construct(self):
        # Define colors for cube faces
        face_colors = {
            "U": YELLOW,
            "D": WHITE,
            "L": BLUE,
            "R": GREEN,
            "F": ORANGE,
            "B": RED
        }

        # Create a 3x3x3 Rubik's cube
        cube = VGroup()
        size = 3
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    cubelet = Cube(side_length=0.9)
                    cubelet.shift(x * RIGHT + y * UP + z * OUT)
                    cubelet.set_fill(face_colors["U"] if y == size - 1 else BLACK, opacity=1)
                    cubelet.set_fill(face_colors["D"] if y == 0 else BLACK, opacity=1)
                    cubelet.set_fill(face_colors["L"] if x == 0 else BLACK, opacity=1)
                    cubelet.set_fill(face_colors["R"] if x == size - 1 else BLACK, opacity=1)
                    cubelet.set_fill(face_colors["F"] if z == size - 1 else BLACK, opacity=1)
                    cubelet.set_fill(face_colors["B"] if z == 0 else BLACK, opacity=1)
                    cube.add(cubelet)

        # Add the cube to the scene
        self.add(cube)
        self.move_camera(phi=75 * DEGREES, theta=45 * DEGREES)

        # Scramble animation
        self.play(
            *[Rotate(cube, angle=random.uniform(PI / 6, PI / 3), axis=random.choice([X_AXIS, Y_AXIS, Z_AXIS])) for _ in
              range(10)])
        self.wait(1)

        # Solve animation (example)
        self.play(
            *[Rotate(cube, angle=-random.uniform(PI / 6, PI / 3), axis=random.choice([X_AXIS, Y_AXIS, Z_AXIS])) for _ in
              range(10)])
        self.wait(1)
