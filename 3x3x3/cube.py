import json
from solve_cube import *
from layoutInfo import *
from train_agents import QLearningAgent

SCRAMBLE_MOVE_LIMIT = 15
# DEBUG FLAG
debug = False

class Rubik_Cube():
    def __init__(self):
        self.running = True
        self.tiles = []
        self.dA = pi/40  # MODIFIED: Rotation step size, balances smoothness and speed

        # center cell
        sphere(pos= cubeOrigin, size= cubeSize, color= cubeOrigin)

        # define position for each cell
        tile_pos = [
            [   vector(-1, 1, 1.5),vector(0, 1, 1.5),vector(1, 1, 1.5),             # FRONT
                vector(-1, 0, 1.5),vector(0, 0, 1.5),vector(1, 0, 1.5),
                vector(-1, -1, 1.5),vector(0, -1, 1.5),vector(1, -1, 1.5)
            ],
            [   vector(1.5, 1, -1), vector(1.5, 1, 0), vector(1.5, 1, 1),           # RIGHT
                vector(1.5, 0, -1), vector(1.5, 0, 0), vector(1.5, 0, 1),
                vector(1.5, -1, -1), vector(1.5, -1, 0), vector(1.5, -1, 1)
            ],
            [   vector(-1, 1, -1.5), vector(0, 1, -1.5), vector(1, 1, -1.5),        # BACK
                vector(-1, 0, -1.5), vector(0, 0, -1.5), vector(1, 0, -1.5),
                vector(-1, -1, -1.5), vector(0, -1, -1.5), vector(1, -1, -1.5)
            ],
            [   vector(-1.5, 1, -1), vector(-1.5, 1, 0), vector(-1.5, 1, 1),        # LEFT
                vector(-1.5, 0, -1), vector(-1.5, 0, 0), vector(-1.5, 0, 1),
                vector(-1.5, -1, -1), vector(-1.5, -1, 0), vector(-1.5, -1, 1)
            ],
            [   vector(-1, 1.5, -1), vector(0, 1.5, -1), vector(1, 1.5, -1),        # TOP
                vector(-1, 1.5, 0), vector(0, 1.5, 0), vector(1, 1.5, 0),
                vector(-1, 1.5, 1), vector(0, 1.5, 1), vector(1, 1.5, 1)
            ],
            [   vector(-1, -1.5, -1), vector(0, -1.5, -1), vector(1, -1.5, -1),     # BOTTOM
                vector(-1, -1.5, 0), vector(0, -1.5, 0), vector(1, -1.5, 0),
                vector(-1, -1.5, 1), vector(0, -1.5, 1), vector(1, -1.5, 1)
            ]
        ]
        
        # faces of the cube
        colors = [
            vector(1,0,0),      # FRONT
            vector(1,1,0),      # RIGHT
            vector(1,0.5,0),    # BACK
            vector(1,1,1),      # LEFT
            vector(0,0,1),      # TOP
            vector(0,1,0)       # BOTTOM
        ]

        angle = [
            (0   , vector(0,0,0)),
            (pi/2, vector(0,1,0)),
            (0   , vector(0,0,0)),
            (pi/2, vector(0,1,0)),
            (pi/2, vector(1,0,0)),
            (pi/2, vector(1,0,0))
        ]

        # sides
        for rank, side in enumerate(tile_pos):
            for vec in side:
                # for each cell; draw a box/block
                tile = box(pos= vec,
                           size= vector(0.98,0.98,0.1),
                           color= colors[rank]
                           )
                tile.rotate(angle= angle[rank][0],
                            axis=angle[rank][1]
                            )
                self.tiles.append(tile)

        # positions
        self.positions = dict.fromkeys(directions, [])

        # variables
        self.rotate = [None,0,0]
        self.moves = []

    def resetPositions(self):
        """
        Reset positions for all tiles and calculate the face each tile belongs to.
        """
        # reset positions
        self.positions = { direction: [] for direction in directions }
        # for direction in directions:
        #     self.positions[direction] = []

        # calculate which face the tile belongs to (post-rotation)
        for tile in self.tiles:
            if tile.pos.z > 0.4:
                self.positions[FRONT].append(tile)
            if tile.pos.z < -0.4:
                self.positions[BACK].append(tile)
            if tile.pos.y > 0.4:
                self.positions[TOP].append(tile)
            if tile.pos.y < -0.4:
                self.positions[BOTTOM].append(tile)
            if tile.pos.x < -0.4:
                self.positions[LEFT].append(tile)
            if tile.pos.x > 0.4:
                self.positions[RIGHT].append(tile)

        for key in self.positions.keys():
            self.positions[key] = set(self.positions[key])      # convert to set

    def animations(self):
        """
        Animate a rotation action on the cube.
        """
        if(debug): print(f'animating {self.rotate}')
        if(debug): print("axis check: ", faceToRotate)

        # identify the face (relative to which) to be rotated
        faceToRotate = self.rotate[0]           
        # map the move (to be performed) to a specific face of the cube
        currentFace = ""                        
        match faceToRotate:
            case "front_clock"| "front_counter":
                currentFace = FRONT
            case "back_clock" | "back_counter":
                currentFace = BACK
            case "top_clock"   | "top_counter":
                currentFace = TOP
            case "bottom_clock" | "bottom_counter":
                currentFace = BOTTOM
            case "left_clock" | "left_counter":
                currentFace = LEFT
            case "right_clock"| "right_counter":
                currentFace = RIGHT
            case None:
                currentFace = ""

        if currentFace:
            # adjust cube for each tile (cell) on the face (to be rotated)
            for tile in self.positions[currentFace]:
                if(debug): print(f"tile before rotation- {tile.pos}")
                if(debug): print("rotate: ", currentFace, " : ", faceToRotate) 
                # rotate each tile of the face
                tile.rotate(angle= (self.dA if "counter" in faceToRotate else -self.dA), 
                            axis= cubeAxis[faceToRotate],
                            origin= cubeOrigin )
                if(debug): print(f"tile after rotation- {tile.pos}")  # MODIFIED: Confirm rotation
            # track number of rotations applied
            self.rotate[1] += self.dA

        # check if current move's rotation is complete 
        # (match current rotation angle with target angle- pi/2)
        if (self.rotate[1] + self.dA/2 > self.rotate[2]) and (self.rotate[1] - self.dA/2 < self.rotate[2]):
            # reset rotation state (for next move)
            self.rotate = [None,0,0]
            self.resetPositions()  # recalculate tile positions and which face each tile belongs to (relative to post-rotation current state)
    

    ### METHODS TO SET ROTATION EVENT FOR EACH FACE OF THE CUBE ###
    def rotate_front_clock(self):
        """
        Rotate the front face (RED) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['front_clock',0,pi/2]
    def rotate_front_counter(self):
        """
        Rotate the front face (RED) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['front_counter',0,pi/2]

    def rotate_right_clock(self):
        """
        Rotate the right face (YELLOW) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['right_clock',0,pi/2]
    def rotate_right_counter(self):
        """
        Rotate the right face (YELLOW) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['right_counter',0,pi/2]

    def rotate_back_clock(self):
        """
        Rotate the back face (ORANGE) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['back_clock',0,pi/2]
    def rotate_back_counter(self):
        """
        Rotate the back face (ORANGE) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['back_counter',0,pi/2]

    def rotate_left_clock(self):
        """
        Rotate the left face (WHITE) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['left_clock',0,pi/2]
    def rotate_left_counter(self):
        """
        Rotate the left face (WHITE) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['left_counter',0,pi/2]
    
    def rotate_top_clock(self):
        """
        Rotate the top face (BLUE) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['top_clock',0,pi/2]
    def rotate_top_counter(self):
        """
        Rotate the top face (BLUE) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['top_counter',0,pi/2]

    def rotate_bottom_clock(self):
        """
        Rotate the bottom face (GREEN) clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['bottom_clock',0,pi/2]
    def rotate_bottom_counter(self):
        """
        Rotate the bottom face (GREEN) counter clockwise
        """
        if self.rotate[0] == None:
            self.rotate = ['bottom_counter',0,pi/2]
    ### ------------------------------------------- ###


    def move(self):
        """
        Perform a move on the cube (based on the sequence of moves).
        """
        if self.rotate[0] == None and len(self.moves) > 0:
            getattr(self, f"rotate_{self.moves[0]}")()

            self.moves.pop(0)

    def scramble(self):
        """
        Scramble the cube by (pseudo) random number of moves.
        """
        for _ in range(SCRAMBLE_MOVE_LIMIT):
            self.moves.append(random.choice(list(possibleMoves.values())))

    def solution(self):
        solve(self.tiles)

    def solveCube(self):
        """
        Perform solution on the cube.
        NOTE: kociemba provides the solution with 3 possible kinds of moves
            1. single clockwise move (eg: F)
            2. single counter clockwise move (eg: F')
            3. double rotation move (eg: F2)    (no distinction for direction, since double rotation would end up in the same state for both directions)
        """
        # get solution sequence
        values = solve(self.tiles)
        values = list(values.split(" "))
        for move in values:
            # deal with double rotation moves
            sequence = list(move)
            if sequence[-1] == '2':
                sequence.pop(-1)
                move = "".join(sequence)
                self.moves.append(mapMoves[move])
                self.moves.append(mapMoves[move])
                print(mapMoves[move])
                print(mapMoves[move])
            else:
                print(mapMoves[move])
                self.moves.append(mapMoves[move])
    
    def solveWithQLearning(self):
        """
        Perform Q-learning solution on the cube (based on training of the QLearning Agent)
        """
        # get solution sequence
        values = solve(self.tiles)
        values = list(values.split(" "))
        for move in values:
            # deal with double rotation moves
            sequence = list(move)
            if sequence[-1] == '2':
                sequence.pop(-1)
                move = "".join(sequence)
                self.moves.append(mapMoves[move])
                self.moves.append(mapMoves[move])
                print(mapMoves[move])
                print(mapMoves[move])
            else:
                print(mapMoves[move])
                self.moves.append(mapMoves[move])

    def control(self):
        wtext(text= "<br>", align="left")
        for text, move in possibleMoves.items():
            if debug: print(f"name= {text} | move= {move}")
            button(bind=getattr(self, f"rotate_{move}"), text= text)
            wtext(text= "  ", align="center")

        wtext(text= "<br><br>", align="left")
        button(bind= self.scramble, text='Scramble')

        # MODIFIED: Corrected button binding above to ensure callbacks are only triggered on click
        # TODO: implement reinforcement learning
        wtext(text= "<br><br>", align="right")
        # button(bind= self.solution, text='solution')
        button(bind= self.solveCube, text='QLearning Agent')    # testing | TODO: flatten states to apply moves for agent
        wtext(text="    ", align="left")
        button(bind= self.solveCube, text='SARSA Agent')        # testing | TODO: flatten states to apply moves for agent
        wtext(text="    ", align="left")
        button(bind= self.solveCube, text='SOLUTION')
        wtext(text= "<br><br><br>", align="right")

    def update(self):
        rate(60)
        self.animations()
        self.move()

    def start(self):
        self.resetPositions()  # MODIFIED: Ensure tile positions are recalculated after each move
        self.control()
        while self.running:
            self.update()

    def reset(self):
        self.resetPositions()

    def qLearningSolution(self):
        with open('./FINAL/QLearingAgent-Q-table.json') as json_file:
            data = json.load(json_file)
            trained_q_table = self.remap_keys(data)

            agent = QLearningAgent()
            # use the trained Q-table for this agent
            agent.q_table = trained_q_table
            steps = 0
            while not self.is_solved(): # and steps < 30:
                current_state = self.state  # TODO: flatten 3d model states as vectors
                # choose action
                action = agent.choose_action(current_state, training=False)
                steps += 1
                print(f"Step {steps}: Move {action}")
                cube.display()
                if cube.is_solved():
                    print(f"SOLVED in {steps} moves")
                    break

            if self.is_solved():
                print(f"Cube solved in {steps} steps!")
            else:
                print("Failed to solve the cube.")

    def remap_keys(self, data):
        return {tuple(pair["key"]): pair["value"] for pair in data}

    def apply_move(self, move):
        """
        Perform the given move on the cube.
        """
        getattr(self, f"rotate_{move}_clock")
        
    def get_state(self):
        """ Serialize the cube's state as a dictionary for the Q-learning agent. """
        return {face: list(self.cube[face].flatten()) for face in self.cube}

    def is_solved(self):
        return all(all(cell == face[0][0] for cell in face.flatten()) for face in self.cube.values())
    
# debug
cube = Rubik_Cube()
cube.start()