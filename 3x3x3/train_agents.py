import os
import json
from collections import defaultdict
import random 
from sys import maxsize
from layoutInfo import SIDE

SIDE = 3
TILES = SIDE*SIDE
possible_moves = ['F', 'B', 'L', 'R', 'U', 'D', "U'"]
EPISODES, STEPS = 1000, 50
SCRAMBLES = 3       # can be adjusted; but shoots up complexity of HOW much the model needs to train (to reverse moves)

debug = False #debug flag


class TrainingCube:
    def __init__(self):
        self.state = {}
        self.reset()
        self.moves_made = []
    
    def reset(self):
        """
        Reset cube to solved state
        """
        # self.state = {
        #     'U': ['B'] * TILES,  # Up - BLUE
        #     'D': ['G'] * TILES,  # Down - GREEN
        #     'F': ['R'] * TILES,  # Front - RED
        #     'B': ['O'] * TILES,  # Back - ORANGE
        #     'L': ['W'] * TILES,  # Left - WHITE
        #     'R': ['Y'] * TILES   # Right - YELLOW
        # }
        self.state = {
            'U': ['W'] * TILES,  # Up - WHITE
            'D': ['Y'] * TILES,  # Down - YELLOW
            'F': ['R'] * TILES,  # Front - RED
            'B': ['O'] * TILES,  # Back - ORANGE
            'L': ['G'] * TILES,  # Left - GREEN
            'R': ['B'] * TILES   # Right - BLUE
        }
        self.moves_made = []
        if debug: print("\nCube reset to solved state: ")
        # self.visualize_cube()
        # return self.state

    def scramble(self):
        """
        Scramble the cube (perform random moves to shuffle the states)
        """
        scramble_sequence = ""
        for _ in range(SCRAMBLES):
            random_move = random.choice(possible_moves)
            self.apply_move(random_move)
            scramble_sequence += " "+random_move
        if debug: print(f"Scramble Sequence: {scramble_sequence}")
    
    def is_solved(self):
        return all(all(color == colors[0] for color in colors) for colors in self.state.values())
    
    def rotate_face_clock(self, face):
        """
        Rotate a face clockwise
        """
        old = self.state[face].copy()
        # self.state[face] = [old[2], old[0], old[3], old[1]]     # CHANGED
        self.state[face] = [old[6], old[3], old[0], old[7], old[4], old[1], old[8], old[5], old[2]]     # CHANGED

    def rotate_face_counter(self, face):
        """
        Rotate a face counter clockwise
        """
        old = self.state[face].copy()
        # self.state[face] = [old[2], old[0], old[3], old[1]]     # CHANGED
        self.state[face] = [old[2], old[5], old[8], old[1], old[4], old[7], old[0], old[3], old[6]]     # CHANGED
    
    def apply_move(self, move):
        """
        Apply a single move to the cube
        """
        # print(f"move performed= {move}")
        face = move[0]
        self.moves_made.append(move)
        self.rotate_face_clock(face) # CHANGED
        # self.rotate_face_clock(face) if move else self.rotate_face_counter(face)
        
        # adjust tiles for other faces (apart from the face the move is performed on)
        if face == 'U':  # Up move (left to right)
            temp = self.state['F'][:SIDE].copy()
            self.state['F'][:SIDE] = self.state['R'][:SIDE]     # FRONT -> RIGHT
            self.state['R'][:SIDE] = self.state['B'][:SIDE]     # RIGHT -> BACK
            self.state['B'][:SIDE] = self.state['L'][:SIDE]     # BACK -> LEFT
            self.state['L'][:SIDE] = temp                       # LEFT -> FRONT

        elif face == "U'":  # Up move (right to left)
            temp = self.state['F'][:SIDE].copy()
            self.state['F'][:SIDE] = self.state['L'][:SIDE]     # FRONT -> RIGHT
            self.state['L'][:SIDE] = self.state['B'][:SIDE]     # RIGHT -> BACK
            self.state['B'][:SIDE] = self.state['R'][:SIDE]     # BACK -> LEFT
            self.state['R'][:SIDE] = temp                       # LEFT -> FRONT
        
        elif face == 'D':  # Down move (left to right)
            temp = self.state['F'][6:].copy()
            self.state['F'][6:] = self.state['R'][6:]     # FRONT -> RIGHT
            self.state['R'][6:] = self.state['B'][6:]     # RIGHT -> BACK
            self.state['B'][6:] = self.state['L'][6:]     # BACK -> LEFT
            self.state['L'][6:] = temp                    # LEFT -> FRONT
            
        elif face == 'L':  # Left move (top to down)
            temp = [self.state['U'][0], self.state['U'][3], self.state['U'][6]]
            self.state['U'][0] = self.state['F'][0]     # FRONT -> UP
            self.state['U'][3] = self.state['F'][3]
            self.state['U'][6] = self.state['F'][6]
            self.state['F'][0] = self.state['D'][0]     # DOWN -> FRONT
            self.state['F'][3] = self.state['D'][3]
            self.state['F'][6] = self.state['D'][6]
            self.state['D'][0] = self.state['B'][0]     # BACK -> DOWN
            self.state['D'][3] = self.state['B'][3]
            self.state['D'][6] = self.state['B'][6]
            self.state['B'][0] = temp[0]                # TOP -> BACK
            self.state['B'][3] = temp[1]
            self.state['B'][6] = temp[2]
            
        elif face == 'R':  # Right move (top to down)
            temp = [self.state['U'][2], self.state['U'][5], self.state['U'][8]]
            self.state['U'][2] = self.state['F'][2]     # FRONT -> UP
            self.state['U'][5] = self.state['F'][5]
            self.state['U'][8] = self.state['F'][8]
            self.state['F'][2] = self.state['D'][2]     # DOWN -> FRONT
            self.state['F'][5] = self.state['D'][5]
            self.state['F'][8] = self.state['D'][8]
            self.state['D'][2] = self.state['B'][2]     # BACK -> DOWN
            self.state['D'][5] = self.state['B'][5]
            self.state['D'][8] = self.state['B'][8]
            self.state['B'][2] = temp[0]                # TOP -> BACK
            self.state['B'][5] = temp[1]
            self.state['B'][8] = temp[2]
            
        elif face == 'F':  # Front move (clockwise)
            temp = self.state['U'][6:].copy()
            self.state['U'][6] = self.state['L'][2]     # LEFT -> TOP
            self.state['U'][7] = self.state['L'][5]
            self.state['U'][8] = self.state['L'][8]
            self.state['L'][2] = self.state['D'][0]     # DOWN -> LEFT
            self.state['L'][5] = self.state['D'][1]
            self.state['L'][8] = self.state['D'][2]
            self.state['D'][0] = self.state['R'][0]     # RIGHT -> DOWN
            self.state['D'][1] = self.state['R'][3]
            self.state['D'][2] = self.state['R'][6]
            self.state['R'][0] = temp[0]                # TOP -> RIGHT
            self.state['R'][3] = temp[1]
            self.state['R'][6] = temp[2]
            
        elif face == 'B':  # Back move (clockwise)
            temp = self.state['U'][:SIDE].copy()
            self.state['U'][0] = self.state['L'][0]     # LEFT -> TOP
            self.state['U'][1] = self.state['L'][3]
            self.state['U'][2] = self.state['L'][6]
            self.state['L'][0] = self.state['D'][6]     # DOWN -> LEFT
            self.state['L'][3] = self.state['D'][7]
            self.state['L'][6] = self.state['D'][8]
            self.state['D'][6] = self.state['R'][2]     # RIGHT -> DOWN
            self.state['D'][7] = self.state['R'][5]
            self.state['D'][8] = self.state['R'][8]
            self.state['R'][2] = temp[0]                # TOP -> RIGHT
            self.state['R'][5] = temp[1]
            self.state['R'][8] = temp[2]

    def display(self):
        self.visualize_cube()

    def visualize_cube(self):
        state = self.state
        """
        Create a visual representation of the cube state
        """
        colors = {
            'W': '⬜', 'Y': '🟨',
            'G': '🟩', 'B': '🟦',
            'O': '🟧', 'R': '🟥'
        }
        
        layout = [
            "         {} {} {}     ",
            "         {} {} {}     ",
            "         {} {} {}     ",
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            "         {} {} {}     ",
            "         {} {} {}     ",
            "         {} {} {}     "
        ]
        
        face_values = {
            'U': state['U'],
            'L': state['L'],
            'F': state['F'],
            'R': state['R'],
            'B': state['B'],
            'D': state['D']
        }
        
        print("\n=== Current Cube State ===")
        print(layout[0].format(*[colors[tile] for tile in face_values['U'][0:3]]))
        print(layout[1].format(*[colors[tile] for tile in face_values['U'][3:6]]))
        print(layout[2].format(*[colors[tile] for tile in face_values['U'][6:9]]))
        print(layout[3].format(
            *[colors[tile] for tile in face_values['L'][0:3]],
            *[colors[tile] for tile in face_values['F'][0:3]],
            *[colors[tile] for tile in face_values['R'][0:3]],
            *[colors[tile] for tile in face_values['B'][0:3]]
        ))
        print(layout[4].format(
            *[colors[tile] for tile in face_values['L'][3:6]],
            *[colors[tile] for tile in face_values['F'][3:6]],
            *[colors[tile] for tile in face_values['R'][3:6]],
            *[colors[tile] for tile in face_values['B'][3:6]]
        ))
        print(layout[5].format(
            *[colors[tile] for tile in face_values['L'][6:9]],
            *[colors[tile] for tile in face_values['F'][6:9]],
            *[colors[tile] for tile in face_values['R'][6:9]],
            *[colors[tile] for tile in face_values['B'][6:9]]
        ))
        print(layout[6].format(*[colors[tile] for tile in face_values['D'][0:3]]))
        print(layout[7].format(*[colors[tile] for tile in face_values['D'][3:6]]))
        print(layout[8].format(*[colors[tile] for tile in face_values['D'][6:9]]))
        print("============================")


class QLearningAgent:
    def __init__(self, learning_rate=0.6, discount_factor=0.99, epsilon=0.7):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.name = "Q-Learning"
        self.moves_made = 0
        self.solves = 0
        
    def get_state_features(self, state):
        features = []
        # Face completion features with high weight
        for face, colors in state.items():
            matches = sum(1 for color in colors if color == colors[0])
            if matches == len(colors):  # Complete face
                features.append(1.0)
            else:
                features.append(matches / len(colors))
        
        # Adjacent matches feature
        for face, colors in state.items():
            adjacent_matches = 0
            pairs = [(0,1), (2,3), (0,2), (1,3)]
            for i, j in pairs:
                if colors[i] == colors[j]:
                    adjacent_matches += 1
            features.append(adjacent_matches / 4)
        
        return tuple(features)
    
    def choose_action(self, state, training=True):
        """
        Chose the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions (i.e. at the terminal state), no optimal action (None)
        """
        self.moves_made += 1                                                    # keep track of total moves
        if training and random.random() < self.epsilon:                                                                  
            return random.choice(possible_moves)                                # for probability epsilon, take random action
        
        features = self.get_state_features(state)
        return max(possible_moves, key=lambda move: self.q_table[features][move])    # otherwise, take best possible action
    
    def learn(self, state, action, reward, next_state):
        """
        Train the agent and populate Q table for reinforced learning.
        """
        current_features = self.get_state_features(state)
        next_features = self.get_state_features(next_state)
        
        next_max_q = max([self.q_table[next_features][move] for move in possible_moves])
        
        current_q = self.q_table[current_features][action]
        new_q = current_q + self.lr * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[current_features][action] = new_q
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class SARSAAgent:
    def __init__(self, learning_rate=0.6, discount_factor=0.99, epsilon=0.7):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.name = "SARSA"
        self.moves_made = 0
        self.solves = 0
    
    def get_state_features(self, state):
        features = []
        # Face completion features with high weight
        for face, colors in state.items():
            matches = sum(1 for color in colors if color == colors[0])
            if matches == len(colors):
                features.append(1.0)
            else:
                features.append(matches / len(colors))
        
        # Adjacent matches feature
        for face, colors in state.items():
            adjacent_matches = 0
            pairs = [(0,1), (2,3), (0,2), (1,3)]
            for i, j in pairs:
                if colors[i] == colors[j]:
                    adjacent_matches += 1
            features.append(adjacent_matches / 4)
        
        return tuple(features)
    
    def choose_action(self, state, training=True):
        self.moves_made += 1
        if training and random.random() < self.epsilon:
            return random.choice(possible_moves)
        
        features = self.get_state_features(state)
        return max(possible_moves, 
                  key=lambda a: self.q_table[features][a])
    
    def learn(self, state, action, reward, next_state, next_action):
        current_features = self.get_state_features(state)
        next_features = self.get_state_features(next_state)
        
        next_q = self.q_table[next_features][next_action]
        current_q = self.q_table[current_features][action]
        
        new_q = current_q + self.lr * (
            reward + self.gamma * next_q - current_q
        )
        self.q_table[current_features][action] = new_q
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def calculate_reward(state, next_state):
    def count_matches(s):
        total = 0
        for face, colors in s.items():
            matches = sum(1 for color in colors if color == colors[0])
            total += matches
        return total

    current_matches = count_matches(state)
    next_matches = count_matches(next_state)

    if cube.is_solved():
        return 100.0  # Solved
    if next_matches > current_matches:
        return 10.0  # Improvement
    if next_matches == current_matches:
        return -2.0  # Stagnation
    return -5.0  # Regression

def train_agents(episodes=1000, max_steps=25):
    q_agent = QLearningAgent()
    sarsa_agent = SARSAAgent()
    
    agents = {
        'Q-Learning': q_agent, 
        'SARSA': sarsa_agent
    }
    results = {
        name : {
            'solved'        : 0, 
            'best_solution' : maxsize, 
            'rewards'       : []
            }
        for name in agents
    }
    
    for agent_name, agent in agents.items():
        for episode in range(episodes):
            cube = TrainingCube()
            cube.reset()
            cube.scramble() 
            cube.display()
            
            total_reward = 0
            solution_sequence = []
            
            current_state = cube.state.copy()      # CHANGED
            action = agent.choose_action(current_state)
            if debug: print(f"this is action: {action}")
            
            for step in range(max_steps):
                solution_sequence.append(action)

                cube.apply_move(action)
                new_state = cube.state     #CHANGED
                reward = calculate_reward(current_state, new_state)
                total_reward += reward
                
                if agent_name == 'SARSA':
                    next_action = agent.choose_action(new_state)
                    agent.learn(current_state, action, reward, new_state, next_action)
                    action = next_action
                else:
                    agent.learn(current_state, action, reward, new_state)
                
                current_state = new_state
                if all(all(color == colors[0] for color in colors) for colors in new_state.values()):
                    results[agent_name]['solved'] += 1
                    results[agent_name]['best_solution'] = min(results[agent_name]['best_solution'], step + 1)
                    break
            
            results[agent_name]['rewards'].append(total_reward)
        # export q_table
        with open(f"./FINAL/Q-Table-{agent_name}.json", "w") as f:
            json.dump(remap_keys(agent.q_table), f)
    
    return results

# =========================================================
# TRAIN AND SOLVE

cube = TrainingCube()
cube.reset()

def train_agent(agent, episodes=1000, max_steps=25):
    for episode in range(episodes):
        if episode and not episode%1000 : print(f"Episode {episode}")
        cube.scramble()

        current_state = cube.state
        for step in range(max_steps):
            action = agent.choose_action(current_state)
            cube.apply_move(action)
            next_state = cube.state
            reward = calculate_reward(current_state, next_state)
            agent.learn(current_state, action, reward, next_state)
            current_state = next_state

            if cube.is_solved():
                agent.solves += 1
                break
    print(f"Training completed. Solves: {agent.solves}")

    # export q_table
    with open("./FINAL/QLearingAgent-Q-table.json", "w") as f:
        json.dump(remap_keys(agent.q_table), f)
        
    return agent

def train_SARSA_agent(agent, episodes=1000, max_steps=25):
    for episode in range(episodes):
        if episode and not episode%1000 : print(f"Episode {episode}")
        cube.scramble()

        current_state = cube.state
        for step in range(max_steps):
            action = agent.choose_action(current_state)
            cube.apply_move(action)
            next_state = cube.state
            reward = calculate_reward(current_state, next_state)
            next_action = agent.choose_action(next_state)
            agent.learn(current_state, action, reward, next_state, next_action)
            current_state = next_state

            if cube.is_solved():
                agent.solves += 1
                break
    print(f"Training completed. Solves: {agent.solves}")

    # export q_table
    with open("./FINAL/QLearingAgent-Q-table.json", "w") as f:
        json.dump(remap_keys(agent.q_table), f)
        
    return agent

def remap_keys(mapping):
    return [{'key':k, 'value': v} for k, v in mapping.items()]

def solve_cube_with_agent(agent):
    cube.reset()
    cube.scramble()
    print("Scrambled Cube:")
    cube.display()

    steps = 0
    while not cube.is_solved() and steps < 30:
        current_state = cube.state
        action = agent.choose_action(current_state, training=False)
        cube.apply_move(action)
        steps += 1
        print(f"Step {steps}: Move {action}")
        cube.display()
        if cube.is_solved():
            print(f"SOLVED in {steps} moves")
            break

    if cube.is_solved():
        print(f"Cube solved in {steps} steps!")
    else:
        print("Failed to solve the cube.")

# =========================================================

# TRAIN AGENTS

if __name__ == '__main__':
    # Initialize and train the Q-Learning agent
    q_agent = QLearningAgent()
    q_trained_agent = train_agent(q_agent, episodes=EPISODES, max_steps=STEPS)

    sarsa_agent = SARSAAgent()
    sarsa_trained_agent = train_SARSA_agent(sarsa_agent, episodes=EPISODES, max_steps=STEPS)

    # Use the trained agent to solve a random cube
    solve_cube_with_agent(q_trained_agent)
    solve_cube_with_agent(sarsa_trained_agent)