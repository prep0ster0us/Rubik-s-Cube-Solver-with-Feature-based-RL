import random
from collections import defaultdict
import time

def visualize_cube_state(state):
    """Create a visual representation of the cube state"""
    color_map = {
        'W': '⬜', 'Y': '🟨',
        'G': '🟩', 'B': '🟦',
        'O': '🟧', 'R': '🟥'
    }
    
    layout = [
        "      {} {}      ",
        "      {} {}      ",
        "{} {} {} {} {} {} {} {}",
        "{} {} {} {} {} {} {} {}",
        "      {} {}      ",
        "      {} {}      "
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
    print(layout[0].format(color_map[face_values['U'][0]], color_map[face_values['U'][1]]))
    print(layout[1].format(color_map[face_values['U'][2]], color_map[face_values['U'][3]]))
    print(layout[2].format(
        color_map[face_values['L'][0]], color_map[face_values['L'][1]],
        color_map[face_values['F'][0]], color_map[face_values['F'][1]],
        color_map[face_values['R'][0]], color_map[face_values['R'][1]],
        color_map[face_values['B'][0]], color_map[face_values['B'][1]]
    ))
    print(layout[3].format(
        color_map[face_values['L'][2]], color_map[face_values['L'][3]],
        color_map[face_values['F'][2]], color_map[face_values['F'][3]],
        color_map[face_values['R'][2]], color_map[face_values['R'][3]],
        color_map[face_values['B'][2]], color_map[face_values['B'][3]]
    ))
    print(layout[4].format(color_map[face_values['D'][0]], color_map[face_values['D'][1]]))
    print(layout[5].format(color_map[face_values['D'][2]], color_map[face_values['D'][3]]))
    print("=====================")

class RubiksCube:
    def __init__(self):
        self.reset()
        self.move_history = []
    
    def reset(self):
        """Initialize solved state"""
        self.state = {
            'U': ['W'] * 4,  # Up - White
            'D': ['Y'] * 4,  # Down - Yellow
            'F': ['G'] * 4,  # Front - Green
            'B': ['B'] * 4,  # Back - Blue
            'L': ['O'] * 4,  # Left - Orange
            'R': ['R'] * 4   # Right - Red
        }
        self.move_history = []
        print("\nCube reset to solved state:")
        self.display()
        return self.state
    
    def display(self):
        visualize_cube_state(self.state)
    
    def rotate_face(self, face):
        """Rotate a face clockwise"""
        old = self.state[face].copy()
        self.state[face] = [old[2], old[0], old[3], old[1]]
    
    def apply_move(self, move):
        """Apply a single move to the cube"""
        face = move[0]
        self.move_history.append(move)
        self.rotate_face(face)
        
        if face == 'U':  # Up move
            temp = self.state['F'][:2].copy()
            self.state['F'][:2] = self.state['R'][:2]
            self.state['R'][:2] = self.state['B'][:2]
            self.state['B'][:2] = self.state['L'][:2]
            self.state['L'][:2] = temp
            
        elif face == 'D':  # Down move
            temp = self.state['F'][2:].copy()
            self.state['F'][2:] = self.state['L'][2:]
            self.state['L'][2:] = self.state['B'][2:]
            self.state['B'][2:] = self.state['R'][2:]
            self.state['R'][2:] = temp
            
        elif face == 'L':  # Left move
            temp = [self.state['U'][0], self.state['U'][2]]
            self.state['U'][0] = self.state['B'][2]
            self.state['U'][2] = self.state['B'][0]
            self.state['B'][2] = self.state['D'][0]
            self.state['B'][0] = self.state['D'][2]
            self.state['D'][0] = self.state['F'][0]
            self.state['D'][2] = self.state['F'][2]
            self.state['F'][0] = temp[0]
            self.state['F'][2] = temp[1]
            
        elif face == 'R':  # Right move
            temp = [self.state['U'][1], self.state['U'][3]]
            self.state['U'][1] = self.state['F'][1]
            self.state['U'][3] = self.state['F'][3]
            self.state['F'][1] = self.state['D'][1]
            self.state['F'][3] = self.state['D'][3]
            self.state['D'][1] = self.state['B'][3]
            self.state['D'][3] = self.state['B'][1]
            self.state['B'][3] = temp[0]
            self.state['B'][1] = temp[1]
            
        elif face == 'F':  # Front move
            temp = [self.state['U'][2], self.state['U'][3]]
            self.state['U'][2] = self.state['L'][3]
            self.state['U'][3] = self.state['L'][1]
            self.state['L'][3] = self.state['D'][0]
            self.state['L'][1] = self.state['D'][1]
            self.state['D'][0] = self.state['R'][0]
            self.state['D'][1] = self.state['R'][2]
            self.state['R'][0] = temp[0]
            self.state['R'][2] = temp[1]
            
        elif face == 'B':  # Back move
            temp = [self.state['U'][0], self.state['U'][1]]
            self.state['U'][0] = self.state['R'][1]
            self.state['U'][1] = self.state['R'][3]
            self.state['R'][1] = self.state['D'][2]
            self.state['R'][3] = self.state['D'][3]
            self.state['D'][2] = self.state['L'][2]
            self.state['D'][3] = self.state['L'][0]
            self.state['L'][2] = temp[0]
            self.state['L'][0] = temp[1]

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
    
    def choose_action(self, state, possible_moves):
        self.moves_made += 1
        if random.random() < self.epsilon:
            return random.choice(possible_moves)
        
        features = self.get_state_features(state)
        return max(possible_moves, 
                  key=lambda a: self.q_table[features][a])
    
    def learn(self, state, action, reward, next_state, possible_moves):
        current_features = self.get_state_features(state)
        next_features = self.get_state_features(next_state)
        
        next_max_q = max([self.q_table[next_features][a] 
                         for a in possible_moves])
        
        current_q = self.q_table[current_features][action]
        new_q = current_q + self.lr * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[current_features][action] = new_q
        
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)

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
    
    def choose_action(self, state, possible_moves):
        self.moves_made += 1
        if random.random() < self.epsilon:
            return random.choice(possible_moves)
        
        features = self.get_state_features(state)
        return max(possible_moves, 
                  key=lambda a: self.q_table[features][a])
    
    def learn(self, state, action, reward, next_state, next_action, possible_moves):
        current_features = self.get_state_features(state)
        next_features = self.get_state_features(next_state)
        
        next_q = self.q_table[next_features][next_action]
        current_q = self.q_table[current_features][action]
        
        new_q = current_q + self.lr * (
            reward + self.gamma * next_q - current_q
        )
        self.q_table[current_features][action] = new_q
        
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)

def calculate_reward(state, next_state):
    """Enhanced reward function"""
    def count_matches(s):
        total = 0
        for face, colors in s.items():
            matches = sum(1 for color in colors if color == colors[0])
            if matches == len(colors):  # Complete face
                total += matches * 8
            else:
                total += matches * 2
            
            pairs = [(0,1), (2,3), (0,2), (1,3)]
            for i, j in pairs:
                if colors[i] == colors[j]:
                    total += 3
        return total
    
    if all(all(c == colors[0] for c in colors) for colors in next_state.values()):
        return 1000.0
    
    current_matches = count_matches(state)
    next_matches = count_matches(next_state)
    
    if next_matches > current_matches:
        return 100.0
    elif next_matches == current_matches:
        return -0.1
    else:
        return -5.0

def print_progress(agent, episode, total_episodes):
    """Print detailed progress information"""
    current_success_rate = (agent.solves / (episode + 1)) * 100
    print(f"\n{agent.name} Progress:")
    print(f"Current success rate: {current_success_rate:.1f}%")
    print(f"Moves this episode: {agent.moves_made / (episode + 1):.1f}")
    print(f"Exploration rate: {agent.epsilon:.3f}")
    if agent.solves > 0:
        print(f"Last solve: Episode {episode + 1}")

def train_both_agents():
    print("Starting Enhanced Rubik's Cube Training")
    
    cube_q = RubiksCube()
    cube_sarsa = RubiksCube()
    q_agent = QLearningAgent()
    sarsa_agent = SARSAAgent()
    possible_moves = ['U', 'D', 'L', 'R', 'F', 'B']
    
    # episodes = 15
    # max_steps = 12
    # TODO: test parameters
    episodes = 5000     # ** RITWIK - change here for number of episodes to run for
    max_steps = 8       # ** RITWIK - can opt to change this, but I'd recommend keeping this 8 (since this is the upper threshold for the number of moves to solve a 2x2x2 cube)

    initial_scramble = 1
    max_scramble = 2
    
    print(f"\nTraining Configuration:")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Initial/Max scramble: {initial_scramble}/{max_scramble}")
    print(f"Learning rate: {q_agent.lr}")
    print(f"Initial exploration rate: {q_agent.epsilon}")
    print(f"Discount factor: {q_agent.gamma}\n")
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}:")
        
        # Progressive difficulty
        if episode < 20:
            scramble_length = initial_scramble
        else:
            scramble_length = min(initial_scramble + ((episode-20) // 10), max_scramble)
            
        scramble_moves = [random.choice(possible_moves) for _ in range(scramble_length)]
        
        for agent, cube in [(q_agent, cube_q), (sarsa_agent, cube_sarsa)]:
            print(f"\nTraining {agent.name}:")
            start_time = time.time()
            
            cube.reset()
            print("Scramble sequence:", ' '.join(scramble_moves))
            for move in scramble_moves:
                cube.apply_move(move)
            cube.display()
            
            current_state = {f: cube.state[f][:] for f in cube.state}
            moves_taken = []
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.choose_action(current_state, possible_moves)
                moves_taken.append(action)
                
                print(f"\nStep {step + 1}: Applying {action}")
                cube.apply_move(action)
                new_state = {f: cube.state[f][:] for f in cube.state}
                
                reward = calculate_reward(current_state, new_state)
                total_reward += reward
                
                if isinstance(agent, SARSAAgent):
                    next_action = agent.choose_action(new_state, possible_moves)
                    agent.learn(current_state, action, reward, new_state, 
                              next_action, possible_moves)
                else:
                    agent.learn(current_state, action, reward, new_state, 
                              possible_moves)
                
                cube.display()
                # time.sleep(0.3)  # Slightly faster visualization      # ** RITWIK - commented out for faster execution
                
                if all(all(c == colors[0] for c in colors) 
                       for colors in new_state.values()):
                    agent.solves += 1
                    print(f"\n🎉 {agent.name} solved it in {step + 1} moves!")
                    print(f"Solution: {' '.join(moves_taken)}")
                    break
                
                current_state = new_state
            
            print(f"\n{agent.name} Episode Summary:")
            print(f"Time: {time.time() - start_time:.2f} seconds")
            print(f"Total reward: {total_reward:.1f}")
            print_progress(agent, episode, episodes)
    
    print("\n=== Final Results ===")
    for agent in [q_agent, sarsa_agent]:
        success_rate = (agent.solves / episodes) * 100
        moves_per_episode = agent.moves_made / episodes
        print(f"\n{agent.name}:")
        print(f"Solved {agent.solves} out of {episodes} episodes")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average moves per episode: {moves_per_episode:.1f}")
        print(f"Final exploration rate: {agent.epsilon:.3f}")


def auto_select_and_solve():
    print("\n=== Rubik's Cube Solver ===")
    print("\nAvailable Moves:")
    print("U - Up Face Clockwise")
    print("D - Down Face Clockwise")
    print("L - Left Face Clockwise")
    print("R - Right Face Clockwise")
    print("F - Front Face Clockwise")
    print("B - Back Face Clockwise")
    
    print("\nRecommended sequences to try:")
    print("1. 'R U'   (Right then Up)")
    print("2. 'F R'   (Front then Right)")
    print("3. 'L R'   (Left then Right)")
    print("4. 'U D'   (Up then Down)")
    
    print("\nStarting training of both methods...")
    
    # Initialize both agents
    cube_q = RubiksCube()
    cube_sarsa = RubiksCube()
    q_agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=1.0)
    sarsa_agent = SARSAAgent(learning_rate=0.1, discount_factor=0.95, epsilon=1.0)
    possible_moves = ['U', 'D', 'L', 'R', 'F', 'B']
    
    # Training parameters
    episodes = 5000
    max_steps = 6
    
    # Training metrics
    q_stats = {'solved': 0, 'moves': [], 'total_reward': 0}
    sarsa_stats = {'solved': 0, 'moves': [], 'total_reward': 0}
    
    # Common patterns
    common_patterns = [
        ['R', 'U'], ['U', 'R'], ['F', 'R'], ['R', 'F'],
        ['L', 'R'], ['R', 'L'], ['U', 'D'], ['D', 'U']
    ]
    
    # Training phase
    print("Training Progress:")
    for episode in range(episodes):
        if episode % 500 == 0:
            print(f"Episode {episode + 1}/{episodes}")
        
        if episode % 3 == 0 and episode < 3000:
            scramble_moves = random.choice(common_patterns)
        else:
            scramble_length = 1 if episode < 2000 else (2 if episode < 4000 else 3)
            scramble_moves = [random.choice(possible_moves) for _ in range(scramble_length)]
        
        for agent, cube, stats in [(q_agent, cube_q, q_stats), (sarsa_agent, cube_sarsa, sarsa_stats)]:
            cube.reset()
            moves_this_episode = 0
            episode_reward = 0
            
            for move in scramble_moves:
                cube.apply_move(move)
            
            current_state = {f: cube.state[f][:] for f in cube.state}
            
            for step in range(max_steps):
                moves_this_episode += 1
                action = agent.choose_action(current_state, possible_moves)
                cube.apply_move(action)
                new_state = {f: cube.state[f][:] for f in cube.state}
                reward = calculate_reward(current_state, new_state) * (1.5 if step < 3 else 1.0)
                episode_reward += reward
                
                if isinstance(agent, SARSAAgent):
                    next_action = agent.choose_action(new_state, possible_moves)
                    agent.learn(current_state, action, reward, new_state, next_action, possible_moves)
                else:
                    agent.learn(current_state, action, reward, new_state, possible_moves)
                
                if all(all(c == colors[0] for c in colors) for colors in new_state.values()):
                    stats['solved'] += 1
                    stats['moves'].append(moves_this_episode)
                    break
                
                current_state = new_state
            stats['total_reward'] += episode_reward
    
    # Calculate and display final statistics
    print("\n=== Training Results ===")
    
    # Q-Learning Results
    q_success_rate = (q_stats['solved'] / episodes) * 100
    q_avg_moves = sum(q_stats['moves']) / len(q_stats['moves']) if q_stats['moves'] else 0
    q_avg_reward = q_stats['total_reward'] / episodes
    
    print("\nQ-Learning Performance:")
    print(f"Success Rate: {q_success_rate:.1f}%")
    print(f"Average Moves When Solved: {q_avg_moves:.1f}")
    print(f"Total Solves: {q_stats['solved']}/{episodes}")
    print(f"Average Reward: {q_avg_reward:.1f}")
    
    # SARSA Results
    sarsa_success_rate = (sarsa_stats['solved'] / episodes) * 100
    sarsa_avg_moves = sum(sarsa_stats['moves']) / len(sarsa_stats['moves']) if sarsa_stats['moves'] else 0
    sarsa_avg_reward = sarsa_stats['total_reward'] / episodes
    
    print("\nSARSA Performance:")
    print(f"Success Rate: {sarsa_success_rate:.1f}%")
    print(f"Average Moves When Solved: {sarsa_avg_moves:.1f}")
    print(f"Total Solves: {sarsa_stats['solved']}/{episodes}")
    print(f"Average Reward: {sarsa_avg_reward:.1f}")
    
    # Select best method
    if q_success_rate > sarsa_success_rate:
        best_agent = q_agent
        best_cube = cube_q
        print("\nQ-Learning selected as best method!")
        print(f"Better success rate: {q_success_rate:.1f}% vs {sarsa_success_rate:.1f}%")
    elif sarsa_success_rate > q_success_rate:
        best_agent = sarsa_agent
        best_cube = cube_sarsa
        print("\nSARSA selected as best method!")
        print(f"Better success rate: {sarsa_success_rate:.1f}% vs {q_success_rate:.1f}%")
    else:
        if q_avg_moves <= sarsa_avg_moves:
            best_agent = q_agent
            best_cube = cube_q
            print("\nQ-Learning selected as best method!")
            print("Equal success rates but better average moves")
        else:
            best_agent = sarsa_agent
            best_cube = cube_sarsa
            print("\nSARSA selected as best method!")
            print("Equal success rates but better average moves")
    
    # Solving loop with best method
    while True:
        print("\nEnter scramble sequence (or 'quit' to exit, 'examples' for move list): ")
        user_input = input().strip().lower()
        
        if user_input == 'quit':
            break
            
        if user_input == 'examples':
            print("\nRecommended sequences:")
            print("R U   (Right then Up)")
            print("F R   (Front then Right)")
            print("L R   (Left then Right)")
            print("U D   (Up then Down)")
            continue
        
        scramble = user_input.upper().split()
        if not all(move in possible_moves for move in scramble):
            print("Invalid moves! Use only: U, D, L, R, F, B")
            continue
        
        print("Finding solution...")
        best_solution = None
        best_length = float('inf')
        
        for attempt in range(3):
            best_cube.reset()
            for move in scramble:
                best_cube.apply_move(move)
            
            current_state = {f: best_cube.state[f][:] for f in best_cube.state}
            moves_taken = []
            best_agent.epsilon = 0.1 + (attempt * 0.2)
            
            for step in range(8):
                action = best_agent.choose_action(current_state, possible_moves)
                moves_taken.append(action)
                best_cube.apply_move(action)
                new_state = {f: best_cube.state[f][:] for f in best_cube.state}
                
                if all(all(c == colors[0] for c in colors) for colors in new_state.values()):
                    if len(moves_taken) < best_length:
                        best_solution = moves_taken.copy()
                        best_length = len(moves_taken)
                    break
                
                current_state = new_state
        
        if best_solution:
            print(f"\n🎉 Solution found: {' '.join(best_solution)}")
        else:
            print("\nCouldn't find solution. Try a simpler sequence.")

if __name__ == "__main__":
    auto_select_and_solve()