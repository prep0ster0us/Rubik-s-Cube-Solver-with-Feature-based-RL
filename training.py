from cube import RubiksCube
from q_learning import QLearningAgent
from sarsa import SARSAAgent

def train_agent(agent_class, cube_size=3, episodes=1000):
    cube = RubiksCube(size=cube_size)
    agent = agent_class(actions=cube.actions)

    for _ in range(episodes):
        cube.scramble()
        state = cube.cube_to_state()
        action = agent.choose_action(state)

        while not cube.is_solved():
            next_state, reward = cube.execute_action(action)
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

    return agent
