from training import train_agent
from q_learning import QLearningAgent
from sarsa import SARSAAgent

def evaluate_agents():
    q_agent = train_agent(QLearningAgent, cube_size=3, episodes=500)
    sarsa_agent = train_agent(SARSAAgent, cube_size=3, episodes=500)

    print("Q-Learning Performance:", q_agent.q_table)
    print("SARSA Performance:", sarsa_agent.q_table)
