
# Rubik’s Cube Solver with Feature-Based Reinforcement Learning

This project implements a Rubik’s Cube Solver using feature-based Reinforcement Learning algorithms. We were successfully able to train reinforcement learning agents to solve a 2x2x2 Rubik’s Cube with working accuracy, provided the agents are trained for enough number of episodes and moves.

Once trained and having learnt the optimal move sets, the trained agents can take a scrambled cube’s configuration as input and provide a sequence of moves to solve the cube.

**Features**

- Reinforcement Learning Algorithms: Implements QLearning and SARSA for solving the Rubik’s Cube.
- 2x2x2 Cube Solver: Successfully trains agents to solve 2x2x2 configurations.
- Interactive Solver: Accepts user-inputted scrambled cube configurations and provides the optimal solution.
- Persisted Training Data: Saves and uses Q-table for quick and efficient solving.

**Project Scope**

Our project focuses on 2x2x2 cube in particular and we were able to achieve all results for this state space. But we did try our hand at expanding to 3x3x3 cubes, and create a visual 3d model for it. Please note that this is sitll under development as of today (December 11, 2024) so it may produce unexpected results or errors.

**_NOTE: While pushing the final changes onto the branch, we had to do a force push due to some issues with merge conflicts; which removed our previous commit history._**

## Prerequisites

Ensure the following dependencies are installed before running the project:

- Python (>=3.10)
- Required libraries: numpy, vpython, kociemba.

Install the dependencies using:

```
pip install -r requirements.txt
```

## How to Run

For 2x2x2 cube, simply execute the ```main.py``` file.

```
python3 main.py
```

### !! **EXPERIMENTAL** !!

### For 3x3x3 cube

> Training the Agents
  ```
  python3 train_agents.py
  ```

This will train 2 separate agents on each algorithm, QLearning and SARSA; and save the learned Q table data as JSON files.

> Visualizing 3d cube in action

```
python3 cube.py
```

This hosts the 3d model on a browser, with buttons to interact for playing individual moves, scrambling the cube and solving it using traditional as well as AI-learned moves.

## Video Demonstration

You can watch a video demonstration of the project [here](https://youtu.be/vRf8seRv1zA).


## Project Insights

### Inferences
- QLearning vs. SARSA: QLearning showed marginally better performance in terms of convergence speed and accuracy for solving 2x2x2 cubes.
- Scalability: While the framework is extendable to 3x3x3 cubes, training time and state space complexity grow significantly.

### Advantages
- Automated Solving: Provides optimal solutions for given cube configurations.
- Data Persistence: Stores training data for reuse, saving computation time.
- Reinforcement Learning Application: Demonstrates the power of RL in solving complex combinatorial problems.

### Disadvantages
- Limited Scalability: The state space for a 3x3x3 cube is exponentially larger than for a 2x2x2 cube, making it computationally intensive.
- Initial Training Time: Training the agent from scratch can be time-consuming.

## Troubleshooting

### Common Issues with 3x3x3 cube model
- **Missing Dependencies:** Make sure thet python version matches the minimum requirements and all the required libraries have been installed.
- **Q-Table Missing:** Ensure the correct Q-table file is available in the specified directory.
- **Scrambled State Input:** Ensure the scrambled state matches the cube’s configuration format.

### Support

If you encounter any issues, feel free to reach out:
> **Email:** ritwik.babu2@gmail.com
>
> **GitHub:** github.com/prep0ster0us

## Future Work
- Work through integrating 3x3x3 cubes and connecting visual model to 2x2x2 cube.
- Optimize state representation to reduce memory usage.
- Explore additional RL algorithms for better convergence.

## License

This project was developed as part of Introduction to Artificial Intelligence course @ University of New Haven in Fall 2024. Feel free to use, modify, and share!

