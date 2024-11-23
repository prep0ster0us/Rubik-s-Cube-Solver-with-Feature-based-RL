class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state, next_action):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * self.q_table.get(next_state, {}).get(next_action, 0)
        self.q_table[state][action] += self.alpha * (q_target - q_predict)
