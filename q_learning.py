import numpy as np

class Q_learning(object):
    

    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.gamma = 0.95

    def q_learn(self, alpha, threshold, episode):
        # init policy
        s = self.env.reset()
        for i in range(episode):
            for s in range(self.env.num_states):
                e = np.random.uniform()
                if e > threshold:
                    a = int(np.argmax(self.Q[s]))
                else:
                    a = np.random.randint(0, 4)

            s1, r1, done = self.env.step(a)
            self.Q[s][a] = self.Q[s][a] + alpha * (r1 + self.gamma * (max(self.Q[s1]) - self.Q[s][a]))
            if done:
                s = self.env.reset()
            else:
                s = s1


    def train(self, alpha, threshold, episodes):
        # train for multiple episodes
        episode_len = 100
        for episode in range(episodes):
            self.q_learn(alpha, threshold, episode_len)

        # get resulting policy
        pi = np.zeros(self.env.num_states)
        for s in range(self.env.num_states):
            pi[s] = np.argmax(self.Q[s])

        return pi


    def run(self):
        s = self.env.reset()
        done = False
        step = 0
        while not done:
            self.env.render()
            (s, r, done) = self.env.step(np.argmax(self.Q[s]))
            if done:
                s = self.env.reset()
            else:
                step += 1