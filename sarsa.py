import numpy as np

class SARSA(object):

    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.pi = np.zeros(env.num_states)
        self.gamma = 0.95

    def sarsa(self, alpha, threshold, episode):
        # init policy
        for s in range(self.env.num_states):
            e = np.random.uniform()
            if e > threshold:
                self.pi[s] = np.argmax(self.Q[s])
            else:
                self.pi[s] = np.random.randint(0, 4)

        # training loop
        s = self.env.reset()
        for i in range(episode):
            a = int(self.pi[s])
            s1, r1, done = self.env.step(a)
            a1 = int(self.pi[s1])
            self.Q[s][a] = self.Q[s][a] + alpha * (r1 + self.gamma * (self.Q[s1][a1] - self.Q[s][a]))
            if done:
                s = self.env.reset()
            else:
                s = s1



    def train(self, alpha, threshold, episodes):
        # train for multiple episodes
        episode_len = 100
        for episode in range(episodes):
            self.sarsa(alpha, threshold, episode_len)

        # get resulting policy
        for s in range(self.env.num_states):
            self.pi[s] = np.argmax(self.Q[s])

        return self.pi


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







            
