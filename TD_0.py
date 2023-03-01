import numpy as np


class TD(object):


    def __init__(self, env):
        self.env = env
        self.V = np.array(env.num_states)
        self.gamma = 0.95

    def estimate(self, pi, alpha, episode):
        s = self.env.reset()
        for i in range(episode):
            a = int(pi[s])
            s1, r1, done = self.env.step(a)
            self.V[s] = self.V[s] + alpha * (r1 + (self.gamma*self.V[s1] - self.V[s]))
            if done:
                s = self.env.reset()
            else:
                s = s1

    
    def train(self, pi, alpha, episodes):
        episode_len = 100
        for episode in range(episodes):
            self.estimate(pi, alpha, episode_len)

        return self.V
            





