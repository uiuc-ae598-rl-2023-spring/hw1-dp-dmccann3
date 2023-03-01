from gridworld import GridWorld
import numpy as np

class VI(object):


    def __init__(self):

        # grid of states 
        self.states = [
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24
        ]

        # actions (up, right, down, left)
        self.actions = [0, 1, 2, 3]

        # init state values and policy
        self.V = [0 for s in self.states]
        self.pi = [None for s in self.states]

        # discount factor
        self.gamma = 0.95

        # init env
        self.gridworld = GridWorld(False)


    def train(self):

        max_iter = 10000
        # Policy Evaluation
        for k in range(max_iter): 
                theta = 1e-20
                delta = theta + 1
                for s in self.states:
                    V = [0 for a in self.actions]
                    delta = 0
                    for a in self.actions:
                        v = self.V[s]
                        self.gridworld.s = s
                        s1, r1, _ = self.gridworld.step(a)                    
                        p = self.gridworld.p(s1, s, a)
                        V[a] += p * (r1 + self.gamma*self.V[s1])
                    self.V[s] = max(V)
                    delta = max(delta, abs(v - self.V[s]))

        return self.V
    
    def get_policy(self):
        V = [0 for a in self.actions]
        for s in self.states:
            for a in self.actions:
                self.gridworld.s = s
                s1, r1, _ = self.gridworld.step(a)
                p = self.gridworld.p(s1, s, a)
                V[a] += p * (r1 + self.gamma*self.V[s1])
            self.pi[s] = np.argmax(V)

        return self.pi
    
    def run(self):
        V = self.train()
        pi = self.get_policy()
        s = self.gridworld.reset()
        done = False
        step = 0
        while not done:
            self.gridworld.render()
            s, r, done = self.gridworld.step(pi[s])
            if done:
                self.gridworld.reset()
            else:
                step += 1


