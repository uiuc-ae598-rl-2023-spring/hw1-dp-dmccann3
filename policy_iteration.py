from gridworld import GridWorld
import numpy as np

class PI(object):

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
        self.pi = [0 for s in self.states]

        # discount factor
        self.gamma = 0.95

        # init env
        self.gridworld = GridWorld(False)

    # need to get transition function and rewards into this function here
    def evaluate(self):
        max_iter = 10000
        # Policy Evaluation
        for k in range(max_iter): # FIXME Most likely need to change this to reflect inifite time horizon
            theta = 5 # FIXME will need to change this most likely
            delta = theta + 1 # init delta greater than theta here to allow while loop to run
            while delta > theta:
                delta = 0
                for s in self.states:
                    v = self.V[s]
                    self.gridworld.s = s
                    s1, r1, _ = self.gridworld.step(self.pi[s])                    
                    p = self.gridworld.p(s1, s, self.pi[s])
                    self.V[s] += p * (r1 + self.gamma*self.V[s1])
                    delta = max(delta, abs(v - self.V[s]))

        return self.V
    
    def improve(self):
        V = [0 for a in self.actions]
        # Policy Improvement
        for s in self.states:
            self.gridworld.s = s
            for a in self.actions:
                s1, r1, _ = self.gridworld.step(a)
                p = self.gridworld.p(s1, s, a)
                V[a] += p * (r1 + self.gamma*self.V[s1]) # FIXME may need to change this to include some sort of argmax
            self.pi[s] = np.argmax(V)

    def train(self):
        for j in range(100):
            self.evaluate()
            self.improve()

        return self.pi
    
    def run(self):
        pi = self.train()
        s = self.gridworld.reset()
        done = False
        step = 0
        while not done:
            self.gridworld.render()
            (s, r, done) = self.gridworld.step(pi[s])
            if done:
                s = self.gridworld.reset()
            else:
                step += 1

    
            




