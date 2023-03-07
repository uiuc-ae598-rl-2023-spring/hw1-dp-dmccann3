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
        self.V_prev = [0 for s in self.states]
        self.pi = [0 for s in self.states]

        # discount factor
        self.gamma = 0.95

        # init env
        self.gridworld = GridWorld(False)

        # for plotting 
        self.V_mean = []
        self.num_eval = []
        self.count_eval = 0

    # need to get transition function and rewards into this function here
    def evaluate(self):
        max_iter = 10000
        self.count_eval += 1
        # Policy Evaluation
        for k in range(max_iter): # FIXME Most likely need to change this to reflect inifite time horizon
            for s in self.states:
                v = self.V[s]
                self.gridworld.s = s
                s1, r1, _ = self.gridworld.step(self.pi[s])                    
                p = self.gridworld.p(s1, s, self.pi[s])
                self.V[s] = np.sum(p * (r1 + self.gamma*self.V[s1]))

            if np.allclose(self.V_prev, self.V, rtol=0, atol=1e-20):
                break
            else:
                self.V_prev = self.V

        # calculate and store mean value function
        mean = (1/25) * sum(self.V)
        self.V_mean.append(mean)
        self.num_eval.append(self.count_eval)

        return self.V
    
    def improve(self):
        V = [0 for a in self.actions]
        pi = [0 for s in self.states]
        # Policy Improvement
        for s in self.states:
            for a in self.actions:
                self.gridworld.s = s
                s1, r1, _ = self.gridworld.step(a)
                p = self.gridworld.p(s1, s, a)
                V[a] = r1 + self.gamma*np.sum(p * self.V[s1]) # FIXME may need to change this to include some sort of argmax
            pi[s] = np.argmax(V)
        self.pi = pi

    def train(self):
        for j in range(100):
            v = self.evaluate()
            self.improve()

        return self.pi
    
    def run(self):
        S = []
        A = []
        R = []
        pi = self.train()
        s = self.gridworld.reset()
        S.append(s)
        A.append(0)
        R.append(0)
        done = False
        step = 0
        while not done:
            self.gridworld.render()
            A.append(pi[s])
            s, r, done = self.gridworld.step(pi[s])
            S.append(s)
            R.append(r)
            if done:
                s = self.gridworld.reset()
            else:
                step += 1
        return S, A, R

    
            




