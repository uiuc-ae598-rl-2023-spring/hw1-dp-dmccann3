from gridworld import GridWorld
import numpy as np
import plotting

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
        self.V_prev = [0 for s in self.states]
        self.pi = [None for s in self.states]

        # discount factor
        self.gamma = 0.95

        # init env
        self.gridworld = GridWorld(False)

        # for plotting
        self.V_mean = []
        self.num_iter = []
        self.count_iter = 0


    def train(self):
        
        theta = 0.000001
        # Policy Evaluation
        while True: 
            delta = 0
            for s in self.states:
                A = [0 for a in self.actions]
                for a in self.actions:
                    self.gridworld.s = s
                    s1, r1, _ = self.gridworld.step(a)                    
                    # p = self.gridworld.p(s1, s, a)
                    A[a] = r1 + self.gamma*self.V[s1]
                V_best = max(A)
                delta = max(delta, abs(V_best - self.V[s]))
                self.V[s] = V_best
            if delta < theta:
                break

            # find mean and store for plotting
            mean = (1/25) * sum(self.V)
            self.V_mean.append(mean)
            self.num_iter.append(self.count_iter)
            self.count_iter += 1

        return self.V
    
    def get_policy(self):
        V = [0 for a in self.actions]
        for s in self.states:
            for a in self.actions:
                self.gridworld.s = s
                s1, r1, _ = self.gridworld.step(a)
                p = self.gridworld.p(s1, s, a)
                V[a] = r1 + self.gamma* np.sum(p * self.V[s1])
            self.pi[s] = np.argmax(V)

        return self.pi
    
    def run(self):
        S = []
        A = []
        R = []
        V = self.train()
        pi = self.get_policy()
        s = self.gridworld.reset()
        S.append(s)
        A.append(0)
        R.append(0)
        done = False
        step = 0
        while not done:
            self.gridworld.render()
            s, r, done = self.gridworld.step(pi[s])
            S.append(s)
            A.append(pi[s])
            R.append(r)
            if done:
                self.gridworld.reset()
            else:
                step += 1
        return S, A, R
        


