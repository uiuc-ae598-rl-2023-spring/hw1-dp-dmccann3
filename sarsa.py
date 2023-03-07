import numpy as np

class SARSA(object):

    def __init__(self, env, num_episodes):
        self.env = env
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.pi = np.zeros(env.num_states)
        self.gamma = 0.95
        self.num_states = self.env.num_states

        # for plotting
        self.Return = np.array([])
        self.episodes = num_episodes
        self.num_episodes = np.arange(0, num_episodes, 1)


    def get_action(self, s, e):
        if np.random.uniform(0, 1) < e:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.Q[s, :])

        return action
    

    def train_sarsa(self, alpha, e):

        # training loop
        for i in range(self.episodes):
            t = 0
            r_list = []
            s = self.env.reset()
            a = self.get_action(s, e)
            while t < self.episodes:               
                s1, r1, done = self.env.step(a)
                r_list.append(r1)
                a1 = self.get_action(s1, e)
                self.Q[s][a] = self.Q[s][a] + alpha * ((r1 + self.gamma * self.Q[s1][a1]) - self.Q[s][a])
                if done:
                    s = self.env.reset()
                else:
                    s = s1
                    a = a1
                    t += 1
            # get return
            return_ = r_list.pop(0)
            for i in range(len(r_list)):
                return_ += (self.gamma**i) * r_list[i] 
            self.Return = np.append(self.Return, np.array([return_]))

        for s in range(self.env.num_states):
            self.pi[s] = np.argmax(self.Q[s, :])

        return self.pi


    def run(self):
        S = []
        A = []
        R = []
        s = self.env.reset()
        S.append(s)
        A.append(0)
        R.append(0)
        done = False
        step = 0
        while not done:
            self.env.render()
            A.append(np.argmax(self.Q[s]))
            s, r, done = self.env.step(np.argmax(self.Q[s]))
            S.append(s)
            R.append(r)
            if done:
                s = self.env.reset()
            else:
                step += 1
        return S, A, R







            
