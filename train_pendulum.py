from sarsa import SARSA
from q_learning import Q_learning
from TD_0 import TD
from discrete_pendulum import Pendulum

def main():
    
    
    # Init discrete pendulum
    pendulum = Pendulum()

    # SARSA
    sarsa = SARSA(pendulum)
    sarsa_pi = sarsa.train(alpha=1e-5, threshold=0.1, episodes=100)
    sarsa.run()

    # Q-Learning
    q_learn = Q_learning(pendulum)
    qlearn_pi = q_learn.train(alpha=1e-5, threshold=0.1, episodes=100)
    q_learn.run()

    # TD(0)
    td_zero = TD(pendulum)
    sarsa_est = td_zero.train(sarsa_pi, alpha=1e-5, threshold=0.1, episodes=100)
    qlearn_est = td_zero.train(qlearn_pi, alpha=1e-5, threshold=0.1, episodes=100)
    


if __name__ == "__main__":
    main()