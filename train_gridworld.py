from policy_iteration import PI
from value_iteration import VI
from sarsa import SARSA
from q_learning import Q_learning
from TD_0 import TD
from gridworld import GridWorld


def main():
    
    # Init GridWorld
    gridworld = GridWorld()

    # Policy Iteration
    policy_iter = PI()
    policy_iter.run()

    # Value Iteration
    value_iter = VI()
    value_iter.run()

    # SARSA
    sarsa = SARSA(gridworld)
    sarsa_pi = sarsa.train(alpha=1e-5, threshold=0.1, episodes=100)
    sarsa.run()

    # Q-Learning
    q_learn = Q_learning(gridworld)
    qlearn_pi = q_learn.train(alpha=1e-5, threshold=0.1, episodes=100)
    q_learn.run()

    # TD(0)
    td_zero = TD(gridworld)
    sarsa_est = td_zero(sarsa_pi, alpha=1e-5, threshold=0.1, episodes=100)
    qlearn_est = td_zero(qlearn_pi, alpha=1e-5, threshold=0.1, episodes=100)


if __name__ == "__main__":
    main()