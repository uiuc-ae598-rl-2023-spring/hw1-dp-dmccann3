from policy_iteration import PI
from value_iteration import VI
from sarsa import SARSA
from q_learning import Q_learning
from TD_0 import TD
from gridworld import GridWorld
import plotting
import numpy as np


def main():
    
    # Init GridWorld
    gridworld = GridWorld()

    # Policy Iteration
    policy_iter = PI()
    S_pi, A_pi, R_pi = policy_iter.run()
    plotting.plot(np.arange(0, len(policy_iter.states), 1), policy_iter.V, 'blue', 'PI State Value Func.', 'V(s)', 'State', 'V(s)', 'Plots/Gridworld/PI/pi_svf_plot.png')
    plotting.plot(policy_iter.num_eval, policy_iter.V_mean, 'blue', 'PI Mean Value Func.', 'V(s)', 'Num. Policy Evaluations', 'Mean Val. Func.', 'Plots/Gridworld/PI/pi_valfunc_plot.png')
    plotting.plot(np.arange(0, len(policy_iter.states), 1), policy_iter.pi, 'red', 'PI Policy', 'pi', 'State', 'Policy pi', 'Plots/Gridworld/PI/pi_policy_plot.png')
    plotting.plot_sar(S_pi, A_pi, R_pi, 'Trained PI Trajectory', 'time (s)', 'Plots/Gridworld/PI/sar.png')



    # Value Iteration
    value_iter = VI()
    S_vi, A_vi, R_vi = value_iter.run()
    plotting.plot(np.arange(0, len(value_iter.states), 1), value_iter.V, 'red', 'VI State Value Func.', 'V(s)', 'State', 'V(s)', 'Plots/Gridworld/VI/vi_svf_plot.png')
    plotting.plot(value_iter.num_iter, value_iter.V_mean, 'blue', 'VI Mean Value Func.', 'V(s)', 'Num. Value Iterations', 'Mean Val. Func.', 'Plots/vi_valfunc_plot.png')
    plotting.plot(np.arange(0, len(value_iter.states), 1), value_iter.pi, 'red', 'VI Policy', 'pi', 'State', 'Policy pi', 'Plots/Gridworld/VI/vi_policy_plot.png')
    plotting.plot_sar(S_vi, A_vi, R_vi, 'Trained VI Trajectory', 'time (s)', 'Plots/Gridworld/VI/sar.png')



    # SARSA
    num_episodes = 500
    returns_a_sarsa = []
    returns_e_sarsa = []
    episodes = []

    # SARSA with alpha = 0.1
    sarsa1 = SARSA(gridworld, num_episodes)
    sarsa_pi = sarsa1.train_sarsa(alpha=0.1, e=0.1)
    S_s, A_s, R_s = sarsa1.run()
    returns_a_sarsa.append(sarsa1.Return)
    returns_e_sarsa.append(sarsa1.Return)
    episodes.append(sarsa1.num_episodes)
    plotting.plot_sar(S_s, A_s, R_s, 'Trained SARSA Trajectory', 'time (s)', 'Plots/Gridworld/SARSA/sar.png')
    plotting.plot_return(sarsa1.num_episodes, sarsa1.Return, 'black', 'SARSA Return over Episodes', 'G', 'Num. Episodes', 'Return', 'Plots/Gridworld/SARSA/sarsa_return_plot.png')
    plotting.plot(np.arange(0, sarsa1.num_states, 1), sarsa_pi, 'black', 'SARSA Policy', 'pi', 'States', 'Policy', 'Plots/Gridworld/SARSA/sarsa_policy_plot.png')

    # SARSA with alpha = 0.3
    sarsa2 = SARSA(gridworld, num_episodes)
    sarsa_pi = sarsa2.train_sarsa(alpha=0.3, e=0.1)
    sarsa2.run()
    returns_a_sarsa.append(sarsa2.Return)
    episodes.append(sarsa2.num_episodes)

    # SARSA with alpha = 0.5
    sarsa3 = SARSA(gridworld, num_episodes)
    sarsa_pi = sarsa3.train_sarsa(alpha=0.5, e=0.1)
    sarsa3.run()
    returns_a_sarsa.append(sarsa3.Return)
    episodes.append(sarsa3.num_episodes)

    legend = ['a = 0.1', 'a= 0.3', 'a = 0.5']
    plotting.plot_mult_return(episodes, returns_a_sarsa, 'SARSA Returns with Varying alpha', legend, 'Return', 'Num. Episodes', 'Plots/Gridworld/SARSA/sarsa_vary_alpha.png')

    # SARSA with epsilon = 0.25
    sarsa4 = SARSA(gridworld, num_episodes)
    sarsa_pi = sarsa4.train_sarsa(alpha=0.1, e=0.25)
    sarsa4.run()
    returns_e_sarsa.append(sarsa4.Return)

    # SARSA with epsilon = 0.5
    sarsa5 = SARSA(gridworld, num_episodes)
    sarsa_pi = sarsa5.train_sarsa(alpha=0.1, e=0.50)
    sarsa5.run()
    returns_e_sarsa.append(sarsa5.Return)

    legend2 = ['e = 0.1', 'e = 0.25', 'e = 0.50']
    plotting.plot_mult_return(episodes, returns_e_sarsa, 'SARSA Returns with Varying epsilon', legend2, 'Return', 'Num. Episodes', 'Plots/Gridworld/SARSA/sarsa_vary_epsilon.png')


 

    # Q-Learning
    num_episodes = 500
    returns_a_q = []
    returns_e_q = []
    episodes = []

    q_learn1 = Q_learning(gridworld, num_episodes)
    qlearn_pi = q_learn1.train_qlearn(alpha=0.1, e=0.1)
    S_q, A_q, R_q = q_learn1.run()
    returns_a_q.append(q_learn1.Return)
    returns_e_q.append(q_learn1.Return)
    episodes.append(q_learn1.num_episodes)
    plotting.plot_sar(S_q, A_q, R_q, 'Trained Q_Learning Trajectory', 'time (s)', 'Plots/Gridworld/Q/sar.png')
    plotting.plot(q_learn1.num_episodes, q_learn1.Return, 'green', 'Q-Learning Return over Episodes', 'G', 'Num. Episodes', 'Return', 'Plots/Gridworld/Q/qlearn_plot.png')
    plotting.plot(np.arange(0, q_learn1.num_states, 1), qlearn_pi, 'green', 'Q-Learning Policy', 'pi', 'Num. States', 'Policy Pi', 'Plots/Gridworld/Q/q_policy_plot')

    # Q with alpha = 0.3
    q_learn2 = Q_learning(gridworld, num_episodes)
    qlearn_pi = q_learn2.train_qlearn(alpha=0.3, e=0.1)
    q_learn2.run()
    returns_a_q.append(q_learn2.Return)
    episodes.append(q_learn2.num_episodes)
    # Q with alpha = 0.3
    q_learn3 = Q_learning(gridworld, num_episodes)
    qlearn_pi = q_learn3.train_qlearn(alpha=0.3, e=0.1)
    q_learn3.run()
    returns_a_q.append(q_learn3.Return)
    episodes.append(q_learn3.num_episodes)

    legend = ['a = 0.1', 'a= 0.3', 'a = 0.5']
    plotting.plot_mult_return(episodes, returns_a_q, 'Q_learning Returns with Varying alpha', legend, 'Return', 'Num. Episodes', 'Plots/Gridworld/Q/q_vary_alpha.png')

    # Q with alpha = 0.3
    q_learn4 = Q_learning(gridworld, num_episodes)
    qlearn_pi = q_learn4.train_qlearn(alpha=0.1, e=0.25)
    q_learn4.run()
    returns_e_q.append(q_learn4.Return)
    # Q with alpha = 0.3
    q_learn5 = Q_learning(gridworld, num_episodes)
    qlearn_pi = q_learn5.train_qlearn(alpha=0.1, e=0.5)
    q_learn5.run()
    returns_e_q.append(q_learn5.Return)
    legend2 = ['e = 0.1', 'e = 0.25', 'e = 0.50']
    plotting.plot_mult_return(episodes, returns_e_q, 'Q_learning Returns with Varying epsilon', legend, 'Return', 'Num. Episodes', 'Plots/Gridworld/Q/q_vary_epsilon.png')


    # # TD(0)
    td_zero = TD(gridworld)
    sarsa_est = td_zero.train(sarsa_pi, alpha=1e-5, episodes=100)
    s_len = np.arange(0, td_zero.env.num_states, 1)
    plotting.plot(s_len, sarsa_est, 'blue', 'TD(0) SARSA Value Function Est.', 'V(s)', 'State', 'V(s)', 'Plots/Gridworld/TD0/sarsa_gridworld.png')
    qlearn_est = td_zero.train(qlearn_pi, alpha=1e-5, episodes=100)
    plotting.plot(s_len, qlearn_est, 'red', 'TD(0) Q-Learning Value Function Est.', 'V(s)', 'State', 'V(s)', 'Plots/Gridworld/TD0/q_gridworld.png')


if __name__ == "__main__":
    main()