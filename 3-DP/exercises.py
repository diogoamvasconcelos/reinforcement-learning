import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
 
############# 1 ###############

def policy_evaluation_exercise():
    env = GridworldEnv()

    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)
    print(v)

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        cached_v = np.copy(V)
        for s in range(env.nS):
            V[s] = 0
            for a in range(env.nA):
                p_a = policy[s, a]
                [(p, s_, r, d)] = env.P[s][a]
                V[s] += p_a * p * (r + discount_factor * cached_v[s_])
            
        if (np.sum(abs(cached_v - V)) < theta):
            break
    return np.array(V)

########### 2 #############

def policy_iteration_exercise():
    pp = pprint.PrettyPrinter(indent=2)
    env = GridworldEnv()

    policy, v = policy_improvement(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    # Test the value function
    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = np.zeros(env.nS)
    while True:
        cached_policy = np.copy(policy)

        #policy eval
        V = policy_eval_fn(policy, env)

        #policy improvement
        for s in range(env.nS): 
            Qs = np.zeros(env.nA)
            for a in range(env.nA):
                [(p, s_, r, d)] = env.P[s][a]
                Qs[a] = p * (r + discount_factor * V[s_])
            
            policy[s] = np.eye(env.nA)[np.argmax(Qs)] #one-hot conversion

        if (np.sum(abs(cached_policy - policy)) == 0):
            break
    
    return policy, V

########### 3 ###############
def value_iteration_exercise():
    pp = pprint.PrettyPrinter(indent=2)
    env = GridworldEnv()

    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    # Test the value function
    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # Implement!
    while True:
        cached_v = np.copy(V)
        for s in range(env.nS): 
            Qs = np.zeros(env.nA)
            for a in range(env.nA):
                [(p, s_, r, d)] = env.P[s][a]
                Qs[a] = p * (r + discount_factor * V[s_])

            V[s] = np.max(Qs)
        if (np.sum(abs(cached_v - V)) < theta):
            break

    #greedy over V to get the policy
    for s in range(env.nS):
        Qs = np.zeros(env.nA)
        for a in range(env.nA):
            [(p, s_, r, d)] = env.P[s][a]
            Qs[a] = V[s_]
        policy[s] = np.eye(env.nA)[np.argmax(Qs)]

    return policy, V

######### 4 #########
def gambler_problem_exercise():
    policy, v = value_iteration_for_gamblers(0.25)

    print("Optimized Policy:")
    print(policy)
    print("")

    print("Optimized Value Function:")
    print(v)
    print("")

    # Plotting Final Policy (action stake) vs State (Capital)
    # x axis values
    x = range(100)
    # corresponding y axis values
    y = v[:100]
    
    # plotting the points 
    plt.plot(x, y)
    
    # naming the x axis
    plt.xlabel('Capital')
    # naming the y axis
    plt.ylabel('Value Estimates')
    
    # giving a title to the graph
    plt.title('Final Policy (action stake) vs State (Capital)')
    
    # function to show the plot
    plt.show()

    # Plotting Capital vs Final Policy
    # corresponding y axis values
    y = policy
    
    # plotting the bars
    plt.bar(x, y, align='center', alpha=0.5)
    
    # naming the x axis
    plt.xlabel('Capital')
    # naming the y axis
    plt.ylabel('Final policy (stake)')
    
    # giving a title to the graph
    plt.title('Capital vs Final Policy')
    
    # function to show the plot
    plt.show()

def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """

    
    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            s: The gambler’s capital. Integer.
            V: The vector that contains values at each state. 
            rewards: The reward vector.
                        
        Returns:
            A vector containing the expected value of each action. 
            Its length equals to the number of actions.
        """ 
        nA = min(s, 100-s)
        Qs = np.zeros(nA)
        for a in range(nA):
            #win
            s_ = s + (a+1)
            Qs[a] += p_h * (rewards[s_] + discount_factor * V[s_])
            #lose
            s_ = s - (a+1)
            Qs[a] += (1-p_h) * (rewards[s_] + discount_factor * V[s_])
        
        return Qs
    
    # Implement!
    nS = 100
    rewards = np.zeros(nS+1)
    rewards[nS] = 1

    V = np.zeros(nS+1)
    policy = np.zeros(nS)

    while True:
        cached_v = np.copy(V)
        for s in range(1, nS):
            Qs = one_step_lookahead(s, V, rewards)
            V[s] = np.max(Qs)

        if (np.sum(abs(cached_v - V)) < theta):
            break

    #greedy over V to get the policy
    for s in range(1, nS):
        Qs = one_step_lookahead(s, V, rewards)
        policy[s] = np.argmax(Qs)
    
    return policy, V

if __name__ == "__main__":
    #policy_evaluation_exercise()
    #policy_iteration_exercise()
    #value_iteration_exercise()
    gambler_problem_exercise()