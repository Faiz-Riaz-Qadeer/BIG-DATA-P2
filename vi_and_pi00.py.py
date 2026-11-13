 ### MDP Value Iteration and Policy Iteration

import numpy as np
from lake_envs import make, render_single

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        For each pair of states in [0, nS - 1] and actions in [0, nA - 1], P[state][action] is a
        list of tuples of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: float
                the reward for transitioning from "state" to "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

# =========================================================================
# PART (a): Policy Iteration Implementation
# =========================================================================

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma: defined at beginning of file
    policy: np.array[nS] - The policy to evaluate. Maps states to actions.
    tol: float - Terminate policy evaluation when max |V_new(s) - V_old(s)| < tol

    Returns
    -------
    value_function: np.ndarray[nS]
    """
    value_function = np.zeros(nS)

    while True:
        delta = 0
        V_new = np.copy(value_function)
        for s in range(nS):
            a = policy[s]
            # Bellman expectation update for state s
            v_s = sum(
                prob * (reward + gamma * value_function[nextstate])
                for prob, nextstate, reward, terminal in P[s][a]
            )
            V_new[s] = v_s
            delta = max(delta, abs(V_new[s] - value_function[s]))
            
        value_function = V_new
        if delta < tol:
            break

    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy, improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma: defined at beginning of file
    value_from_policy: np.ndarray - The value calculated from the policy
    policy: np.array - The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
    """
    new_policy = np.zeros(nS, dtype='int')

    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            q_sa[a] = sum(
                prob * (reward + gamma * value_from_policy[nextstate])
                for prob, nextstate, reward, terminal in P[s][a]
            )
        # Select action with maximum expected Q-value
        new_policy[s] = np.argmax(q_sa)

    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

    Returns
    -------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    num_iterations = 0

    while True:
        num_iterations += 1
        # Step 1: Policy Evaluation
        value_function = policy_evaluation(P, nS, nA, policy, gamma=gamma, tol=tol)
        
        # Step 2: Policy Improvement
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma=gamma)
        
        # Step 3: Check for policy convergence
        if np.array_equal(new_policy, policy):
            break
            
        policy = new_policy

    print(f"Policy Iteration converged in {num_iterations} iterations.")
    return value_function, policy


# =========================================================================
# PART (b): Value Iteration Implementation
# =========================================================================

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Learn value function and policy by using value iteration method.

    Returns
    -------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    num_iterations = 0

    # Step 1: Iteratively compute optimal value function V*(s)
    while True:
        num_iterations += 1
        delta = 0
        V_new = np.copy(value_function)
        for s in range(nS):
            q_sa = np.zeros(nA)
            for a in range(nA):
                q_sa[a] = sum(
                    prob * (reward + gamma * value_function[nextstate])
                    for prob, nextstate, reward, terminal in P[s][a]
                )
            V_new[s] = np.max(q_sa)
            delta = max(delta, abs(V_new[s] - value_function[s]))

        value_function = V_new
        if delta < tol:
            break

    # Step 2: Extract optimal policy pi*(s) from optimal value function
    for s in range(nS):
        q_sa = np.zeros(nA)
        for a in range(nA):
            q_sa[a] = sum(
                prob * (reward + gamma * value_function[nextstate])
                for prob, nextstate, reward, terminal in P[s][a]
            )
        policy[s] = np.argmax(q_sa)

    print(f"Value Iteration converged in {num_iterations} iterations.")
    return value_function, policy


# =========================================================================
# PART (c): Execution & Comparison (Deterministic vs. Stochastic)
# =========================================================================

def run_environment_tests(env_name):
    print("=" * 60)
    print(f" RUNNING EXPERIMENTS FOR ENVIRONMENT: {env_name}")
    print("=" * 60)

    env = make(env_name)

    print("\n--- Policy Iteration ---")
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print("Optimal Values (PI):\n", V_pi.reshape((4, 4)))
    print("Optimal Policy (PI):\n", p_pi.reshape((4, 4)))

    print("\n--- Value Iteration ---")
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print("Optimal Values (VI):\n", V_vi.reshape((4, 4)))
    print("Optimal Policy (VI):\n", p_vi.reshape((4, 4)))

    print("\nSimulating 1 episode with Policy Iteration result:")
    render_single(env, p_pi, 100)


if __name__ == "__main__":
    # Test 1: Deterministic Environment
    run_environment_tests("Deterministic-4x4-FrozenLake-v0")
    
    # Test 2: Stochastic Environment
    run_environment_tests("Stochastic-4x4-FrozenLake-v0")