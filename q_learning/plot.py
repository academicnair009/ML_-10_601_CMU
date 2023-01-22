import argparse
import numpy as np

from environment import MountainCar, GridWorld

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.

    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.

            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.

    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *not* have the bias term 
folded in.
"""


def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate

    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


def select_eps_greedy_action(q_val_arr, epsilon, actions):
    ##https://piazza.com/class/l6xoswmdxo10m/post/1758
    if (np.random.rand() < epsilon or epsilon == 0):
        # greedy action
        return np.argmax(q_val_arr)
        # q_val=q_val_arr[0]
    else:
        return np.random.choice(actions)


if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    if env_type == "mc":
        env = MountainCar(mode, False)
    elif env_type == "gw":
        env = GridWorld(mode, False)
    else:
        raise Exception(f"Invalid environment type {env_type}")
    # R is reward list.
    # #print(str(env.action_space) +"and "+ str(env.state_space))
    R = []
    act_count = env.action_space
    state_count = env.state_space
    # w_0 =0
    weights = np.zeros((act_count, 1 + state_count), dtype=np.float64)
    #print("shape of weights is " + str(weights.shape))
    for episode in range(episodes):

        # Get the initial state by calling env.reset()
        # init_state = env.reset()
        rewards = 0
        state_arr = env.reset()
        state_arr = np.hstack((1, state_arr))
        # state_arr = np.zeros((state_count))
        # #print(state_map)
        # for i in state_map:
        #     # state_arr[i] = state_map[i]
        #     #print(state_arr)

        # state_arr = np.reshape(state_arr, (-1, 1))

        for iteration in range(max_iterations):
            # Compute the fold in bias for state
            # if state arr has dimension has sx1 then make it (s+1)x1
            #print("shape of weights is " + str(weights.shape))
            #print("shape of state arr " + str(state_arr.shape))
            # compute q_value for that state (return |a|x1 matrix)
            q_val_arr = weights @ state_arr

            # Select an action based on the state via the epsilon-greedy strategy
            action = select_eps_greedy_action(q_val_arr, epsilon, [0, 1, 2])
            # Take a step in the environment with this action, and get the
            # returned next state, reward, and done flag
            state_next, reward, done = env.step(action)
            state_next = np.hstack((1, state_next))
            q_val_arr_next = weights @ state_next
            # Using the original state, the action, the next state, and
            # the reward, update the parameters. Don't forget to update the
            # bias term!
            weights[action] -= lr * (q_val_arr[action] - (reward + gamma * (np.max(q_val_arr_next)))) * (state_arr)
            rewards += reward
            state_arr = state_next
            # Remember to break out of this inner loop if the environment signals done!
            if done:
                break

            ##print("h stack is now " + str(state_arr))
        R.append(rewards)
    ##print("Returns is ")
    ##print(R)
    ##print("Weights")
    ##print(weights)
    np.savetxt(weight_out, weights, fmt="%.18e", delimiter=" ")
    np.savetxt(returns_out, R, fmt="%.18e", delimiter=" ")

    # Save your weights and returns. The reference solution uses
    # np.savetxt(..., fmt="%.18e", delimiter=" ")