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


if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    if env_type == "mc":
        env = MountainCar(mode)
    elif env_type == "gw":
        env = GridWorld(mode)
    else: raise Exception(f"Invalid environment type {env_type}")

    # print(str(env.action_space) +"and "+ str(env.state_space))
    ##initilaizing
    R = []
    act_count = env.action_space
    state_count = env.state_space
    w_0 =0
    weight = np.zeros((act_count, state_count+1))

    for episode in range(episodes):
        flag = False
        # Get the initial state by calling env.reset()
        init_state = env.reset()
        reward = 0
        state_arr = env.reset()
        # state_arr = np.zeros((state_count))
        # print(state_map)
        # for i in state_map:
        #     # state_arr[i] = state_map[i]
        #     print(state_arr)
        state_arr = np.reshape(state_arr, (-1, 1))

        for iteration in range(max_iterations):
            if(flag):
                break
            # max_q=None
            q_list = []
            action =0
            q_val = 0

            if(np.random.random_sample()<=epsilon):
                action = np.random.randint(0,act_count)
                q_val_arr = np.matmul(state_arr.T,weight[action].T)+w_0
                print("state is "+ str(state_arr.T))
                print("wt is " + str(weight.T))
                print("first q val is" +str(q_val_arr))
                q_val=q_val_arr[0]
            else:
                for i in range(act_count):
                    q_list.append(np.matmul(state_arr.T,weight[i].T)+w_0)
                action=np.argmax(q_list)
                print("second q val is" +str(q_list[action][0]))
                q_val=q_list[action][0]
            s_next = np.zeros((state_count))
            q_list.clear()
            for i in range(len(env.step(action)[0])):
                print("action is "+str(action))
                s_next[i]=env.step(action)[0][i]
            s_next = np.reshape(s_next,(-1,1))
            for i in range(act_count):
                q_list.append(np.matmul(w_0+s_next.transpose(),weight[i].transpose()))
            x = lr*(q_val-(gamma*(max(q_list))+env.step(action)[1]))
            reward= reward+env.step(action)[1]
            print("q_val is" +str(q_val))
            weight[action]= weight[action] - x*state_arr.transpose()
            w_0-= x
            # print(env.step(action))
            if env.step(action)[2]==True:
                break
            state_arr = s_next
            R.append(reward)
    print("Returns is ")
    print(R)
    print("Weights")
    print(weight)
    print("init state is " + str(init_state))

    # Select an action based on the state via the epsilon-greedy strategy

            # Take a step in the environment with this action, and get the 
            # returned next state, reward, and done flag

            # Using the original state, the action, the next state, and 
            # the reward, update the parameters. Don't forget to update the 
            # bias term!

            # Remember to break out of this inner loop if the environment signals done!

    
    # Save your weights and returns. The reference solution uses 
    # np.savetxt(..., fmt="%.18e", delimiter=" ")
