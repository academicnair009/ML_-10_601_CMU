from environment import MountainCar
# environment is the executable python library provided by the course intructors
import numpy as np
import sys


def weight(state, w, b):  # state is the dictionary here
    sTw = np.zeros(3)

    for i in range(act_set):
        for j in range(len(state[0])):
            sTw[i] += w[j][i] * state[j]
    return sTw + b


def Action_select(q_vals, epsilon):
    prob = np.random.random(1)
    if prob < 1 - epsilon:
        a = np.argmax(q_vals)
    else:
        a = np.random.randint(0, 3)
    return a


def Q_train(alpha, gamma, epsilon, max_iterations):
    w = np.zeros((SS, act_set))  # Initialize
    b = 0  # Initialize
    Rewards = []

    for noe in range(episodes):
        state = Car.reset()
        print(state)
        r = 0  # Initialize reward
        done = False

        for m in range(max_iterations):
            if done == True:
                break

            q_vals = weight(state, w, b)
            a = Action_select(q_vals, epsilon)
            Q = q_vals[a]
            Sprime, reward, done = Car.step(a)

            '''Computing q_pi (s,a)'''
            Qprime = weight(Sprime, w, b)
            Q_next = max(Qprime)

            '''Gradient Update'''
            grad = alpha * (Q - (reward + gamma * Q_next))
            for j in range(len(state)):
                w[j][a] = w[j][a] - grad * state[j]

            b = b - grad
            state = Sprime
            r += reward

            ## Rendering ##
            '''Executed to see improvements after every 1000 episodes else it slows the overall execution'''
            # if noe % 1000 == 0:
            #     # MountainCar.render(Car)

        # env
        Rewards.append(r)

    MountainCar.close(Car)
    return w, b, Rewards


if __name__ == "__main__":
    pass

mode = sys.argv[1]
weight_out = sys.argv[2]
return_out = sys.argv[3]
episodes = int(sys.argv[4])
max_iter = int(sys.argv[5])
epsilon = float(sys.argv[6])
gamma = float(sys.argv[7])
alpha = float(sys.argv[8])

Car = MountainCar(mode)
SS = Car.state_space
print("initial state "+str(SS))
act_set = 3  # Action space has 3 options: (Left, No Action, Right)

W, B, Rewards = Q_train(alpha, gamma, epsilon, max_iter)
print(W)
# Weight files
'''Writing the output of the weights of the model learned'''
with open(weight_out, 'w+') as wt_file:
    wt_file.write('%s' % (B) + '\n')
    for j in range(SS):
        for i in range(act_set):
            wt_file.write('%s' % (W[j, i]) + '\n')

# Return files
'''Writing the values obtained by implementation of Q-learning algorithm after every iteration'''
with open(return_out, 'w+') as ret_file:
    for j in range(episodes):
        ret_file.write('%s' % (Rewards[j]) + '\n')