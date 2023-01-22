import numpy as np
import sys
import matplotlib.pyplot as plt
import math
def negative_loglikelihood(y,prod):
    # sig = sigmoid(np.dot(theta.transpose(),x))
    # # J = np.sum(-y.T * x * theta.T) + np.sum(np.exp(x * theta.transpose()))+ np.sum(np.log(y))
    # # j = -np.sum(y* np.log(sig) + (1 - y) * np.log(1 - sig)) / x.shape[1]  # compute cost
    # m=x.shape[0]
    # J = (np.dot(-(y.T), np.log(expit(np.dot(X, theta)))) - np.dot((np.ones((m, 1)) - y).T, np.log(
    #     np.ones((m, 1)) - (expit(np.dot(X, theta))).reshape((m, 1))))) / m + (regTerm / (2 * m)) * np.linalg.norm(
    #     theta[1:])


    ll= (y*prod) - np.log(1+np.exp(prod))

    return ll
def find_negative_log_likelihood(theta, X, y):
    N = X.shape[0]
    sum = 0
    for i in range(0, N):
        dot_p = np.dot(theta, X[i])
        sum += y[i]*dot_p - math.log(1 + math.exp(dot_p))
    return (-1/N)*sum

neg_log = []


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(theta, X, y, num_epoch, learning_rate):
    (r, c) = X.shape
    sum=0
    for i in range(num_epoch):
        for j in range(r):

            np_dot = np.dot(theta.transpose(), X[j])
            # ll= negative_loglikelihood(y[j],np_dot)
            # sum=sum+ll
            theta = theta + (y[j] - sigmoid(np_dot)) * X[j] * learning_rate
        # neg_log.append(-1*sum/len(y))
        neg_log.append(find_negative_log_likelihood(theta=theta,X=X,y=y))
    print(neg_log)
    return theta


def predict(theta, X, file_output):
    r, c = X.shape
    # print_list = np.empty(shape=[0, 1])
    print_list = []

    f = open(file_output, "a")
    for i in range(r):
        np_dot = np.dot(theta.transpose(), X[i])
        if sigmoid(np_dot) < 0.5:
            print_list.append(0)
            f.write(str(0))
        else:
            print_list.append(1)
            f.write(str(1))
        f.write("\n")
    f.close()
    return print_list


def compute_error(y_pred, y):
    e=0
    for i in range(len(y_pred)):
        if int(y[i])!= int(y_pred[i]):
            e=e+1
    return e/len(y_pred)


if __name__ == '__main__':
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metric_out = sys.argv[6]
    num_epoch = int(sys.argv[7])
    learning_rate = float(sys.argv[8])

    full_data = np.loadtxt(open(train_input, "rb"), delimiter="\t")
    print(full_data)
    theta_size = full_data.shape[1]
    y = full_data[:, 0]
    unfold_x = full_data[:,1:]
    print("unfolded shape "+str(unfold_x.shape))
    x = np.insert(unfold_x, 0, 1, axis=1)
    print(x.shape)
    theta = np.zeros(theta_size)

    coeff = train(theta, x, y, num_epoch, learning_rate)
    neg_log_train = np.array(neg_log)
    neg_log=[]


    full_data = np.loadtxt(open(train_input, "rb"), delimiter="\t")
    print(full_data)
    theta_size = full_data.shape[1]
    y = full_data[:, 0]
    unfold_x = full_data[:,1:]
    print("unfolded shape "+str(unfold_x.shape))
    x = np.insert(unfold_x, 0, 1, axis=1)
    print(x.shape)
    theta = np.zeros(theta_size)

    coeff = train(theta, x, y, num_epoch, learning_rate*0.1)
    neg_log_1 = np.array(neg_log)
    neg_log=[]


    full_data = np.loadtxt(open(train_input, "rb"), delimiter="\t")
    print(full_data)
    theta_size = full_data.shape[1]
    y = full_data[:, 0]
    unfold_x = full_data[:,1:]
    print("unfolded shape "+str(unfold_x.shape))
    x = np.insert(unfold_x, 0, 1, axis=1)
    print(x.shape)
    theta = np.zeros(theta_size)

    coeff = train(theta, x, y, num_epoch, learning_rate*0.01)
    neg_log_2 = np.array(neg_log)


    plt.figure()  # In this example, all the plots will be in one figure.
    plt.plot(neg_log_train, label="n=10-3")
    plt.plot(neg_log_1, label="n=10-4")
    plt.plot(neg_log_2, label="n=10-5")
    plt.legend()

    # plt.plot(neg_log_train,label="train")
    # plt.plot(neg_log_val,label="validation")

    plt.savefig('all_val.png', dpi=300, bbox_inches='tight')

    y_train_pred = predict(coeff, x, train_out)

    full_data = np.loadtxt(open(test_input, "rb"), delimiter="\t")
    y_train = y
    y = full_data[:, 0]
    unfold_x = full_data[:,1:]
    x = np.insert(unfold_x, 0, 1, axis=1)
    y_test_predict = predict(coeff, x, test_out)
    print("y train is"+ str(y_train_pred))
    print("y test is"+ str(y_test_predict))
    with open(metric_out,'w') as error_out:
        error_out.write("error(train): "+"{:.6f}".format(compute_error(y_train_pred, y_train))+'\n')
        error_out.write("error(test): " + "{:.6f}".format(compute_error(y_test_predict,y)) + '\n')