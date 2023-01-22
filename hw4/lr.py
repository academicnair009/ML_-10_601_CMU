import numpy as np
import sys


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
    for i in range(num_epoch):
        for j in range(r):
            np_dot = np.dot(theta.transpose(), X[j])
            theta = theta + (y[j] - sigmoid(np_dot)) * X[j] * learning_rate
            print("shaspe of theta ", theta.shape)
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
    print("final theta is ",str(coeff))
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