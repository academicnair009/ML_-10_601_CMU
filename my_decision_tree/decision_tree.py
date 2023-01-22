import sys
import numpy as np
from math import log as log


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.parent_attr = None
        self.isLeft = None
        self.num_of_zeros = None
        self.num_of_ones = None

    # def insert(self, val):
    #     if self.val is None:
    #         self.val = val
    #         return
    #
    #     if self.val == val:
    #         return
    #
    #     if val<self.val:
    #         if self.left is not None:
    #             self.left.insert(val)
    #             return
    #         self.left = Node(val)
    #         return
    #
    #     if self.right is not None:
    #         self.right.insert(val)
    #         return
    #     self.right = Node(val)


if __name__ == '__main__':
    # train_input = sys.argv[1]
    # test_input = sys.argv[2]
    # max_depth = int(sys.argv[3])
    # train_out = sys.argv[4]
    # test_out = sys.argv[5]
    # metrics_out = sys.argv[6]
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    model = 1


    def my_mode(numpy_arr):
        values, cnts = np.unique(numpy_arr, return_counts=True)
        if len(values) == 1:
            return values[0]
        index = np.argmax(cnts)
        # print("mode is " + str(values[index]))
        # print("cnts value is ",cnts)
        if cnts[0] == cnts[1]:
            return 1
        return values[index]


    def read_file(train_file_path: str):
        header = np.genfromtxt(train_file_path, delimiter="\t", dtype=str, max_rows=1)

        data = np.genfromtxt(fname=train_file_path, delimiter="\t", skip_header=1,
                             filling_values=1)
        # data=np.delete(data, 0, axis=0)
        print(data)
        return data, header


    def majority_vote(data):
        required_data = data[:, -1]
        return my_mode(required_data)


    def calc_h(d_train):
        # print("in function calc_h")
        if (d_train.size == 0):
            return 0
        fraction_zeros = np.count_nonzero(d_train == 0) / d_train.size
        # print("the fraction of zeros is "+str(fraction_zeros))
        fraction_ones = 1 - fraction_zeros
        # print("the fraction of ones is "+str(fraction_ones))
        # print(np.count_nonzero(d_train == 0))
        # print(fraction_ones)
        # print(fraction_zeros)
        if fraction_ones == 0 or fraction_zeros == 0:
            return 0
        entropy = -1 * (fraction_ones * log(fraction_ones, 2) + fraction_zeros * log(fraction_zeros, 2))
        return entropy


    def calc_mi(data_for_mi, i: int):
        # print("in function calc_mi")
        y = data_for_mi[:, -1]
        entropy_y = calc_h(y)
        data_0_split = data_for_mi[data_for_mi[:, i] == 0]
        data_1_split = data_for_mi[data_for_mi[:, i] == 1]
        mi = entropy_y - ((data_0_split.size / data_for_mi.size) * calc_h(data_0_split[:, -1]) + (
                data_1_split.size / data_for_mi.size) * calc_h(data_1_split[:, -1]))
        return mi, data_0_split, data_1_split


    def func_split(data):
        # print("in function split")
        index_max_mi = 0
        max_mi = 0
        left_data = None
        right_data = None
        for col in range(data.shape[1] - 2, -1, -1):
            # print("checking in column "+str(col))
            temp_mi, temp_left, temp_right = calc_mi(data, col)
            if temp_mi > max_mi:
                max_mi = temp_mi
                index_max_mi = col
                left_data = temp_left
                right_data = temp_right
        return index_max_mi, max_mi, left_data, right_data


    def pretty_print(root, header_arg, depth=0):
        temp_pipe = "| " * (depth-3)
        if (root.parent_attr is not None):
            print(temp_pipe + header_arg[root.parent_attr] + " [" + str(root.num_of_zeros) + " 0/" + str(
                root.num_of_ones) + " 1 ]")
        if root.left is not None:
            pretty_print(root.left, header_arg, depth + 1)
        if root.right is not None:
            pretty_print(root.right, header_arg, depth + 1)


    def tree_builder(input_data, depth, header_arg=None):
        # print("the current depth is "+str(depth))
        # will be increased
        p = Node()
        _,c=input_data.shape
        c=c-1
        # print(input_data)
        entropy = calc_h(input_data[:, -1])
        uniq_0 = np.count_nonzero(input_data[:, -1] == 0)
        uniq_1 = np.count_nonzero(input_data[:, -1] == 1)
        p.num_of_ones = uniq_1
        p.num_of_zeros = uniq_0
        # print("entopy ", entropy)
        if depth == max_depth or entropy == 0 or input_data.size == 0 or depth==c:
            # not (input_data[:, -1] == 0).all() or \
            p.vote = majority_vote(input_data)
            return p

        index_max_mi, max_mi, left_data, right_data = func_split(input_data)
        # print("type of index_max_mi is ",type(index_max_mi))
        # print("the value of attribute is" + header_arg[index_max_mi])
        # print("left data", left_data)
        # print("right data", right_data)
        p.attr = index_max_mi
        # print("| " * (depth + 1), header[p.attr])
        p.left = tree_builder(left_data, depth + 1, header_arg)
        p.left.parent_attr = p.attr
        p.left.isLeft = True
        p.right = tree_builder(right_data, depth + 1, header_arg)
        p.right.parent_attr = p.attr
        # p.right.isLeft = False
        # print("| "*(depth+1),header[p.attr])

        return p


    def predict(node, feature):
        if node.vote is not None:
            return node.vote
        if feature[node.attr] == 0 and node.vote is None:
            # print("going left")
            return predict(node.left, feature)
        else:
            # print("going right")
            return predict(node.right, feature)


    def predict_in_file(testing_data, vote, filename, type="train"):
        rows, _ = testing_data.shape
        err = 0
        f = open(filename, "a")
        for i in range(0, rows):
            # np.append(predict_arr, predict(vote, testing_data[i]))
            f.write(str(predict(vote, testing_data[i]))+ "\n")
            if predict(vote, testing_data[i])!= testing_data[:, -1][i]:
                err = err + 1
        f.close()
        f = open(metrics_out, "a")
        f.write("error(" + type + "): " + str(float(format(err / rows, '.6f'))) + "\n")
        f.close()


    train_data, header = read_file(train_input)
    test_data, _ = read_file(test_input)
    new_model = tree_builder(train_data, 0, header)
    pretty_print(new_model,header,max_depth)
    predict_in_file(train_data, new_model, train_out)
    predict_in_file(test_data, new_model, test_out, "test")
