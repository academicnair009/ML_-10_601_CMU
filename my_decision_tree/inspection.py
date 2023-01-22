# This is a sample Python script.
import numpy as np
import sys
from math import log as log

if __name__ == '__main__':
    infile_train = sys.argv[1]
    metrics = sys.argv[2]
    model = 1


    def my_mode(numpy_arr):
        values, cnts = np.unique(numpy_arr, return_counts=True)
        index = np.argmax(cnts)
        print("mode is " + str(values[index]))
        if cnts[0] == cnts[1]:
            return 1
        return values[index]


    def read_file(train_file_path: str):
        data = np.genfromtxt(fname=train_file_path, delimiter="\t", skip_header=1,
                             filling_values=1)
        return data


    def train(data):
        required_data = data[:, -1]
        return my_mode(required_data)


    def predict(data, filename):
        predicted = np.empty(data.size)
        print("mode inside predict is" + str(model))
        predicted.fill(model)
        np.savetxt(filename, predicted, fmt="%1d", delimiter=",", newline="\n")


    # def calc_error(train_df, test_df):
    #     train_error = 0
    #     test_error = 0
    #     for i in range(train_df.size):
    #         if train_df[i] != model:
    #             train_error = train_error + 1
    #     for i in range(test_df.size):
    #         if test_df[i] != model:
    #             test_error = test_error + 1
    #     f = open(metrics, "a")
    #     f.write("error(train): " + str(float(format(train_error / train_df.size, '.6f'))) + "\n")
    #     f.write("error(test): " + str(float(format(test_error / test_df.size, '.6f'))) + "\n")
    #     f.close()
    def calc_h(d_train):
        fraction_zeros = np.count_nonzero(d_train == 0)/d_train.size
        fraction_ones = 1-fraction_zeros
        print(np.count_nonzero(d_train == 0))
        print(fraction_ones)
        print(fraction_zeros)
        entropy = -1*(fraction_ones * log(fraction_ones, 2)+ fraction_zeros * log(fraction_zeros,2))
        f = open(metrics, "a")
        f.write("entropy: " + str(float(format(entropy, '.6f'))) + "\n")
        if model==0:
            f.write("error: " + str(float(format(fraction_ones, '.6f'))) + "\n")
        else:
            f.write("error: " + str(float(format(fraction_zeros, '.6f'))) + "\n")



    train_data = read_file(infile_train)
    model = train(train_data)
    calc_h(train_data[:, -1])
#model now has 0 or 1, we need to analyze
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
