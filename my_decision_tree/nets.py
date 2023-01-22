import numpy as np
import sys

if __name__ == '__main__':
    infile_train = sys.argv[1]
    # infile_test = sys.argv[2]
    # train_label = sys.argv[3]
    # test_label = sys.argv[4]
    # metrics = sys.argv[5]
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

    #
    # def train(data):
    #     required_data = data[:, -1]
    #     return my_mode(required_data)
    #
    #
    # def predict(data, filename):
    #     predicted = np.empty(data.size)
    #     print("mode inside predict is" + str(model))
    #     predicted.fill(model)
    #     np.savetxt(filename, predicted, fmt="%1d", delimiter=",", newline="\n")


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


    train_data = read_file(infile_train)

    print(train_data[train_data[:,2]==0])
    # model = train(train_data)
    # predict(train_data[:, -1], train_label)
    # predict(test_data[:, -1], test_label)
    # calc_error(train_data[:, -1], test_data[:, -1])