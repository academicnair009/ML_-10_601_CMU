import csv
import numpy as np
import sys

VECTOR_LEN = 300  # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt


################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


# def take_input_write_to_file(data,word2v):

def trim(xi, vector_dic):
    list_new_x = []
    print(len(xi))
    for i in range(len(xi)):
        # inside the first column of xi
        sentence_arr = xi[i][1].split(" ")
        new_sentence = ""
        for j in range(len(sentence_arr)):
            if sentence_arr[j] in vector_dic.keys():
                new_sentence = new_sentence + sentence_arr[j] + " "
        list_new_x.append([xi[i][0], new_sentence])
    print(str(i) + "th element is " + str(list_new_x))
    return np.array(list_new_x)


def calculate_feature(x_nd_array, word2vec):
    final = []
    r,c = x_nd_array.shape
    for i in range(r):
        # for ith review
        sentence_arr = x_nd_array[i][1].split(" ")
        l=0
        if sentence_arr[0].strip() != '':
            sum = word2vec.get(sentence_arr[0].strip())
            l=l+1
        if(len(sentence_arr)>1):
            for j in range(1, len(sentence_arr)):
                if sentence_arr[j].strip() != '':
                    sum = np.add(sum, word2vec[sentence_arr[j].strip()])
                    l = l + 1
            sum = np.divide(sum, float(l))
            label = np.array([float(x_nd_array[i][0])])
            sum = np.concatenate((label, sum))
            print(sum.shape)

            # sum=np.append(np.array([[float(x_nd_array[i])]]), sum, axis=1)
            # np.insert(sum,0,float(x_nd_array[i]))
            final.append(sum)
    print(np.array(final))
    return final


def task(in_file, out_file,word):
    x = load_tsv_dataset(in_file)
    # data_val_1 = load_tsv_dataset(val_input)
    # data_test_1 = load_tsv_dataset(test_input)
    w2v = load_feature_dictionary(word)  # dictionary of decimal values in nd array format
    x_trim = trim(x, w2v)
    features = calculate_feature(x_trim, w2v)
    np.savetxt(out_file, features, fmt='%.6f', delimiter="\t")

if __name__ == '__main__':
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    test_input = sys.argv[3]
    word2vec = sys.argv[4]
    train_out = sys.argv[5]
    val_out = sys.argv[6]
    test_out = sys.argv[7]
    #
    # train_input = "train_small.tsv"
    # word2vec = "word2vec.txt"
    task(train_input,train_out,word2vec)
    task(val_input,val_out,word2vec)
    task(test_input,test_out,word2vec)





