import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)

    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}

    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}

    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file


# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def my_lse(arr):
    arr_max = np.max(arr)
    diff = arr - arr_max
    sum = np.exp(diff).sum()
    return arr_max + np.log(sum)


def forwardbackward(seq, loginit, logtrans, logemit):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)
    output = []
    error=0
    final_length=0
    likelihood=0
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    validation_data = seq
    # print("validation data is \n"+str(validation_data))
    with open(predicted_file, "w") as f:
        for outer in range(len(validation_data)):
            a = np.zeros((len(validation_data[outer]),M))
            b= np.zeros((len(validation_data[outer]),M))
            for inner1 in range(M):
                # print("1. inner1 "+ str(words_to_indices.get(validation_data[outer][0][0])))
                a[0][inner1]=loginit[inner1]+logemit[inner1][words_to_indices.get(validation_data[outer][0][0])]
                b[len(validation_data[outer])-1][inner1]=0
                # print("alpha is "+str(a))
                # print("beta  is "+str(b))


            for t in range(1,len(validation_data[outer])):
                for inner2 in range(M):
                    at_1=a[t-1,:]
                    # print("alpha t-1 is " + str(a))
                    emiss = logemit[inner2,words_to_indices.get(validation_data[outer][t][0])]
                    trans = logtrans[:,inner2]
                    # print("trans + alpha_t_1 is "+str(trans+at_1))
                    a[t,inner2]=emiss+my_lse(trans+at_1)#TODO CHECK MULTIPLICATION/ADDITION
                    # a[t,inner2]=emiss+np.sum((np.dot(trans,at_1)))#TODO CHECK MULTIPLICATION/ADDITION
            # for t in range(1,len(validation_data[outer])):
            #     for inner2 in range(M):
            #         at_1=a[t-1,:]
            #         emiss = logemit[inner2,words_to_indices.get(validation_data[outer][t][0])]
            #         trans = logtrans[:,inner2]
            #         a[t,inner2]=emiss*np.sum((np.dot(trans,at_1)))#TODO CHECK MULTIPLICATION/ADDITION
            for t in range(len(validation_data[outer])-2,-1,-1):
                for inner3 in range(M):
                    trans= logtrans[inner3][:]
                    emiss = logemit[:,words_to_indices.get(validation_data[outer][t+1][0])]
                    b_next = b[t+1][:]
                    b[t][inner3] = my_lse(emiss+trans+b_next)
        #     for t in range(len(validation_data[outer])-2,-1,-1):
        #         for inner3 in range(M):
        #             trans= logtrans[inner3][:]
        #             emiss = logemit[:,words_to_indices.get(validation_data[outer][t+1][0])]
        #             b_next = b[t+1][:]
        #             b[t][inner3] = np.sum(b_next*trans*emiss)
            print("Final alpha = "+str(a))
            print("Final Beta = "+str(b))
            for i in range(len(validation_data[outer])):
                p=a[i,:]+b[i,:]

                pred=np.argmax(p.transpose())
                # print("Actual values are "+str(validation_data[outer][i][1]))
                # print("predicted tags are "+ str(pred))
                output.append(pred)
                for key in tags_to_indices:
                    if tags_to_indices[key]==pred:
                        f.write(validation_data[outer][i][0]+"\t"+key + "\n")
                if(tags_to_indices.get(validation_data[outer][i][1])==pred):
                    error=error+1
            f.write("\n")
            # print(a[-1,:])
            likelihood+=my_lse(a[-1,:])

        f.close()
        final_length=final_length+len(validation_data[outer])
        # likelihood+=np.sum(a[-1,:])
        # print("final len "+str(final_length))
        # print("final acc "+str(error))
    return error/len(output),output,likelihood


    # Initialize log_alpha and fill it in

    # alpha = np.zeros((L,M))
    # print(alpha)


    # Initialize log_beta and fill it in

    # Compute the predicted tags for the sequence

    # Compute the log-probability of the sequence

    # Return the predicted tags and the log-probability
    pass

# def fwd_func(validation_array,emiss,trans,initial):
#     alfa = np.zeros((len(validation_array),trans.shape[0]))
#     alfa =
#     print(alfa)


if __name__ == "__main__":
    # Get the input data

    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.

    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    # print("validation_data" + str(validation_data))
    # print("words_to_indices "+str(words_to_indices))
    # print("tags_to_indices "+str(tags_to_indices))
    # print("hmminit " +str(hmminit))
    # print("hmmemit "+str(hmmemit))
    # print("hmmtrans "+str(hmmtrans))
    # print("predicted_file "+str(predicted_file))
    # print("predicted_file "+str(metric_file))

    log_init = np.log(hmminit)
    log_emmit = np.log(hmmemit)
    log_tran = np.log(hmmtrans)
    seq_2d = []
    #getting the sequence from validation data
    # for i in range(0,len(validation_data)):
    #     # x_list_i = validation_data[i]
    #     col = []
    #     for j in range(0,len((validation_data[i]))):
    #         col.append(validation_data[i][j][0])
    #         # print(col)
    #     seq_2d.append(col)
    # print(seq_2d)



    err,predict_list,ll=forwardbackward(validation_data,log_init,log_tran,log_emmit)
    # import matplotlib.pyplot as plt
    #
    x = [10,100,1000,10000]
    x=np.array(x)
    x=np.log(x)
    # out=[]
    # for a in x:
    #     err, predict_list, ll = forwardbackward(validation_data[:a], log_init, log_tran, log_emmit)
    #     out.append(ll/a)
    #     print(out)
    #
    # plt.show()
    # # with open(metric_file, "w") as f:
    # #     f.write("Average Log-Likelihood: "+str(ll/len(validation_data))+"\n")
    # #     f.write("Accuracy: "+str(err)+"\n")
    #
    print(err)
    print("LL "+str(ll/len(validation_data)))
    # print(predict_list)
    valid_list_to_plot = [-87.97,-80.83,-70.46,-61.09]
    train_to_plot = [-80.54, -74.99,-67.59,-60.61]
    from matplotlib import pyplot as plt
    plt.plot(x, valid_list_to_plot, 'r', label="Valid LL")  # plotting t, a separately
    plt.plot(x, train_to_plot, 'b', label="Train LL")  # plotting t, b separately

    plt.xlabel("Number of sequences")
    plt.ylabel("Average log likely")
    plt.legend()

    plt.savefig('PLOT1.png')
#python3 forwardbackward1.py en_data/validation.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_data/hmminit.txt en_data/hmmemit.txt en_data/hmmtrans.txt en_data/predicted.txt en_data/metrics.txt
#python3 forwardbackward1.py toy_data/validation.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_data/hmminit.txt toy_data/hmmemit.txt toy_data/hmmtrans.txt toy_data/predicted.txt toy_data/metrics.txt
