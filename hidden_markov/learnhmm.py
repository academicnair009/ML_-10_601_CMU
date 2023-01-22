import argparse
import numpy as np

index_tag_input_file =''
def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()
    global index_tag_input_file
    index_tag_input_file = args.index_to_tag
    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


def wordLister(directory:str):
    examples = None
    with open(index_tag_input_file, "r") as f:
        examples = f.read().strip().split("\n")
        # print(examples)
    return examples
if __name__ == "__main__":
    # Collect the input data

    # Initialize the initial, emission, and transition matrices

    # Increment the matrices

    # Add a pseudocount

    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter="\t" for the matrices)
    x,y,z,w,v,u = get_inputs()
    print("x is "+str((x)))
    print("y is "+str((y)))
    print("z is "+str((z)))
    print("w is "+str((w)))
    print("v is "+str((v)))
    print("u is "+str((u)))


    given_index_tags=wordLister(index_tag_input_file)
    # print("index tags are" +str(given_index_tags))
    pi_len = len(given_index_tags)

    #non initialized np arrayfor init matrix
    init_1=np.ones([pi_len])
    # print(init_1)

    #initialize B matrix
    b_mat_1 = np.ones([pi_len,pi_len])

    #initialize emission mat
    emiss = np.ones([pi_len,len(y.keys())])
    x=x[:10000]
    for i in range(0,len(x)):
        # for td_j in td_i:
        #     print(td_j)
        # print(x[i])
        for j in range(0,len(given_index_tags)):
            if x[i][0][1]==given_index_tags[j]:
               init_1[j]= init_1[j]+1
            for k in range(1,len(x[i])):
                for l in range(0, len(given_index_tags)):
                    if x[i][k-1][1]==given_index_tags[j] and x[i][k][1]==given_index_tags[l]:
                        b_mat_1[j][l]=b_mat_1[j][l]+1
            # for m in range(0,len(emiss[z.get(given_index_tags[j])])):
        for n in range(0,len(x[i])):
            emiss[z.get(x[i][n][1]),y.get(x[i][n][0])] = emiss[z.get(x[i][n][1]),y.get(x[i][n][0])] + 1

    for i in range(len(b_mat_1)):
        b_mat_1[i] = b_mat_1[i]/np.sum(b_mat_1[i])

    for i in range(len(emiss)):
        emiss[i] = emiss[i]/np.sum(emiss[i])
    total_init = np.sum(init_1)
    init_1 = init_1/total_init
    # print("final init matrix is "+str(init_1))
    # print("final b matrix is "+str(b_mat_1))
    # print("final emm matrix is "+str(emiss))
    np.savetxt(w, init_1, delimiter='\n')
    # np.savetxt(u, b_mat_1, delimiter='\n', newline=" ")
    np.savetxt(u, b_mat_1, delimiter=" ", newline="\n")
    np.savetxt(v, emiss, delimiter=" ", newline="\n")


#python3 learnhmm.py en_data/train.txt en_data/index_to_word.txt en_data/index_to_tag.txt en_data/hmminit.txt en_data/hmmemit.txt en_data/hmmtrans.txt

##python3 learnhmm.py fr_data/train.txt fr_data/index_to_word.txt fr_data/index_to_tag.txt fr_data/hmminit.txt fr_data/hmmemit.txt fr_data/hmmtrans.txt
#python3 learnhmm.py toy_data/train.txt toy_data/index_to_word.txt toy_data/index_to_tag.txt toy_data/hmminit.txt toy_data/hmmemit.txt toy_data/hmmtrans.txt


