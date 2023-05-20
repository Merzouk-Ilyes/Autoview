
import numpy as np

def data_preparation(data):


    inputs = []
    labels = []
    sequences = {}
    query_nodes_without_labels = []
    query_nodes_labels = []

    #print("data:",data)
    for i in  range(0,len(data)):
        sequence = []
        query_idx = "Q" + str(i)
        sequences[query_idx] = vector_tree_to_array(data[i],sequence)


    #print("data after (sequences):", sequences)

    for query_nodes in sequences.values():
        #print(query_nodes)
        for i in range(0, len(query_nodes)):
            inputs.append(query_nodes[i][0][:8])
            labels.append(query_nodes[i][0][8:])

        query_nodes_without_labels.append(inputs)
        query_nodes_labels.append(labels)
    #print("query_nodes_without_labels:",query_nodes_without_labels)
    #print("query_nodes_labels:",query_nodes_labels)
    # inputs = inputs.reshape(-1, lookback)
    # labels = labels.reshape(-1, 1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.5 * len(data))
    print("test_portion:",test_portion)

    # train_x = np.zeros(len(data) -  test_portion,dtype=object)
    # train_y = np.zeros(len(data) -  test_portion,dtype=object)


    train_x = []
    train_y = []

    # train_x = np.concatenate(( query_nodes_without_labels[:-test_portion]))
    # train_y = np.concatenate((query_nodes_labels[:-test_portion]))
    train_x = query_nodes_without_labels[:-test_portion]
    train_y = query_nodes_labels[:-test_portion]
    #     # print("train_x:", train_x)
    #     # print("train_y:", train_y)


    # train_x = np.concatenate( query_nodes_without_labels[:-test_portion])
    # train_y = np.concatenate( query_nodes_labels[:-test_portion])
    test_x = (query_nodes_without_labels[-test_portion:])
    test_y = (query_nodes_labels[-test_portion:])

    return train_x,train_y ,test_x,test_y



def vector_tree_to_array(v_tree,sequence):
    for i in range(0,len(v_tree)):
        #print(v_tree[i])
        if len(v_tree[i]) > 1:
            vector_tree_to_array(v_tree[i],sequence)
        else:
            sequence.append(v_tree[i])
    return sequence


