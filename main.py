import pprint
import random

import Database
from Benefit_Estimation_Model.encoder_model.data_preparation import data_preparation
from Benefit_Estimation_Model.encoder_model.gru import train, evaluate
from Benefit_Estimation_Model.plan_extraction import getCostPlanJson
from Benefit_Estimation_Model.sql2fea import TreeBuilder, ValueExtractor
from MV_Condidate_Generation import join_graph, computing_cost
from MV_Condidate_Generation.get_frequency import getFrequency
from MV_Condidate_Generation.merge_plan import MergePlan
from MV_Condidate_Generation.query_estimated_cost import GetQueriesEstimatedCost
from MV_Condidate_Generation.workload import ParseWorkload
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from MV_Condidate_Generation.dataset_schema import Dataset_Schema, Dataset_Schema2
from MV_Condidate_Generation.selecting_mv_condidates import Get_Views_Info_From_MVPP
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("START AUTOVIEW")
    startime = time.time()
    connexion = Database.connect()
    Path_Workload = '/Users/ilyes/Downloads/JOB_PROD'

    # # PARSING==================================================================
    # List_All_queries, List_All_queries_with_their_Tables, List_All_queries_with_their_Tables_and_their_Predicates, \
    #     All_Join_Predicates, All_selection_Predicates, List_All_queries_with_their_Select_Attributes, Workload_Size, \
    #     List_All_queries_with_their_parsed_format = ParseWorkload(Path_Workload)
    #
    # # DBMS COST ESTIMATION================================================================
    # Rewritten_List_All_queries_with_their_EstimatedCost, Rewritten_Total_Queries_Estimated_Cost = GetQueriesEstimatedCost(
    #     List_All_queries,
    #     connexion)
    #
    # # for each query generate two tree : one with selection predicat and on called light whiout selection predicates
    # # according to the optimal join order extracted from postgresql
    # Dic_Query_With_Query_Join_tree_graph = {}
    # Dic_Query_With_Query_Join_tree_graph_Light = {}
    # Dic_Query_by_Oreder_In_The_Workload = {}
    #
    # startime_generate_query_trees = time.time()
    # i = 0
    #
    # ListQueries = list(List_All_queries_with_their_Tables.keys())
    # for query in List_All_queries_with_their_Tables:
    #     print(query)
    #     # print(List_All_queries_with_their_Join_Order[query])
    #     Dico_query_tables_and_predicates = dict(List_All_queries_with_their_Tables_and_their_Predicates[query])
    #     Dico_query_tables_and_selectAttributes = dict(List_All_queries_with_their_Select_Attributes[query])
    #     query_join_order = List_All_queries_with_their_Tables[query]
    #
    #     # print("line52" ,Dico_query_tables_and_selectAttributes)
    #
    #     Query_Join_tree_graph = join_graph.Create_Graph(query,
    #                                                     Dico_query_tables_and_predicates,
    #                                                     query_join_order)
    #     Path_to_MVPP = '/Users/ilyes/Downloads/gml/' + str(i) + '.gml'
    #     nx.write_gml(Query_Join_tree_graph, Path_to_MVPP)
    #
    #     Dic_Query_With_Query_Join_tree_graph[query] = Query_Join_tree_graph
    #     Dic_Query_by_Oreder_In_The_Workload[query] = i
    #     i += 1
    # endtime_generate_query_trees = time.time() - startime_generate_query_trees
    #
    # # GRAPH VISUALISATION
    # # nx.draw(Query_Join_tree_graph,with_labels=True)
    # # plt.draw()
    # # plt.show()
    #
    # # THE MERGING PHASE
    # Dic_Id_With_MVPP_graph = {}
    # # the folowwing loop performs the rotation of query graph for merging
    # t1 = OrderedDict(sorted(Dic_Query_by_Oreder_In_The_Workload.items(), key=lambda x: x[1]))
    # lst = list(Dic_Query_With_Query_Join_tree_graph.keys())
    # i = 0
    #
    # Queries_Order_For_Merging = list(t1.keys())
    # MVPP = MergePlan(Queries_Order_For_Merging, Dic_Query_With_Query_Join_tree_graph)
    #
    # Path_to_MVPP = '/Users/ilyes/Downloads/gml/MVPP' + str(i) + '.gml'
    # nx.write_gml(MVPP, Path_to_MVPP)
    #
    # # MVPP GRAPH VISUALISATION
    # # nx.draw(MVPP,with_labels=True)
    # # plt.draw()
    # # plt.show()
    #
    # # COST COMPUTING
    # MVPP_With_Selection_0_With_Cost, List_Nodes_With_SQL_Script = computing_cost.ComputeCost(
    #     MVPP,
    #     All_Join_Predicates,
    #     All_selection_Predicates,
    #     Dico_query_tables_and_selectAttributes,
    #     Dataset_Schema,
    #     Dataset_Schema2,
    #     connexion)
    #
    # frequency = getFrequency(MVPP, Dic_Query_With_Query_Join_tree_graph)
    #
    # views_with_cost = Get_Views_Info_From_MVPP(MVPP_With_Selection_0_With_Cost,
    #                                            frequency,
    #                                            ListQueries)
    #
    # MV_Condidates = {}
    # for v in views_with_cost:
    #     if v in frequency:
    #         views_with_cost[v].append({"Frequency":frequency[v] } )
    #         views_with_cost[v].append({"Benefit": frequency[v] * views_with_cost[v][0]["Total cost"] })
    # print("views_with_cost:", views_with_cost)
    #
    # views_with_cost = OrderedDict(sorted(views_with_cost.items(), key=lambda x: x[1][0]["Total cost"] * x[1][3]["Frequency"] , reverse=True))
    #
    # for v in views_with_cost.items():
    #
    #     if (v[1][0] != 0):
    #         MV_Condidates[v[0]] = v[1]
    #
    # MV_Condidates = dict(list(MV_Condidates.items())[:5])
    #
    # for v in MV_Condidates:
    #     MV_Condidates[v].append(List_Nodes_With_SQL_Script[v])
    #
    # print("THE BEST MV Condidates (Sorted from best to worst) :======================================================================")
    # for v in MV_Condidates:
    #     print(v , " : ", MV_Condidates[v])

    #================BENEFIT ESTIMATION PHASE===============================================================

    #================ENCODING===============================================================================
    import sys
    from Benefit_Estimation_Model.ImportantConfig import Config
    config = Config()
    sys.stdout = open(config.log_file, "w")

    plans,queries = getCostPlanJson(connexion,Path_Workload)

    treeBuilder = TreeBuilder()

    tensor_vectors = []
    for plan in plans:
        tensor_vectors.append(treeBuilder.plan_to_feature_tree(plan))


    #print("tensor_vectors size:", len(tensor_vectors))
    train_x, train_y, test_x, test_y,total_costs_train,total_costs_test =data_preparation(tensor_vectors)
    print("train_x length:",len(train_x))
    print("train_y length:",len(train_y))
    print("total_costs_train length:",len(total_costs_train))
    print("total_costs_test length:",len(total_costs_test))
    print("test_x length:",len(test_x))
    print("test_y length:",len(test_y))



    # import torch
    # import torch.nn as nn
    #
    #
    # class GRUModel(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size,num_layers,num_directions):
    #         super(GRUModel, self).__init__()
    #         self.hidden_size = hidden_size
    #         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    #         self.fc = nn.Linear(hidden_size, output_size)
    #         self.linear = nn.Linear(hidden_size, output_size)  # Add linear layer
    #         self.num_layers = num_layers
    #         self.num_directions = num_directions
    #     # def forward(self, x):
    #     #     hidden = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
    #     #
    #     #     print("x:",x)
    #     #     output, hidden = self.gru(x, hidden[0])
    #     #     outputs = self.linear(output).squeeze(2)  # Squeeze the last dimension
    #     #     return outputs
    #     def forward(self, x):
    #         #print(x)
    #         batch_size = x.size(0)
    #         hidden = torch.zeros(num_layers * num_directions,1 , self.hidden_size).to(x.device)
    #         #print(hidden.shape)
    #         output, hidden = self.gru(x, hidden[0])
    #         outputs = self.linear(output).squeeze(1)  # Squeeze the last dimension
    #         return outputs
    #
    # print("encoded tensor:" , train_x[0])
    #
    # # Convert the training data to tensors
    # train_x = [torch.stack(sample).flatten() for sample in train_x]
    # train_x = torch.stack(train_x)
    # print("train_x shape:", train_x.shape)
    #
    # total_costs_train = torch.stack(total_costs_train)
    #
    # # Define the model
    # input_size = train_x.shape[1]  # Size of each input tensor
    # hidden_size = 16  # Number of units in the hidden layer
    # output_size = total_costs_train.shape[1]   # Size of each output tensor
    # num_directions = 1
    # num_layers = 2  # Update the number of layers to 3
    #
    # model = GRUModel(input_size, hidden_size, output_size,num_layers ,num_directions)
    #
    # # Define the loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # # train_y = [torch.stack(sample) for sample in train_y]
    # # print(torch.stack(train_y).shape)
    # # Training loop
    # num_epochs = 100
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #
    #     outputs = model(train_x)
    #     # loss = criterion(outputs, torch.stack(train_y).squeeze(2))
    #     loss = criterion(outputs, torch.squeeze(total_costs_train))
    #
    #     # Backward and optimize
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     # Print the loss every 10 epochs
    #     if (epoch + 1) % 10 == 0:
    #         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    #
    #     # Save the trained model
    # torch.save(model.state_dict(), "gru_model.pth")
    # ################################################################################@
    import numpy as np
    # from sklearn.metrics import accuracy_score, f1_score
    #
    # # Load the trained model
    # model_path = "gru_model.pth"  # Replace with the path to your trained model
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    #
    # test_x = [torch.stack(sample).flatten() for sample in test_x]
    # test_data = torch.stack(test_x)
    # # Preprocess the test data (if necessary) and convert it to tensors
    # # test_data = torch.stack([torch.stack(sequence) for sequence in test_x])
    #
    #
    # # Perform inference
    # with torch.no_grad():
    #     predicted_outputs = model(test_data)
    #
    value_extractor = ValueExtractor()
    #
    # for i in range(0,len(predicted_outputs)):
    #     predicted_outputs[i] = value_extractor.decode(predicted_outputs[i])
    #
    # # Convert the predicted labels to a numpy array
    # # predicted_labels = torch.argmax(predicted_outputs, dim=-1).cpu().numpy()
    # print('predicted_outputs:', predicted_outputs)
    #
    ground_truth_labels = []
    # # Flatten the test_y list and extract the values from the tensors
    for item in total_costs_test:
      ground_truth_labels.append([ label.item() for label in  item])
    #
    # # Convert the ground truth labels to a numpy array
    ground_truth_labels = np.array(ground_truth_labels)
    # for i in range(0,len(ground_truth_labels)):
    #     ground_truth_labels[i] = value_extractor.decode(ground_truth_labels[i])
    # print("ground_truth_labels:", ground_truth_labels)
    #
    # # Convert the ground truth labels to binary labels based on a threshold
    # threshold = 0.9  # Adjust the threshold as per your requirements
    # predicted_labels = torch.where(predicted_outputs >= threshold, 1, 0).cpu().numpy().flatten()
    #
    # binary_ground_truth_labels = np.where(np.array(ground_truth_labels) >= threshold, 1, 0)
    #
    # # Compute accuracy
    # accuracy = accuracy_score(binary_ground_truth_labels, predicted_labels)
    # print("Accuracy:", accuracy)
    # #
    # # # Compute F1-score
    # f1 = f1_score(binary_ground_truth_labels, predicted_labels, average="macro")
    # print("F1-score:", f1)
    # ###############################################################################@
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # Convert the predicted outputs and labels to numpy arrays
    # predicted_outputs = predicted_outputs.squeeze().cpu().numpy()
    # ground_truth_labels = np.array(ground_truth_labels)
    #
    # # Prepare the x-axis values (indices)
    # indices = np.arange(len(predicted_outputs))
    #
    # # Plot the predicted outputs and the labels
    # plt.figure(figsize=(10, 6))
    # plt.plot(indices, predicted_outputs, color='red', label='Predicted Outputs')
    # plt.plot(indices, ground_truth_labels, color='blue', label='Labels')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.title('Predicted Outputs vs Labels')
    # plt.legend()
    # plt.show()
    #
    #







    ####################################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #######TESTING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #Tensorflow / Keras
    from tensorflow import keras  # for building Neural Networks
    import tensorflow as tf
    print('Tensorflow/Keras: %s' % keras.__version__)  # print version
    from keras.models import Sequential  # for creating a linear stack of layers for our Neural Network
    from keras import Input  # for instantiating a keras tensor
    from keras.layers import Bidirectional, GRU, RepeatVector, Dense, \
        TimeDistributed  # for creating layers inside the Neural Network

    # Data manipulation
    import pandas as pd  # for data manipulation

    print('pandas: %s' % pd.__version__)  # print version
    import numpy as np  # for data manipulation

    print('numpy: %s' % np.__version__)  # print version

    # Sklearn
    import sklearn

    print('sklearn: %s' % sklearn.__version__)  # print version
    from sklearn.preprocessing import MinMaxScaler  # for feature scaling

    # Visualization
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go

    print('plotly: %s' % plotly.__version__)  # print version

    ##### Step 1 - Specify parameters
    timestep = 18
    scaler = MinMaxScaler(feature_range=(-1, 1))

    train_x = [torch.stack(sample).flatten() for sample in train_x]
    test_x = [torch.stack(sample).flatten() for sample in test_x]

    #total_costs_train = torch.stack(total_costs_train)
    #total_costs_test = torch.stack(total_costs_test)

    # # Convert tensors to tuples
    # array_of_tuples = [tuple(tensor.tolist()) for tensor in train_x[2]]
    #
    # # Keep only unique tuples
    # unique_tuples = list(set(array_of_tuples))
    #
    # # Convert back to tensors
    # unique_tensors = [torch.tensor(tuple_) for tuple_ in unique_tuples]
    # print('train_x:',train_x )
    # print('total_costs_train:',total_costs_train)
    # print('test_x:',test_x)
    # print('total_costs_test:',total_costs_test)
    # Use fit to train the scaler on the training data only, actual scaling will be done inside reshaping function
    #scaler.fit(np.array(train_x).reshape(-1, 1))

    # Convert the list of NumPy arrays to a single NumPy array
    # for x in train_x:
    #     train_x_numpy.append(tf.stack(x).numpy())
    # train_x_numpy = np.array(train_x_numpy)

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    train_x = [np.array(x) for x in train_x ]
    test_x = [np.array(x) for x in test_x ]
    # Determine the maximum sequence length


    max_length = max(len(seq) for seq in train_x)
    max_length2 = max(len(seq) for seq in test_x)

    # Pad the sequences
    train_x = pad_sequences(train_x, maxlen=max_length, padding='post', truncating='post',dtype='float32')
    test_x = pad_sequences(test_x, maxlen=max_length, padding='post', truncating='post',dtype='float32')
    train_x_tensors = []
    for i in range(0 ,len(train_x)):
        train_x_tensors.append(tf.convert_to_tensor(train_x[i]))
    train_x_tensors = tf.convert_to_tensor(train_x_tensors)
    #train_x_tensors = np.array(train_x_tensors)
    print('train_x shape:', train_x_tensors)
    test_x_tensors = []
    for i in range(0, len(test_x)):
        test_x_tensors.append(tf.convert_to_tensor(test_x[i]))
    test_x_tensors = tf.convert_to_tensor(test_x_tensors)
    # Reshape the input data
    # train_x_reshaped = train_x_numpy.reshape(train_x_numpy.shape[0], train_x_numpy.shape[1])
    # Specify the input shape based on a single input sample
    input_shape = ( 1,train_x.shape[1])
    ##### Step 3 - Specify the structure of a Neural Network
    model = Sequential(name="GRU-Model")  # Model
    model.add(Input(shape=input_shape,
                    name='Input-Layer'))  # Input Layer - need to speicfy the shape of inputs
    model.add(Bidirectional(GRU(units=32, activation='tanh', recurrent_activation='sigmoid', stateful=False),
                            name='Hidden-GRU-Encoder-Layer'))  # Encoder Layer
    model.add(RepeatVector(train_x.shape[1], name='Repeat-Vector-Layer'))  # Repeat Vector
    model.add(Bidirectional(
        GRU(units=32, activation='tanh', recurrent_activation='sigmoid', stateful=False, return_sequences=True),
        name='Hidden-GRU-Decoder-Layer'))  # Decoder Layer
    model.add(TimeDistributed(Dense(units=1, activation='linear'), name='Output-Layer'))  # Output Layer, Linear(x) = x

    ##### Step 4 - Compile the model
    model.compile(optimizer='adam',  # default='rmsprop', an algorithm to be used in backpropagation
                  loss='mean_squared_error',
                  # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['MeanSquaredError', 'MeanAbsoluteError'],
                  # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
                  loss_weights=None,
                  # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
                  weighted_metrics=None,
                  # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
                  run_eagerly=None,
                  # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                  steps_per_execution=None
                  # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  )

    # Assuming train_x is a list of Torch tensors
    total_costs_train_tensors = []
    for i in range(0, len(total_costs_train)):
        total_costs_train_tensors.append(tf.convert_to_tensor(total_costs_train[i]))
    total_costs_train_tensors = tf.convert_to_tensor(total_costs_train_tensors)
    print(total_costs_train_tensors)
    train_x_tensors = np.reshape(train_x_tensors, (train_x_tensors.shape[0], 1, train_x_tensors.shape[1]))
    train_x_tensors = tf.convert_to_tensor(train_x_tensors, dtype=tf.float32)

    ##### Step 5 - Fit the model on the dataset
    history = model.fit(train_x_tensors,  # input data
                        total_costs_train_tensors,  # target data
                        batch_size=1,
                        # Number of samples per gradient update. If unspecified, batch_size will default to 32.
                        epochs=50,
                        # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
                        verbose=1,
                        # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
                        callbacks=None,
                        # default=None, list of callbacks to apply during training. See tf.keras.callbacks

                        # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
                        # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
                        shuffle=True,
                        # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
                        class_weight=None,
                        # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                        sample_weight=None,
                        # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
                        initial_epoch=0,
                        # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
                       # steps_per_epoch=None,
                        # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                        validation_steps=None,
                        # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
                        #validation_batch_size=None,
                        # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
                        validation_freq=10,
                        # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
                        max_queue_size=10,
                        # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                        workers=1,
                        # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
                        use_multiprocessing=True,
                        # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
                        )

    ##### Step 6 - Use model to make predictions
    # Predict results on training data
    # pred_train = model.predict(X_train)

    test_x_tensors = np.reshape(test_x_tensors, (test_x_tensors.shape[0], 1, test_x_tensors.shape[1]))
    test_x_tensors = tf.convert_to_tensor(test_x_tensors, dtype=tf.float32)
    # Predict results on test data
    pred_test = model.predict(test_x_tensors)
    predictions = []
    for i in range(0,len(pred_test)):
        predictions.append(pred_test[i][0])

    print('Prediction:', predictions)

    # Prepare the x-axis values (indices)
    indices = np.arange(len(predictions))

    # Plot the predicted outputs and the labels
    plt.figure(figsize=(10, 6))
    plt.plot(indices, predictions, color='red', label='Predicted Outputs')
    plt.plot(indices, ground_truth_labels, color='blue', label='Labels')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predicted Outputs vs Labels')
    plt.legend()
    plt.show()
    ##### Step 7 - Print Performance Summary
    print("")
    print('-------------------- Model Summary --------------------')
    model.summary()  # print model summary
    print("")
    print('-------------------- Weights and Biases --------------------')
    print("Too many parameters to print but you can use the code provided if needed")
    print("")
    # for layer in model.layers:
    #    print(layer.name)
    #    for item in layer.get_weights():
    #        print("  ", item)
    # print("")

    # Print the last value in the evaluation metrics contained within history file
    print('-------------------- Evaluation on Training Data --------------------')
    for item in history.history:
        print("Final", item, ":", history.history[item][-1])
    print("")

    # Evaluate the model on the test data using "evaluate"
    print('-------------------- Evaluation on Test Data --------------------')
    # test_x_numpy = test_x.numpy().astype(np.float32)
    results = model.evaluate(test_x_tensors,
                             ground_truth_labels)
    print("Evaluation===========================================")
    print("Test Loss:", results[1])
    print("Test Accuracy:", results[2])
    print("")