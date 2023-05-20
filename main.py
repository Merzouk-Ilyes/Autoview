import pprint

import Database
from Benefit_Estimation_Model.encoder_model.data_preparation import data_preparation
from Benefit_Estimation_Model.encoder_model.gru import train, evaluate
from Benefit_Estimation_Model.plan_extraction import getCostPlanJson
from Benefit_Estimation_Model.sql2fea import TreeBuilder
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
    train_x, train_y, test_x, test_y =data_preparation(tensor_vectors)
    # print("train_x:",train_x)
    # print("train_y:",train_y)
    # print("test_x:",test_x)
    # print("test_y:",test_y)
    # batch_size = 1
    # train_x = torch.stack(train_x[0])
    # train_y = torch.tensor(train_y[0])
    # print("train_x:",train_x)
    #
    #
    # train_data = TensorDataset(train_x, train_y)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    # # for batch in train_loader:
    # #     print("batch:", batch)
    # #
    # #
    # lr = 0.001
    # gru_model = train(train_loader, lr, model_type="GRU")
    #
    # gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y)

    import torch
    import torch.nn as nn


    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.linear = nn.Linear(hidden_size, output_size)  # Add linear layer

        def forward(self, x):
            hidden = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            output, hidden = self.gru(x, hidden)
            outputs = self.linear(output).squeeze(2)  # Squeeze the last dimension
            return outputs


    # Define the model
    input_size = 8  # Size of each input tensor
    hidden_size = 16  # Number of units in the hidden layer
    output_size = 1  # Size of each output tensor

    model = GRUModel(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert the training data to tensors
    train_x = [torch.stack(sample) for sample in train_x]
    train_y = [torch.stack(sample) for sample in train_y]
    print(torch.stack(train_x).shape)
    print(torch.stack(train_y).shape)
    train_x = torch.stack(train_x)

    # Training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()

        outputs = model(train_x)
        loss = criterion(outputs, torch.stack(train_y).squeeze(2))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "gru_model.pth")
################################################################################@
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    # Load the trained model
    model_path = "gru_model.pth"  # Replace with the path to your trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the test data (if necessary) and convert it to tensors
    test_data = torch.stack([torch.stack(sequence) for sequence in test_x])

    # Perform inference
    with torch.no_grad():
        predicted_outputs = model(test_data)

    # Convert the predicted labels to a numpy array
   # predicted_labels = torch.argmax(predicted_outputs, dim=-1).cpu().numpy()

    print('predicted_outputs:',predicted_outputs[0])

    # Flatten the test_y list and extract the values from the tensors
    ground_truth_labels = [label.item() for label in test_y[0]]

    # Convert the ground truth labels to a numpy array
    ground_truth_labels = np.array(ground_truth_labels)
    print("ground_truth_labels:",ground_truth_labels)

    # Convert the ground truth labels to binary labels based on a threshold
    threshold = 0.5  # Adjust the threshold as per your requirements
    predicted_labels = torch.where(predicted_outputs >= threshold, 1, 0).cpu().numpy().flatten()

    binary_ground_truth_labels = np.where(np.array(ground_truth_labels) >= threshold, 1, 0)

    # Compute accuracy
    accuracy = accuracy_score(binary_ground_truth_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Compute F1-score
    f1 = f1_score(binary_ground_truth_labels, predicted_labels, average="macro")
    print("F1-score:", f1)
    ###############################################################################@
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert the predicted outputs and labels to numpy arrays
    predicted_outputs = predicted_outputs.squeeze().cpu().numpy()
    ground_truth_labels = np.array(ground_truth_labels)

    # Prepare the x-axis values (indices)
    indices = np.arange(len(predicted_outputs))

    # Plot the predicted outputs and the labels
    plt.figure(figsize=(10, 6))
    plt.plot(indices, predicted_outputs, color='red', label='Predicted Outputs')
    plt.plot(indices, ground_truth_labels, color='blue', label='Labels')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predicted Outputs vs Labels')
    plt.legend()
    plt.show()

