# import networkx as nx
# import numpy as np
#
# # Define sample graph
# G = nx.DiGraph()
# G.add_nodes_from([1, 2, 3, 4, 5])
# G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
#
# # Define one-hot encoding dictionary
# one_hot_dict = {'SELECT': 0, 'FROM': 1, 'WHERE': 2, 'AND': 3, 'OR': 4, 'NOT': 5, '(': 6, ')': 7, '*': 8, ',': 9,
#                 '=': 10, '<': 11, '>': 12}
#
# # Define maximum node size
# max_nodes = 10
#
# # Define tensor shape
# tensor_shape = (max_nodes, len(one_hot_dict))
#
# # Initialize tensor with zeros
# tensor = np.zeros(tensor_shape)
#
# # Iterate through nodes of graph
# for i, node in enumerate(G.nodes()):
#     if i >= max_nodes:
#         break
#     # One-hot encode node label
#     label = G.nodes[node]['label']
#     one_hot = np.zeros(len(one_hot_dict))
#     one_hot[one_hot_dict[label]] = 1
#     tensor[i] = one_hot


import pandas as pd




def Encoding(conn):
    # Write SQL query with EXPLAIN ANALYZE
    sql_query = "EXPLAIN ANALYZE SELECT * FROM aka_name; "

    # Load query plan into pandas DataFrame
    df = pd.read_sql_query(sql_query, conn)
    # Extract join type
    df['join_type'] = df['QUERY PLAN'].str.extract(r'Join (.*?) ')

    # Extract index type
    df['index_type'] = df['QUERY PLAN'].str.extract(r'Index Scan using (.*?) on')

    # Extract predicates
    df['predicates'] = df['QUERY PLAN'].str.extract(r'Index Cond: \((.*?)\)')

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['join_type', 'index_type'])

    # Concatenate features into a dense vector
    features = pd.concat([df_encoded['predicates'], df_encoded.drop(['QUERY PLAN', 'predicates'], axis=1)], axis=1)

    print(features)


