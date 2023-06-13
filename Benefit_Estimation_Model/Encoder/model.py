import sys
from Benefit_Estimation_Model.ImportantConfig import Config
import time
from Benefit_Estimation_Model.Encoder.data_preparation import data_preparation
from Benefit_Estimation_Model.Encoder.testing import testing
from Benefit_Estimation_Model.Encoder.training import training
from Benefit_Estimation_Model.plan_extraction import getCostPlanJson
from Benefit_Estimation_Model.sql2fea import TreeBuilder
def Encoder(connexion,Path_Workload):
    config = Config()
    sys.stdout = open(config.log_file, "w")

    plans, queries = getCostPlanJson(connexion, Path_Workload)

    treeBuilder = TreeBuilder()

    tensor_vectors = []
    for plan in plans:
        tensor_vectors.append(treeBuilder.plan_to_feature_tree(plan))

    train_x, test_x, total_costs_train, total_costs_test = data_preparation(tensor_vectors)
    print("train_x length:", len(train_x))
    print("total_costs_train length:", len(total_costs_train))
    print("test_x length:", len(test_x))
    print("total_costs_test length:", len(total_costs_test))



    # Tensorflow / Keras
    from tensorflow import keras  # for building Neural Networks
    print('Tensorflow/Keras: %s' % keras.__version__)  # print version
    import torch
    # Data manipulation
    import pandas as pd  # for data manipulation
    print('pandas: %s' % pd.__version__)  # print version
    import numpy as np  # for data manipulation
    print('numpy: %s' % np.__version__)  # print version
    # Sklearn
    import sklearn
    print('sklearn: %s' % sklearn.__version__)  # print version
    startime = time.time()
    print("startime:" , startime)

    # ===============================================================================================================
    # =================== DATA PRE PROCESSING ====================================================================================
    #################################################################################################################
    train_x = [torch.stack(sample).flatten() for sample in train_x]
    train_x = [np.array(x) for x in train_x]
    test_x = [torch.stack(sample).flatten() for sample in test_x]
    test_x = [np.array(x) for x in test_x]

    # # Determine the maximum sequence length
    #max_length_train = max(len(seq) for seq in train_x)
    #max_length_test = max(len(seq) for seq in test_x)
    #max_length = max(max_length_train,max_length_test)

    max_length = 33

    # ===============================================================================================================
    # =================== TRAINING ====================================================================================
    ###################################################################################
    #training(train_x,total_costs_train,max_length)


    #===============================================================================================================
    #=================== TESTING ====================================================================================
    #################################################################################################################
    testing(test_x,total_costs_test , max_length)