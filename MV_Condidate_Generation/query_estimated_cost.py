import Database
import numpy as np


def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


# =====================================================================================
def GetQueriesEstimatedCost(List_All_queries, connexion):
    # connexion = Database.connect()

    List_All_queries_with_their_EstimatedCost = {}
    i = 0
    Total_Queries_Estimated_Cost = 0
    for SQLquery in List_All_queries:
        idQuery = 'Q' + str(i)
        i += 1
        query_cost, JsonPlan_Format, Plan_rows, Plan_Width = Database.optimizer_cost(connexion, SQLquery,
                                                                                     True)

        List_Plan_Node_Total_Cost = json_extract(JsonPlan_Format, "Total Cost")

        List_Plan_Node_Type = json_extract(JsonPlan_Format, "Node Type")
        if len(List_Plan_Node_Total_Cost) == len(List_Plan_Node_Type):
            # get the total cost only of the scan, hash, and hash join nodes
            # x = [List_Plan_Node_Total_Cost[i] for i in range(0,len(List_Plan_Node_Total_Cost)) if List_Plan_Node_Type[i] not in ('Aggregate,Sort')]
            x = [List_Plan_Node_Total_Cost[i] for i in range(0, len(List_Plan_Node_Total_Cost)) if
                 List_Plan_Node_Type[i] != "Gather Merge" and
                 List_Plan_Node_Type[i] != 'Aggregate' and List_Plan_Node_Type[i] != 'Sort']

        else:
            print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
            quit()

        Plan_Total_Cost = np.array(List_Plan_Node_Total_Cost).sum()
        # print("Plan_Total_Cost = ",Plan_Total_Cost)
        X_Plan_Total_Cost = np.array(x).sum()
        # print('X_Plan_Total_Cost = ',X_Plan_Total_Cost)

        # Total_Queries_Estimated_Cost = Total_Queries_Estimated_Cost + X_Plan_Total_Cost
        Total_Queries_Estimated_Cost = Total_Queries_Estimated_Cost + Plan_Total_Cost

        List_All_queries_with_their_EstimatedCost[idQuery] = Plan_Total_Cost
        # List_All_queries_with_their_EstimatedCost[idQuery] = X_Plan_Total_Cost
    return List_All_queries_with_their_EstimatedCost, Total_Queries_Estimated_Cost
