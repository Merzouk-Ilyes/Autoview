import os
import re
import json
from moz_sql_parser import parse

from MV_Condidate_Generation.dataset_schema import Dataset_Schema
from MV_Condidate_Generation.extract_predicats import ExtractPredicates
from MV_Condidate_Generation.group_predicates import GroupPredicates


def ParseWorkload(Path_Workload):
    All_Queries = []
    dataset_Schema = Dataset_Schema()
    All_queries_with_id_query = {}
    All_queries_with_their_Tables = {}
    Queries_with_id_and_filename = {}


    j=0
    for filename in os.listdir(Path_Workload):
        # Check if the file is a query file
        if filename.endswith('.sql'):
            # Open the file containing the queries
            # Get_JO_from_file(os.path.join(Path_Workload, filename))
            with open(os.path.join(Path_Workload, filename), 'r') as file:
                # Read the queries
                query = file.read()
                All_Queries.append(query)
                id_query = 'Q' + str(j)
                j += 1
                Queries_with_id_and_filename[id_query] = filename


    Workload_Size = len(All_Queries)
    print('nb queries :', Workload_Size)

    All_Queries_With_Indexes = {}
    All_Queries_Parsed = {}
    All_Join_Predicates = []
    All_selection_Predicates = {}
    List_All_queries_with_their_Tables_and_their_Predicates = {}
    List_All_queries_with_their_Select_Attributes = {}

    index_predicate = 1
    i = 0
    for query in All_Queries:  # for each query in the Workload
        id_query = 'Q' + str(i)
        i += 1
        All_queries_with_their_Tables[id_query] = []
        All_queries_with_id_query[id_query] = []
        All_queries_with_id_query[id_query] = query



        Parsed_Query = parse(query)
        All_Queries_Parsed[id_query] = Parsed_Query
        # print(Parsed_Query )
        # =======================================================
        # data = json.loads(Parsed_Query)
        tables = []
        predicates = {}
        predicates["and"] = []

        def traverse_json(data):
            if isinstance(data, dict):
                for key, value in data.items():

                    if (key.lower() == "from"):
                        tables.append(value[0])
                    if ((key.lower() == "inner join" ) or (key.lower() == "join" ) ):
                        tables.append(value)

                    if (key.lower() == "and"):
                        if (len(predicates['and']) == 0):
                            predicates['and'] = value
                        else:
                            for value in value:
                                predicates['and'].append(value)

                    traverse_json(value)
            elif isinstance(data, list):
                for item in data:
                    traverse_json(item)

        traverse_json(Parsed_Query)

        # print("tables EXPLICIT:",tables)
        # print("predicates EXPLICIT:",predicates)

        # =======================================================
        # Getting the list of tables accesed by the query
        # Tables_list = Parsed_Query['from']

        # Extracting the list of all predicats from the query
        Predicates = []
        # Predicates_Dictionary = Parsed_Query['where']
        Predicates_List = ExtractPredicates(predicates, Predicates)

        All_Join_Predicates, Selection_Predicates_By_Table, All_selection_Predicates, index_predicate = \
            GroupPredicates(Predicates_List, tables, All_Join_Predicates, All_selection_Predicates,
                            index_predicate)
        List_All_queries_with_their_Tables_and_their_Predicates[id_query] = Selection_Predicates_By_Table

        # Get the SELECT attributes
        DicoListSelectAttributes = Parsed_Query['select']
        Select_Attributes_By_Table = {}
       # print("DicoListSelectAttributes:",DicoListSelectAttributes)

       # print(tables)
       #  for table in tables:  # initialization
       #          table_name = re.sub(r'\d+', '', table['name'])
       #
       #          Select_Attributes_By_Table[table_name] = []
       #  print(Select_Attributes_By_Table)
       #  for element in DicoListSelectAttributes:
       #
       #      if isinstance(element, dict):
       #          key = list(dict(element).keys())
       #          value = element[key[0]]
       #          print('THIS IS VALUE:',value)
       #          table_name = list(value.values())[0].split('.')[0]
       #          table_name = re.sub(r'\d+', '', table_name)
       #          if (table_name == "miidx"):
       #              table_name = "mi_idx"
       #          elif (table_name == "a"):
       #              table_name = "an"
       #          Select_Attributes_By_Table[table_name].append(value)
       #
       #  List_All_queries_with_their_Select_Attributes[id_query] = Select_Attributes_By_Table

        for table in tables:
            if dataset_Schema[table["name"]] not in All_queries_with_their_Tables[id_query]:
                All_queries_with_their_Tables[id_query].append(dataset_Schema[table["name"]])

    file.close()

    #print("ALL QUERIES WITH THEIR QUERY ID:",All_queries_with_id_query)

    return All_Queries, \
        All_queries_with_their_Tables, \
        List_All_queries_with_their_Tables_and_their_Predicates, \
        All_Join_Predicates, \
        All_selection_Predicates, \
        Workload_Size, All_Queries_Parsed,All_queries_with_id_query,Queries_with_id_and_filename
