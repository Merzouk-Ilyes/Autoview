import os
import re

from moz_sql_parser import parse

from MV_Condidate_Generation.dataset_schema import Dataset_Schema
from MV_Condidate_Generation.extract_predicats import ExtractPredicates
from MV_Condidate_Generation.group_predicates import GroupPredicates


def ParseWorkload(Path_Workload):
    All_Queries = []
    dataset_Schema = Dataset_Schema()

    for filename in os.listdir(Path_Workload):
        # Check if the file is a query file
        if filename.endswith('.sql'):
            # Open the file containing the queries
            # Get_JO_from_file(os.path.join(Path_Workload, filename))
            with open(os.path.join(Path_Workload, filename), 'r') as file:
                # Read the queries
                query = file.read()
                All_Queries.append(query)

    Workload_Size = len(All_Queries)
    print('nb queries :', Workload_Size)

    All_Queries_With_Indexes = {}
    All_Queries_Parsed = {}
    All_Join_Predicates = []
    All_selection_Predicates = {}
    List_All_queries_with_their_Tables_and_their_Predicates = {}
    List_All_queries_with_their_Select_Attributes = {}
    All_queries_with_their_Tables = {}

    index_predicate = 1
    i = 0
    for query in All_Queries:  # for each query in the Workload
        id_query = 'Q' + str(i)
        i += 1
        All_queries_with_their_Tables[id_query] = []
        Parsed_Query = parse(query)
        All_Queries_Parsed[id_query] = Parsed_Query
        # Getting the list of tables accesed by the query
        Tables_list = Parsed_Query['from']

        # Extracting the list of all predicats from the query
        Predicates = []
        Predicates_Dictionary = Parsed_Query['where']
        Predicates_List = ExtractPredicates(Predicates_Dictionary, Predicates)

        All_Join_Predicates, Selection_Predicates_By_Table, All_selection_Predicates, index_predicate = \
            GroupPredicates(Predicates_List, Tables_list, All_Join_Predicates, All_selection_Predicates,
                            index_predicate)
        List_All_queries_with_their_Tables_and_their_Predicates[id_query] = Selection_Predicates_By_Table

        # Get the SELECT attributes
        DicoListSelectAttributes = Parsed_Query['select']
        Select_Attributes_By_Table = {}
        for table in Tables_list:  # initialization
            Select_Attributes_By_Table[table['name']] = []
        for element in DicoListSelectAttributes:

            if isinstance(element, dict):
                key = list(dict(element).keys())
                value = element[key[0]]
                table_name = list(value.values())[0].split('.')[0]
                table_name = re.sub(r'\d+', '', table_name)
                if (table_name == "miidx"):
                    table_name = "mi_idx"
                elif (table_name == "a"):
                    table_name = "an"
                Select_Attributes_By_Table[table_name].append(value)

        List_All_queries_with_their_Select_Attributes[id_query] = Select_Attributes_By_Table

        for table in Tables_list:
            if dataset_Schema[table["name"]] not in All_queries_with_their_Tables[id_query]:
                All_queries_with_their_Tables[id_query].append(dataset_Schema[table["name"]])

    file.close()

    return All_Queries, \
        All_queries_with_their_Tables, \
        List_All_queries_with_their_Tables_and_their_Predicates, \
        All_Join_Predicates, \
        All_selection_Predicates, \
        List_All_queries_with_their_Select_Attributes, \
        Workload_Size, All_Queries_Parsed
