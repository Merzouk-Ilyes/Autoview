import re
from moz_sql_parser import parse
import sqlparse


def Rewrite_query(queries_with_corresponding_views, All_queries_with_id_query, mv_candidates, views_to_be_materialized, Queries_with_id_and_filename):
    rewritten_queries = {}
    id_to_alias = {'21': 'an', '01': 'at', '02': 'ci', '03': 'chn', '04': 'cct', '05': 'cn', '06': 'ct', '07': 'cc',
                   '08': 'it',
                   '09': 'k', '10': 'kt', '11': 'lt', '12': 'mc', '13': 'mi', '14': 'mi_idx', '15': 'mk', '16': 'ml',
                   '17': 'n',
                   '18': 'pi', '19': 'rt', '20': 't'}

    for query in queries_with_corresponding_views:
        # Find the positions of "SELECT", "FROM", and the first "ON" keywords

        if (queries_with_corresponding_views[query] in views_to_be_materialized):
            rewritten_queries[query] = []
            rewritten_queries[query].append(All_queries_with_id_query[query])
            join = "inner join"

            views_tables = queries_with_corresponding_views[query].split('_J_')
            views_tables = [id_to_alias[id] for id in views_tables]
            # print("====================================================================================")
            # print("VIEWS_TABLES:",query , queries_with_corresponding_views[query] , views_tables)
            print('FILE NAME=>',Queries_with_id_and_filename[query])

            select_index = All_queries_with_id_query[query].lower().find("select")
            # from_index = All_queries_with_id_query[query].lower().find("from")
            from_index = find_word_index(All_queries_with_id_query[query].lower(), "from")
            indexes = []
            start_index = 0

            # while True:
            #     index = All_queries_with_id_query[query].find("inner join", start_index)
            #     if index == -1:
            #         break
            #     indexes.append(index)
            #     start_index = index + 1
            pattern = r"\b" + re.escape(join) + r"\b"
            indexes = [match.start() for match in re.finditer(pattern, All_queries_with_id_query[query],re.IGNORECASE)]

            # print("indexes:",indexes)

            # Extract the three parts of the query
            part1 = All_queries_with_id_query[query][:from_index].strip()
            # part2 = All_queries_with_id_query[query][from_index:indexes[1]].strip()

          #  print("part1:" , part1)

            part3 = All_queries_with_id_query[query][indexes[0]:].strip()

            # print("Part 1:", part1)
            #  print("Part 2:", part2)
           # print("Part 3:", part3)
            on_index = part3.lower().find("on")
            and_index = part3.lower().find("and")

            pattern = r"\b" + re.escape(join) + r"\b"
            all_inner_join_indexes_part3 = [match.start() for match in re.finditer(pattern, part3,re.IGNORECASE)]

            # print("on_index",on_index)
            # print("and_index",and_index)
            # print("predicate=>", part3[on_index+2:and_index])

            as_index = part3.lower().find("as")
          #  print("PART 3 INNER INDEXES:",all_inner_join_indexes_part3)
            view_query_relation_table = part3[as_index + 2:on_index]
            view_query_predicate = part3[on_index + 2:all_inner_join_indexes_part3[1]]
           # print("LINE 58",view_query_predicate)

            view_query_predicate = view_query_predicate.split('AND')

            for i in range(0, len(view_query_predicate)):
                pred = view_query_predicate[i].split("=")
                # if (view_query_relation_table.strip() not in view_query_predicate[i]):
                #     view_query_predicate[i] = " v_" + queries_with_corresponding_views[query].lower() + '.' + \
                #                               view_query_predicate[i].split('.')[0].strip() + '_' + \
                #                               view_query_predicate[i].split('.')[1].strip() + ' '
                if ((len(pred) == 2)):
                    #  print("LINE 96:", ('.' in pred[0]) & (pred[0].split('.')[0].strip() in views_tables),('.' in pred[1]) & (pred[1].split('.')[0].strip() in views_tables), pred)

                    pred1 = re.sub(r'\d+', '', pred[1].split('.')[0].strip())
                    pred0 = re.sub(r'\d+', '', pred[0].split('.')[0].strip())
                    if (('.' in pred[1]) & (pred1 in views_tables)):
                        pred[1] = " v_" + queries_with_corresponding_views[query].lower() + '.' + pred1 + '_' + pred[1].split('.')[1].strip() + ' '
                    if (('.' in pred[0]) & (pred0 in views_tables)):
                        pred[0] = " v_" + queries_with_corresponding_views[query].lower() + '.' + pred0 + '_' + pred[0].split('.')[1].strip() + ' '
                view_query_predicate[i] = '='.join(pred)
            predicate_rewritten = "AND".join(view_query_predicate)


            # predicate_rewritten = "=".join(view_query_predicate)
            part3_rewritten = part3[:on_index + 2] + predicate_rewritten + part3[all_inner_join_indexes_part3[1]:]
            rewritten_query = part1 + " FROM v_" + queries_with_corresponding_views[
                query].lower() + ' ' + part3_rewritten
           # print("line97", part3_rewritten)


            # Getting all inner join indexes
            pattern = r"\b" + re.escape(join) + r"\b"
            all_inner_join_indexes = [match.start() for match in re.finditer(pattern, rewritten_query,re.IGNORECASE)]

            # Getting all ON indexes
            pattern = r"\b" + re.escape('on ') + r"\b"
            all_on_indexes = [match.start() for match in re.finditer(pattern, rewritten_query,re.IGNORECASE)]



            all_on_indexes_without_last = all_on_indexes[:-1]
            all_inner_join_indexes_without_first = all_inner_join_indexes[1:]


            pre_rewritten_query = rewritten_query
            for i in range(0, len(all_inner_join_indexes_without_first)):
                # Getting all inner join indexes
                pattern = r"\b" + re.escape(join) + r"\b"
                all_inner_join_indexes = [match.start() for match in re.finditer(pattern, pre_rewritten_query,re.IGNORECASE)]


                # Getting all ON indexes
                pattern = r"\b" + re.escape('on ') + r"\b"
                all_on_indexes = [match.start() for match in re.finditer(pattern, pre_rewritten_query,re.IGNORECASE)]

                all_on_indexes = all_on_indexes[:-1]
                all_inner_join_indexes = all_inner_join_indexes[1:]

                first_part_of_rewritten_query = pre_rewritten_query[:all_on_indexes[i] + 2]
                last_part_of_rewritten_query = pre_rewritten_query[all_inner_join_indexes[i]:]
                predicates_concatenated = pre_rewritten_query[
                                          all_on_indexes[i] + 2: all_inner_join_indexes[i]]

                predicates = predicates_concatenated.split('AND')
                if "And" in predicates[0]:
                    predicates = predicates[0].split('And')
                print("LINE 134:",predicates)
               # print("=========>>>>>>>>",rewritten_query[ all_inner_join_indexes_without_first[i]:])
               # print("FULL QUERY:",rewritten_query,'<><><><><>', 'PREDICATS CONCATENATED:',predicates_concatenated)
                # print("VIEWS TABLES:",views_tables)
               #print("line 87:",pre_rewritten_query)
                for i in range(0, len(predicates)):
                    pred = predicates[i].split('=')
                    if ((len(pred) == 2)):
                      #  print("LINE 96:", ('.' in pred[0]) & (pred[0].split('.')[0].strip() in views_tables),('.' in pred[1]) & (pred[1].split('.')[0].strip() in views_tables), pred)
                        pred1 = re.sub(r'\d+', '', pred[1].split('.')[0].strip())
                        pred0 = re.sub(r'\d+', '', pred[0].split('.')[0].strip())
                        if (('.' in pred[1]) & (pred1 in views_tables)):
                                pred[1] = " v_" + queries_with_corresponding_views[query].lower()+'.' + pred1 + '_' + pred[1].split('.')[1].strip() + ' '
                        if (('.' in pred[0]) & (pred0 in views_tables)):
                                pred[0] = " v_" + queries_with_corresponding_views[query].lower()+'.' + pred0 + '_' + pred[0].split('.')[1].strip() + ' '
                    else:
                        pred0 = re.sub(r'\d+', '', pred[0].split('.')[0].strip())

                        if (('.' in pred[0]) & (pred0 in views_tables)):
                            pred[0] = " v_" + queries_with_corresponding_views[query].lower() + '.' + pred0 + '_' + \
                                      pred[0].split('.')[1].strip() + ' '


                    predicates[i] = '='.join(pred)
                predicates_concatenated = "AND".join(predicates)

                pre_rewritten_query = first_part_of_rewritten_query + predicates_concatenated  + last_part_of_rewritten_query
                #print('LINE 100:', pre_rewritten_query)

            rewritten_query = pre_rewritten_query

            # Getting all ON indexes
            pattern = r"\b" + re.escape('on ') + r"\b"
            all_on_indexes = [match.start() for match in re.finditer(pattern, rewritten_query,re.IGNORECASE)]

            first_part_of_rewritten_query = rewritten_query[:all_on_indexes[-1] +2 ]
            predicates_concatenated = rewritten_query[all_on_indexes[-1] +2 :]

            predicates = predicates_concatenated.split('AND')
          #  print("line 181:" , predicates_concatenated)


            for i in range(0, len(predicates)):
                pred = predicates[i].split('=')

                if ((len(pred) == 2)):
                    pred1 = re.sub(r'\d+', '', pred[1].split('.')[0].strip())
                    pred0 = re.sub(r'\d+', '', pred[0].split('.')[0].strip())
                    if (('.' in pred[1]) & (pred1 in views_tables)):
                        pred[1] = " v_" + queries_with_corresponding_views[query].lower() + '.' + pred1 + '_' + pred[1].split('.')[1].strip() + ' '
                    if (('.' in pred[0]) & (pred0 in views_tables)):
                        pred[0] = " v_" + queries_with_corresponding_views[query].lower() + '.' + pred0 + '_' + pred[0].split('.')[1].strip() + ' '
                predicates[i] = '='.join(pred)

            predicates_concatenated = " AND".join(predicates)
            rewritten_query = first_part_of_rewritten_query + predicates_concatenated + ';'

            #Rewritting projections
            Parsed_Query = parse(rewritten_query)
            # projections = Parsed_Query['select']

            if(isinstance(Parsed_Query['select'] , list)):
                projections = Parsed_Query['select']
            else:
                projections = [Parsed_Query['select']]

            from_index = find_word_index(rewritten_query.lower(), "from")

            rewritten_query_without_projections =  rewritten_query[from_index:].strip()
            for i in range(0,len(projections)):
                p_table = projections[i]['value']['min'].split('.')[0]
                p_table = re.sub(r'\d+', '', p_table)

                if (p_table.strip() in views_tables):
                    projections[i]['value']['min'] = "v_" + queries_with_corresponding_views[query].lower()+ "." + p_table + "_" +\
                    projections[i]['value']['min'].split('.')[1]

            rewritten_projection = []
            for p in projections:
                rewritten_projection.append("min(" +  p['value']["min"] + ")" + " AS " + p["name"] + " ")
            rewritten_projection = ",".join(rewritten_projection)
            rewritten_projection = "SELECT " +  rewritten_projection

            rewritten_query = rewritten_projection + rewritten_query_without_projections


            rewritten_queries[query].append(rewritten_query)

            #print("REWRITTEN QUERY======>>:", rewritten_queries)



    return rewritten_queries



def find_word_index(sentence, word):
    pattern = r"\b{}\b".format(re.escape(word))
    match = re.search(pattern, sentence)
    if match:
        return match.start()
    return -1