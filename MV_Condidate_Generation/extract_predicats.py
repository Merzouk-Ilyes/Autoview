def ExtractPredicates(Predicates_Dictionary, Predicates):
    for operator in Predicates_Dictionary.keys():
        if operator in ['or', 'and']:
            predicates_list = Predicates_Dictionary[operator]
            for predicat in predicates_list:
                ExtractPredicates(predicat, Predicates)
        else:
            operator = list(dict(Predicates_Dictionary).keys())
            list_predicate_members = dict(Predicates_Dictionary)[operator[0]]
            left_member = list_predicate_members[0]
            right_member = list_predicate_members[1]
            if isinstance(right_member, dict):  # is a dictionary
                type_of_right_member = list(dict(right_member).keys())
                if type_of_right_member[0] == 'literal':
                    second = str(right_member[type_of_right_member[0]])
                    selection_predicate = left_member, operator[0], second
                    Predicates.append(selection_predicate)
                else:
                    print('ERROR')

            else:  # is a list
                selection_predicate = left_member, operator[0], right_member
                Predicates.append(selection_predicate)

    return Predicates
