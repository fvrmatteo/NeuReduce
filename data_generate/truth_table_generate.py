import sys
import re
import numpy as np
from functools import reduce

class TruthTableGenerate(object):
    """Generate basic truth table.

    Attributes:
        n_variables: The number of variables.
        indexes: The index collection of maxterm.
        maxterm: The collection of maxterm 
    """
    def __init__(self, n_variables):
        self.n_variables = n_variables
        self.indexes = []
        self.maxterm = []
        self.exprs = []
        self.truth_tables = {}
        self.boolean_op = ["&", "|", "^"]
        self.variables = ["x", "~x", "y", "~y"]
    
    def maxterm_generate(self):
        """
        Generate maxterms of Principal Conjunctive Normal Form.
        For example, two variables x and y has 4 maxterms
        [x|y, x|~y, ~x|y, ~x|~y], and three variables
        x, y and z has 8 maxterms

        Returns: None. 
        """
        v_string = "xyzwst"[:self.n_variables]

        # temp_list is a list used to generate maxterm,
        # For example, if v_string="xy", 
        # temp_list=[['x', '~x'], ['|'], ['y', '~y']]
        temp_list = []
        for i in range(len(v_string)):
            temp_list.append([v_string[i], "~"+v_string[i]])
            temp_list.append("|")

        def combination(list1, list2):
            return [str(i) + str(j) for i in list1 for j in list2]

        self.maxterm = reduce(combination, temp_list[:-1])

    def index_combine(self, start_index, temp):
        """
        Recursively generate all possible combinations of
        maxterm.

        For example, if we have two variables x and y, then
        [0, 2] means M_0 and M_2, which also is [x|y, ~x|~y]

        Args:
            start_index: An integer which indicate start index
            temp: A temporary list

        Returns: None
        """
        if len(temp) == len(self.maxterm) - 1:
            self.indexes.append(list(set(temp)))
            return

        for i in range(start_index, len(self.maxterm)):
            temp.append(i)
            self.index_combine(i, temp)
            temp.pop()

        return

    def bvtb_generate(self):
        """
        Generate Basic Variables Truth Tables.
        If we have two variables, this function will generate
        [[0, 0, 1, 1], [0, 1, 0, 1]] which is the truth table
        of x and y.

        Returns:
            A list.
        """
        base_v_list = []
        for i in range(self.n_variables):
            temp = []
            n_bolck = 2**(i + 1)
            for j in range(n_bolck):
                temp += [j % 2] * (2**self.n_variables // n_bolck)
            base_v_list.append(temp)
        return base_v_list

    def truth_table_generate(self):
        """Generate truth table of 2**(2**n_variable)-1  basic expressions"""
        self.maxterm_generate()
        self.index_combine(0, [])
        # Remove duplicate index
        self.indexes = list(set([tuple(i) for i in self.indexes]))

        for item in self.indexes:
            expr = ""
            for index in item:
                if len(item) == 1:
                    expr += self.maxterm[index] + "&"
                else:
                    expr += "(" + self.maxterm[index] + ")&"
            self.exprs.append(expr[:-1])
        
        # =========================
        self.exprs = []
        with open("truth_table_3_variables.txt", "r") as data:
            for line in data:
                line = line.strip("\n")
                if len(line) <= 40:
                    e = re.split(" = ", line)
                    self.exprs.append(e[0])
        # =========================

        variables_value = self.bvtb_generate()
        # After transpose, each row contains the value of all variables
        variables_value = np.array(variables_value).T.tolist()

        v_string = "xyzwst"[:self.n_variables]
        all_list = []
        for value in variables_value:
            # Assign a value to each variable
            for i in range(self.n_variables):
                exec(v_string[i]+ "=%d" % value[i])
            temp_list=[]
            for expr in self.exprs:
                temp_list.append(eval(expr) % 2)
            all_list.append(temp_list)

        truth_table = []
        for i in range(len(all_list[0])):
            truth_table.append([all_list[k][i] for k in range(2**self.n_variables)])
        for i in range(len(self.exprs)):
            self.truth_tables[self.exprs[i]] = truth_table[i]
        self.truth_tables["-1"] = [1] * (2**self.n_variables)
        self.exprs.append("-1")

    # def what_is_the_PCNF_form_of_expr(self, expr):
    #     v_string = "xyzwst"[:n_variables]
    #     xyz = self.bvtb_generate()
    #     xyz = np.array(xyz).T.tolist()
    #     tb = []
    #     for val in xyz:
    #         for i in range(self.n_variables):
    #             exec(v_string[i] + "=%d" % val[i])
    #         tb.append(eval(expr) % 2)
    #     print(expr, tb)
    #     for k, v in self.truth_tables.items():
    #         if v == tb:
    #             print(k, v)
    #             print("")

# def generate_all_possiable(self):
#         base_list = self.generate_base_list()
        
#         # Function for the parameter of reduce()
#         def combination(list1, list2):
#             return [str(i) + str(j) for i in list1 for j in list2]
        
#         return reduce(combination, base_list)

            

# if __name__ == "__main__":
#     n_variables = int(sys.argv[1])
#     obj = TruthTableGenerate(n_variables=n_variables)
#     obj.truth_table_generate()
#     print(*obj.exprs, sep="\n")
#     for k, v in obj.truth_tables.items():
#         print(k, v)
#     v1 = ["x", "~x"]
#     op1 = ["&", "|", "^"]
#     v2 = ["y", "~y"]
#     op2 = ["&", "|", "^"]
#     v3 = ["z", "~z"]
#     base_list = [["~"], ["("], v1, op1, ["("], v2, op2, v3, [")"], [")"]]

#     def combination(list1, list2):
#         return [str(i) + str(j) for i in list1 for j in list2]

#     expr = reduce(combination, base_list)
#     for i in expr:
#         obj.what_is_the_PCNF_form_of_expr(i)
