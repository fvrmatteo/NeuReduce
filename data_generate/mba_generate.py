#!/usr/bin/python2

import argparse
import random
import pandas as pd
from truth_table_generate import TruthTableGenerate
from functools import reduce
from sympy import Matrix, zeros, lcm
from tqdm import tqdm



class MBA_Generator(object):
    """Generate Linear MBA obfuscation expressions, each terms in
    expression is the simplest form of two variables's boolean
    expression, we list those simplest form as member variable expr.

    Attributes:
        n_terms: The number of terms which is the simplest boolean expression
            e.g. -2x-2y+4(x&y)+(x&~y)+(x^y)-~(x^y)+(~x|y)=0 has 7 terms.
        exprs: 2^(2^n_variables) basic expressions, which can deduce
            all other expressions. And we use -1 to represent constant.
        truth_tables: The truth of expressions in exprs.
        indexes: We enumerate all possible combinations of 
            expressions in exprs, and use subscripts to 
            represent expressions, we collect these possible
            to a list indexes.  
    """

    def __init__(self, n_terms=5, n_variables=2):
        """Init MBA_Generator with n_terms"""
        self.n_terms = n_terms
        self.n_variables = n_variables
        self.exprs = ["x&y", "x|y", "x&~y", "~x&y", "x^y", "~(x|y)", "~(x^y)", 
            "~y", "x|~y", "~x", "~x|y", "~(x&y)", "x", "y", "-1"]
        self.truth_tables = {
            "x": [0, 0, 1, 1],
            "y": [0, 1, 0, 1],
            "x&y": [0, 0, 0, 1],
            "x&~y": [0, 0, 1, 0],
            "~x&y": [0, 1, 0, 0],
            "x^y": [0, 1, 1, 0],
            "x|y": [0, 1, 1, 1],
            "~(x|y)": [1, 0, 0, 0],
            "~(x^y)": [1, 0, 0, 1],
            "~y": [1, 0, 1, 0],
            "x|~y": [1, 0, 1, 1],
            "~x": [1, 1, 0, 0],
            "~x|y": [1, 1, 0, 1],
            "~(x&y)": [1, 1, 1, 0],
            "-1": [1, 1, 1, 1],
            }
        self.indexes = []
        self.ans = []

    def index_combine(self, start_index, temp_list):
        """
        Recursively generate all possible combinations of
        non-repeating n_term expressions in exprs and store
        all possible combination in self.indexes.

        Args:
            start_index: An integer which indicate start index
            temp_list: A temporary list

        Returns: None
        """

        if len(temp_list) == self.n_terms:
            self.indexes.append(list(temp_list))
            return

        # for i in range(start_index, len(self.exprs)):
        i = start_index
        while i < len(self.exprs):
            temp_list.append(i)
            self.index_combine(i+1, temp_list)
            temp_list.pop()
            i += 10**(self.n_variables - 2)

    def calculate_v(self, F):
        """
        Calculate the nullspaces v with Fv=0.
        
        Args:
            F: A matrix which is the input of Fv=0.
        
        Returns: A vector v, which is nullspace of Fv = 0,
        there is no 0 component in v, else return all 0 vector.
        """

        # Solutions is nullspace list of Fv=0
        solutions = F.nullspace()

        v = zeros(self.n_terms, 1)
        # Add all solutions together, it is also a solution
        for sol in solutions:
            v += sol

        # Find index of first zero-value
        index = self.first_zero_index(v)

        # Try to generate non-zero matrix v, if there is no 0 in v,
        # v is the final solution and we return it, else we change
        # v through the linear combination of sol in solutions.
        while index != -1:
            # If there is only one solution of Fv=0, 
            # and v contains 0, it means that for this F, 
            # there is no solution containing n_terms terms, 
            # e.g., v=[-1,0,1,1], the solution only contains 3 terms
            if len(solutions) < 2:
                return zeros(v.shape[0], v.shape[1])

            # Add non-zero value of nullpace to matrix v
            has_non_zero_nullspace = False
            for sol in solutions:
                if sol[index] != 0:
                    has_non_zero_nullspace = True
                    v += sol
                    break

            if has_non_zero_nullspace == False:
                return zeros(v.shape[0], v.shape[1])

            # Try to eliminate the next 0 in v
            index = self.first_zero_index(v)
        return v

    def first_zero_index(self, vector):
        """
        Find the index of first zero-value in vector

        Args:
            v: A vector, which is a solution of FV=0.

        Returns: 
            The index of first zero-value,
            or -1 if there is no zero-value in vector.
        """
        for i in range(0, len(vector)):
            if vector[i] == 0:
                return i

        return -1

    def expression_generate(self, index_of_exprs, v):
        """
        Generate a canonical output

        Args:
            index_of_exprs: The index of expression in self.exprs
            v: The nullspace of F
        
        Returns: None
        """

        left = ""
        right = ""
        index = 0

        for i in index_of_exprs:
            sign = True
            coefficient = self.exprs[i]

            # generate coefficient
            if len(coefficient) > 1:
                coefficient = "(" + coefficient + ")"

            if coefficient== "-1":
                if v[index] > 0:
                    sign = False
                    coefficient = str(v[index])
                else:
                    sign = True
                    coefficient = str(-v[index])
            elif v[index] > 0:
                sign = True
                if v[index] > 1:
                    coefficient = str(v[index]) + "*" + coefficient
            else:
                sign = False
                if v[index] < -1:
                    coefficient = str(-v[index]) + "*" + coefficient

            if (i < 2) or (not left):
                if left:
                    left += "+" + coefficient if sign else "-" + coefficient
                else:
                    left = coefficient if sign else "-" + coefficient
            else:
                if right:
                    right += "-" + coefficient if sign else "+" + coefficient
                else:
                    right = "-" + coefficient if sign else coefficient

            index += 1

        if ("x" not in left) and ("x" not in right):
            left += "+x"
            right += "+x"
        if ("y" not in left) and ("y" not in right):
            left += "+y"
            right += "+y"
        if self.n_variables > 2:
            if ("z" not in left) and ("z" not in right):
                left += "+z"
                right += "+z"
        if "+" not in left and "-" not in left and "*" not in left:
            if left[0] == "(" and left[len(left) - 1] == ")":
                left = left[1:len(left)-1]

        self.ans.append([left, right])

    def generate(self):
        """
        For each possible combination of terms in exprs(i.e. we
        choice n_terms from exprs to combine),
        """

        self.index_combine(0, [])

        for i in tqdm(range(len(self.indexes))):
            index = self.indexes[i]
            # Generate matrix F
            F = Matrix()
            for i in index:
                F = F.col_insert(F.shape[1], 
                        Matrix(self.truth_tables.get(self.exprs[i])))

            # Calculate the vector v that satisfy Fv=0
            v = self.calculate_v(F)

            # If v is a zero vector, which means Fv = 0 has no non-trivial
            # solution, we jump this combination and process next combination
            if v == zeros(self.n_terms, 1):
                continue

            # Make sure v is integer, sometimes there is decimal in v,
            # e.g. v = [-1/3,-1/3,-1/3,-2/3,1], val.q will take the
            # denominator of v, i.e. [i.q for i in v] = [3,3,3,3,1],
            # then we calculate the LCM of denominator and transform 
            # v to integer.
            m = lcm([val.q for val in v])
            v = m * v

            # format output from v and expressions
            self.expression_generate(index, v)

if __name__ == "__main__":
    """
    Accept 2 parameters from command line, first is the number of
    terms, second is the number of variables.
    """

    # parser = argparse.ArgumentParser(description='Generate MBA datasets.')
    # parser.add_argument('--n_terms', type=int, default=5,
    #     help='number of terms')
    # parser.add_argument('--n_varis', type=int, default=2,
    #     help='number of variables')
    # parser.add_argument('--dest', type=str, default="./data/dataset.csv",
    #     help='the file used to store datasets.')

    # args = parser.parse_args()

    # ge = MBA_Generator(n_terms=args.n_terms,n_variables=args.n_varis)

    # if args.n_varis > 2:
    #     th = TruthTableGenerate(args.n_varis)
    #     th.truth_table_generate()
    #     ge.truth_tables = th.truth_tables
    #     ge.exprs = th.exprs
    for i in range(3, 8):
        ge = MBA_Generator(n_terms=i, n_variables=2)
        ge.generate()
        df = pd.DataFrame(ge.ans)
        df.to_csv("./dataset.csv", mode="a+", header=False, index=False)
    # df.to_csv(args.dest, mode="a+", header=False, index=False)