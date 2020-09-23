import os
import pandas as pd
import random

from z3 import *
from random import choice
from tqdm import tqdm

def is_equivalent(left, right, num_bits):
    '''
    Judge the equilalence of left and right.
    '''
    x, y, z, t, a, b, c, d, e = z3.BitVecs('x y z t a b c d e', num_bits)
    s = z3.Solver()
    s.add(eval(left) != eval(right))
    if s.check() == z3.unsat:
        return True
    return False


def main():


    # file = './train.csv'
    file = './2_varis.csv'
    file2 = './3_varis.csv'
    file3 = './more_varis.csv'

    variables = ['x', 'y']
    replacement_variables = ['a', 'b', 'c', 'd', 'e', 't']

    dataset = []
    with open(file, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    dataset = dataset[:40000]
    print(len(dataset))

    with open(file2, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))

    dataset = dataset[:80000]
    print(len(dataset))

    with open(file3, 'r') as f:
        for line in f:
            # if len(line) < 78:
            dataset.append(line.strip().split(','))

    dataset = dataset[:100000]
    print(len(dataset))

    # num_bits = 8
    # ans = []
    # count = 0
    # for i in tqdm(range(len(dataset))):
    #     if is_equivalent(dataset[i][0], dataset[i][1], num_bits):
    #         count += 1
    #         ans.append(dataset[i])

    # print(f'Total samples: {count}')

    random.shuffle(dataset)
    ds = pd.DataFrame(dataset)
    ds.to_csv('train_data.csv', header=False, index=False)


if __name__ == '__main__':
    main()