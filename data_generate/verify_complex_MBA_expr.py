from z3 import *
import pandas as pd

data = pd.read_csv("dataset.csv", header=None)
origin = list(data[0])
mba_confusion = list(data[1])

x = BitVec('x', 32)
y = BitVec('y', 32)
z = BitVec('z', 32)

print("Total number of data:", len(origin))
count = 0
for i in range(0, len(origin)):
    print "No.%d:" % i,
    solve(eval(origin[i]) != eval(mba_confusion[i]))
    count += 1
print("The number of test sample:", count)