#!/usr/bin/python3

import func_timeout
import re
from sspam import simplifier
import sys
import time
import z3
import pandas as pd

# class Logger(object):
#     def __init__(self, fileN="Default.log"):
# 		self.terminal = sys.stdout
# 		self.log = open(fileN, "w+")
 
# 	def write(self, message):
# 		self.terminal.write(message)
# 		self.log.write(message)
 
# 	def flush(self):
# 		pass

def verify_equivalent(leftExpre, rightExpre, bitnumber=8):
	x,y,z,a,b,c,d,e = z3.BitVecs("x y z a b c d e", bitnumber)

	try:
		leftEval = eval(leftExpre)
		rightEval = eval(rightExpre)
	except:
		return 'unsat'

	solver = z3.Solver()
	solver.add(leftEval != rightEval)
	result = solver.check()

	return str(result)

def main(sourcefilename, desfilename):
	ds = pd.read_csv(sourcefilename, header=None)
	f = open(desfilename, "w+")
	source = ds[0]
	target = ds[1]

	test_count = 100
	correct_count = 0
	sum_solve_time = 0
	for i in range(test_count):
		print "No.%d" % (i + 1),
		print "=" * 50
		print "MBA expr:", source[i]
		print "Target expr:", target[i]
		start = time.time()
		try:
			#use func_timeout module:doitreturnvalue=func_timeout(timeout, doit, args)
			simStr = func_timeout.func_timeout(3600, simplifier.simplify, args=(source[i],))
		except func_timeout.FunctionTimedOut:
			#enough time to clean the z3 circument.
			time.sleep(1)
			print "Solve result: Time out"
			print "Time out."
		#except:
			#print "errors!!"
		else:
			end = time.time()
			elapsed = end - start
			sum_solve_time += elapsed
			print "Solve result:", simStr
			z3Result = verify_equivalent(target[i], simStr)
			if z3Result != 'unsat':
				print "Solve Flase"
			else:
				correct_count += 1
				print "Solve True"
			print "Time =", elapsed
			print ""

	print "#Correct samples:", correct_count
	print "Average solve time: %.4f" % (sum_solve_time/correct_count)
	
	
if __name__ == "__main__":
	sourcefilename = "../../data/linear/test/test_data.csv"
	# desfilename = "sspam_verify_result.txt"
	# sys.stdout = Logger(desfilename)
	# main(sourcefilename, desfilename)
	# print(simplifier.simplify('(x&y)-(~x&y)+(x^y)+3*(~(x|y))-(~(x^y))-(x|~y)-(~x)-1'))
	print(verify_equivalent('(x&y)-(~x&y)+(x^y)+3*(~(x|y))-(~(x^y))-(x|~y)-(~x)-1', '(((((x+y)-(((-x)+3)&y))-(3*(x|y)))+(2*(x^y)))+3)'))


