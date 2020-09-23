from arybo.lib import MBA
import func_timeout
import io
import re
import sys
import time
import pandas as pd
import sys
 
class Logger(object):
	def __init__(self, fileN="Default.log"):
		self.terminal = sys.stdout
		self.log = open(fileN, "w+")
 
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
 
	def flush(self):
		pass
 

def AryboSimplify(bitnum, sourceExpreStr):
	"""simplify the expression by arybo
	Args:
		bitnum: the number of the bit of the variables.
		sourceExpreStr: expression is simplified.
	Returns:
		result: the result after simplification.
	Raise:
		None.
	"""
	mba = MBA(bitnum)
	x = mba.var("x")
	y = mba.var("y")
	z = mba.var("z")
	t = mba.var("t")
	a = mba.var("a")
	b = mba.var("b")
	c = mba.var("c")
	d = mba.var("d")
	e = mba.var("e")
	f = mba.var("f")

	res = eval(sourceExpreStr)
	fio = io.StringIO()
	print(res,file=fio)

	result = fio.getvalue()
	fio.close()

	return result



def main(sourcefilename, desfilename, bitnum):
	fwrite = open(desfilename, "w")
	fwrite.write("#source,correct,simplified,simfilied time\n")

	ds = pd.read_csv(sourcefilename, header=None)
	source = ds[0]
	target = ds[1]

	num_of_test_data = 100

	solved_count = 0
	total_solve_time = 0
	for i in range(num_of_test_data):
		print("No.%d" % (i + 1), end=' ')
		print("=" * 50)
		print("MBA expr:", source[i])
		print("Target expr:", target[i])
		start_time = time.time()
		try:
			#use func_timeout module:doitreturnvalue=func_timeout(timeout, doit, args)
			simExpreStr = func_timeout.func_timeout(3600, AryboSimplify, args=(bitnum, source[i]))
		except func_timeout.FunctionTimedOut:
			#enough time to clean the z3 circument.
			time.sleep(2)
			print("Solve result: Time out")
			print("Time out.")
		#except:
			#print "errors!!"
		else:
			end_time = time.time()
			elapsed = end_time - start_time
			solved_count += 1
			simExpreStr = simExpreStr.replace("\n", "")
			print("Solve result:", simExpreStr)
			print()


if __name__ == "__main__":  
	sourcefilename = "../../data/linear/test/test_data.csv"
	desfilename = "arybo_verify_result.txt"
	bitNum = 8
	# sys.stdout = Logger(desfilename)
	# main(sourcefilename, desfilename, bitNum)
	print(AryboSimplify(4, '(x&y)-(~x&y)+(x^y)+3*(~(x|y))-(~(x^y))-(x|~y)-(~x)-1'))
