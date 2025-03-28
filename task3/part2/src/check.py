import math
import time

from lgbt import lgbt

def read_lines(file):
	with open(file, "r") as f:
		return_file = f.read().split('\n')
	return return_file

def check_func(lines, func, desc="Check"):
	passed = 0
	failed = 0

	for line in lgbt(list(lines), desc=" read", hero='unicorn'):
		sep_line = line.split(' ')
		res = func(sep_line)
		if res:
			passed += 1
		else:
			failed += 1
	print(desc)
	print(f'passed {passed}  failed {failed} from {len(lines)}')


def main():
	fpow = read_lines("out_pow.txt")
	fsin = read_lines("out_sin.txt")
	fsqrt = read_lines("out_sqrt.txt")

	pow_func = lambda x:  math.isclose(math.pow(float(x[1]),float(x[2])), float(x[3]), rel_tol=1e-3) 
	sin_func = lambda x:  math.isclose(math.sin(float(x[1])), float(x[3]), rel_tol=1e-1) 
	sqrt_func= lambda x:  math.isclose(math.sqrt(float(x[1])), float(x[3]), rel_tol=1e-3) 


	print(type(fpow[:-1:]))
	check_func(fpow[:-1:], pow_func, "pow")
	check_func(fsin[:-1:], sin_func, "sin")
	check_func(fsqrt[:-1:], sqrt_func, "sqrt")
	
if __name__ == "__main__":
	main()