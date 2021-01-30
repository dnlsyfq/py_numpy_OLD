import numpy as np

c = np.loadtxt('data.csv',delimiter=',',usecols=(6,),unpack=True)
N = len(c)

print(f"Median = {np.median(c)}")

sorted_close = np.msort(c)

print(f"Middle = { sorted_close[int((N-1)/2)] }")

print(f"Variance = { np.var(c) }")