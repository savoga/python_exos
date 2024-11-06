import time
from multiprocessing import Pool
import numpy as np

n_iter = 100000
n_fill = 10000

def compute_sum():
    s = 0
    for i in range(n_iter):
        s += 1
    return s

def parallel_fill(n_fill):
    # Pool: the number of processes created by default is equal to the number of CPUs on the machine
    with Pool() as pool:
        results = [pool.apply_async(compute_sum) for _ in range(n_fill)] # we create n_fill jobs (?)
        results = [result.get() for result in results] # get() waits before the result is here (= asynchronous)
        # --> take the results that are finished first? (= synchronous?)
    return results

# we create the main module to act like a "guard" and prevent
# any unintended behavior in the processes
# this block also ensures that the multiprocessing code runs 
# only when the script is executed directly, not when it is imported as a module.
if __name__ == "__main__":
    start_time = time.time()
    arr_fill = np.zeros(n_fill)
    arr_fill[:] = parallel_fill(n_fill)
    print(round(time.time()-start_time,2))
    

# start_time = time.time()
# arr_fill = np.zeros(n_fill)
# for i in range(n_fill):
#     arr_fill[i] = compute_sum()
# print(round(time.time()-start_time,2)) 

