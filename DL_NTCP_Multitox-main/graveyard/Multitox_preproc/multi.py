"""
from multiprocessing import Process, Manager
import multiprocessing

def dothing(L, i):  # the managed list `L` passed explicitly.
    L.append("anything")

if __name__ == "__main__":
    with Manager() as manager:
        L = manager.list()  # <-- can be shared between processes.
        processes = []
        for i in range(5):
            p = Process(target=dothing, args=(L,i))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print(L)


def double(a):
    return a * 2

def driver_func():
    PROCESSES = 1
    with multiprocessing.Pool(PROCESSES) as pool:
        params = [(1, ), (2, ), (3, ), (4, )]
        results = [pool.apply_async(double, p) for p in params]

        for r in results:
            print('\t', r.get())

if __name__ == "__main__":
    driver_func()

"""


import itertools
from multiprocessing import Pool, freeze_support

def func(a, b):
    print(a, b)

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)

def main():
    pool = Pool()
    a_args = [1,2,3]
    second_arg = 1
    pool.map(func_star, zip(a_args, itertools.repeat(second_arg)))

if __name__=="__main__":
    freeze_support()
    main()