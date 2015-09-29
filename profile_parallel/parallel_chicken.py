"""
The multiprocessing joke.
"""
from __future__ import print_function

import multiprocessing

def printer(word):
    print(word, end=' ')

def main():
    print()
    print('Why did the parallel chicken cross the road?')
    answer = 'To get to the other side.'
    pool = multiprocessing.Pool(processes=6)

    print()
    pool.map(printer, answer.split())
    pool.close()
    pool.join()
    print()
    print()


if __name__ == '__main__':
    main()
