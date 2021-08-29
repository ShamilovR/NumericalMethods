import numpy as np
from prettytable import PrettyTable
import pandas as pd
import math

ITERATION_LIMIT = 1000
ACCURACY = 1e-3

pd.set_option('precision', 8)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 180)

A = np.array([[1., 2./15, -11./15, 1./15],
              [1./15, 1., 1./15, 1./15],
              [1./31, 1./31, 1., -1./31],
              [1./7, 1./7, 1./7, -1.]])

b = np.array([25./15, -370./7/15, 807./14/31, -3./7])

x0 = np.array(input().split()).astype(float)
for i in range(0,2):
    print('Точность: ', ACCURACY)
    print('Начальные значения: ', x0)

    print('========= Метод Зейделя =========')
    columns = [
        'n+1',
        'x1^n', 'x1^(n+1)', '|x1^(n+1) - x1^n|',
        'x2^n', 'x2^(n+1)', '|x2^(n+1) - x2^n|',
        'x3^n', 'x3^(n+1)', '|x3^(n+1) - x3^n|',
        'x4^n', 'x4^(n+1)', '|x4^(n+1) - x4^n|',
        's'
    ]
    data = []
    x = x0

    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        diffs = np.absolute(x_new - x)
        data.append([
            it_count,
            x[0], x_new[0], diffs[0],
            x[1], x_new[1], diffs[1],
            x[2], x_new[2], diffs[2],
            x[3], x_new[3], diffs[3],
            np.sum(diffs)
        ])
        if (diffs[0] <= ACCURACY and diffs[1] <= ACCURACY and diffs[2] <= ACCURACY and diffs[3] <= ACCURACY):
            break
        #if np.sum(np.absolute(diffs)) <= ACCURACY:
            #break
        x = x_new

    print(pd.DataFrame(data, columns=columns).to_string(index=False))

    print('========= Метод спуска =========')
    columns=[
        'n+1',
        'x1^n', 'x1^(n+1)', 'r1',
        'x2^n', 'x2^(n+1)', 'r2',
        'x3^n', 'x3^(n+1)', 'r3',
        'x4^n', 'x4^(n+1)', 'r4',
        'sum'
    ]
    data = []
    x = x0

    for it_count in range(1, ITERATION_LIMIT):
      r = A.dot(x) - b
      temp_1 = np.transpose(A).dot(r)
      temp_2 = A.dot(temp_1)
      temp_3 = r.dot(temp_2) / temp_2.dot(temp_2)
      rs = temp_3 * temp_1
      x_new = x - rs

      data.append([it_count,
        x[0], x_new[0], np.absolute(rs[0]),
        x[1], x_new[1], np.absolute(rs[1]),
        x[2], x_new[2], np.absolute(rs[2]),
        x[3], x_new[3], np.absolute(rs[3]),
        np.sum(np.absolute(rs))
      ])

      if all(np.absolute(i) <= ACCURACY for i in rs):
       break
      x = x_new
    print(pd.DataFrame(data, columns=columns).to_string(index=False))
    ACCURACY = 1e-5 
