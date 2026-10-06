
import pandas as pd
import numpy as np
import math
from numpy import array


def cal_weight(x):

    # minmax_scaler = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)))
    # x = minmax_scaler(x)
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))


    rows = x.index.size  
    cols = x.columns.size  
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]


    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf


    d = 1 - E.sum(axis=0)

    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
     

    w = pd.DataFrame(w)
    return w


if __name__ == '__main__':

    import pandas as pd
    # df = np.loadtxt('OriginalData.txt')
    df = pd.read_csv('ReplacedData.txt', sep=' ', header=None)

    # df.dropna()

    w = cal_weight(df)  #
    # w.index = df.columns
    # w.columns = ['weight']
    print(w)
    np.savetxt('entropy_vector.txt',w)

    
    scores = []
    for i in range(len(df)):
        scores.append(np.dot(df.values[i], w.values)[0])
   
    work_choice = []
    for i in range(int(len(scores)/8)):
        work_scores = scores[i*8:(i+1)*8]
        max_score_index = work_scores.index(max(work_scores))
        print(work_scores, max_score_index)
        work_choice.append(max_score_index)
    np.savetxt('work_choice.txt', work_choice)
    print('love world')
 

