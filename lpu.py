"""This script converts the data
to LPU's format
"""


import os
import numpy as np
import pandas as pd


def to_lpu(x, y=None):
    """Converts a numpy 2-D matrix to LPU format as a string
    """
    d1, d2 = x.shape
    rows = []
    if y is not None:
        assert y.shape == (d1,)

    for row in range(d1):
        rows.append([])
        if y is not None:
            rows[-1].append(str(y[row]))

        for attr in range(d2):
            v = x[row, attr]
            if v > 0:
                rows[-1].append(f'{attr + 1}:{v}')

        rows[-1] = ' '.join(rows[-1])

    return '\n'.join(rows)


def queryframe_to_lpu(dfx, dfquery, classcolumn):
    """Given a data frame and a query frame,
    produces a list of LPU triplets with an index
    """
    for query in dfquery.iterrows():
        index = query[0]
        positive = np.array([int(x) for x in query[1]['query'].split(',')])
        target = np.array([int(x) for x in query[1]['target'].split(',')])
        positive_data = dfx.iloc[positive, :].drop(classcolumn, axis=1)
        target_data = dfx.iloc[target, :].drop(classcolumn, axis=1)

        mask = np.zeros(dfx.shape[0], dtype=np.bool)
        mask[target] = True
        targety = np.zeros(dfx.shape[0], dtype=np.int)
        targety[mask] = 1
        targety[~mask] = -1

        positive_lpu = to_lpu(positive_data.as_matrix())
        unlabeled_lpu = to_lpu(target_data.as_matrix())
        test_lpu = to_lpu(dfx.iloc[:, 1:].as_matrix(), targety)

        yield index, positive_lpu, unlabeled_lpu, test_lpu

def cnae():
    DATA_PATH = 'data'
    TARGET_PATH = os.path.join(DATA_PATH, 'cnae-query')

    dfx = pd.read_csv(os.path.join(DATA_PATH, 'CNAE-9.data'), header=None)
    dfquery = pd.read_csv(os.path.join(DATA_PATH, 'CNAE-9_query.csv'))

    for index, positive_lpu, unlabeled_lpu, test_lpu in queryframe_to_lpu(dfx, dfquery, 0):
        with open(os.path.join(TARGET_PATH, str(index) + '.pos'), 'w') as pos:
            pos.write(positive_lpu)
        with open(os.path.join(TARGET_PATH, str(index) + '.unlabel'), 'w') as unlabel:
            unlabel.write(unlabeled_lpu)
        with open(os.path.join(TARGET_PATH, str(index) + '.test'), 'w') as test:
            test.write(test_lpu)


def synthetics():
    DATA_PATH = 'data'
    TARGET_PATH = os.path.join(DATA_PATH, 'synth-query')
    #DATASETS = ('densebinary', 'denseinteger', 'sparsebinary', 'sparseinteger')
    DATASETS = ('sparseinteger',)

    for ds in DATASETS:
        dfx = pd.read_csv(os.path.join(DATA_PATH, ds + '.csv'))
        dfquery = pd.read_csv(os.path.join(DATA_PATH, ds + '_query.csv'))

        for index, positive_lpu, unlabeled_lpu, test_lpu in queryframe_to_lpu(dfx, dfquery, 'target'):
            with open(os.path.join(TARGET_PATH, ds + str(index) + '.pos'), 'w') as pos:
                pos.write(positive_lpu)
            with open(os.path.join(TARGET_PATH, ds + str(index) + '.unlabel'), 'w') as unlabel:
                unlabel.write(unlabeled_lpu)
            with open(os.path.join(TARGET_PATH, ds + str(index) + '.test'), 'w') as test:
                test.write(test_lpu)


def main():
    cnae()
    synthetics()


if __name__ == '__main__':
    main()
