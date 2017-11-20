"""Reads the MULAN datasets for Pandas format
"""


import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def main():
    DATA_PATH = os.path.expanduser('~/Documents/datasets/mulan')
    DATASETS = ('birds-test',
                'birds-train',
                'CAL500',
                'emotions',
                'emotions-test',
                'emotions-train',
                'mediamill',
                'mediamill-test',
                'mediamill-train',
                'yeast',
                'yeast-test',
                'yeast-train')
    CLASSES = ('birds', 'CAL500', 'emotions', 'mediamill', 'yeast')

    classes = {}
    for cls in CLASSES:
        xmlfile = os.path.join(DATA_PATH, cls + '.xml')
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        classes[cls] = [child.attrib['name'] for child in root]

    dframes = {}

    for ds in DATASETS:
        csvfile = os.path.join(DATA_PATH, ds + '.csv')
        df = pd.read_csv(csvfile, escapechar='\\')
        df.columns = [s.strip("'") for s in df.columns]
        actualset = ds.split('.')[0].split('-')[0]
        assert all(c in df.columns for c in classes[actualset])

        dframes[ds] = df
        data = df.drop(classes[actualset], axis=1)
        labels = df[classes[actualset]]

        data.to_csv(os.path.join(DATA_PATH, ds + '_X.csv'), index=False)
        labels.to_csv(os.path.join(DATA_PATH, ds + '_Y.csv'), index=False)

    # also adds birds dataset concatenating the train and test sets
    birds = pd.concat([
                pd.read_csv(os.path.join(DATA_PATH, 'birds-train.csv'),
                            escapechar='\\'),
                pd.read_csv(os.path.join(DATA_PATH, 'birds-test.csv'),
                            escapechar='\\')
            ])
    birds.columns = [s.strip("'") for s in birds.columns]
    data = birds.drop(classes['birds'], axis=1)
    labels = birds[classes['birds']]
    dframes['birds'] = birds

    data.to_csv(os.path.join(DATA_PATH, 'birds_X.csv'), index=False)
    labels.to_csv(os.path.join(DATA_PATH, 'birds_Y.csv'), index=False)


if __name__ == '__main__':
    main()
