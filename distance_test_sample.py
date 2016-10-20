import os
import pandas as pd
import numpy as np

os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample.csv')
distance_pos = offline['User_id'].groupby([offline['distance'],offline['label']]).count().reset_index()
distance = []

test = pd.read_csv('../data/test_sample.csv')

for i in xrange(test.shape[0]):
    dist = offline['distance'][i]
    pos = distance_pos['User_id'][(distance_pos['distance'] == dist) & (distance_pos['label'] == 1)]
    all = distance_pos['User_id'][(distance_pos['distance'] == dist)].sum()
    distance.append(pos / all)
offline['distance'] = distance

offline.to_csv('../data/distance_sample_test.csv',index=False)
