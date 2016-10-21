import os
import pandas as pd
import numpy as np

offline = pd.read_csv('../data/base_sample.csv')
distance_pos = offline['User_id'].groupby([offline['distance'],offline['label']]).count().reset_index()
distance = []

test = pd.read_csv('../data/test_sample.csv')

for i in xrange(test.shape[0]):
    dist = offline['distance'][i]
    pos = distance_pos['User_id'][(distance_pos['distance'] == dist) & (distance_pos['label'] == 0)]
    alldis = distance_pos['User_id'][(distance_pos['distance'] == dist)].sum()
    dis_pro = pos / alldis
    distance.append(dis_pro.values[0])
offline['distance'] = distance

test.to_csv('../data/distance_sample_test.csv',index=False)
