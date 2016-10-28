import os
import pandas as pd
import numpy as np
       
os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample.csv')

distance_pos = offline['User_id'].groupby([offline['distance'],offline['label']]).count().reset_index()
distance = []

dis_rate_pos =  offline['User_id'].groupby([offline['dis_rate'],offline['label']]).count().reset_index()
dis_rate = []

week_pos =  offline['User_id'].groupby([offline['weekd'],offline['label']]).count().reset_index()
week = []


for i in xrange(offline.shape[0]):
    if i % 10000 == 0:
        print(i)
    dist = offline['distance'][i]
    pos = distance_pos['User_id'][(distance_pos['distance'] == dist) & (distance_pos['label'] == 0)]
    total = distance_pos['User_id'][(distance_pos['distance'] == dist)].sum()
    prob = pos / total
    distance.append(prob.values[0])
    
    disrate = offline['dis_rate'][i]
    pos = dis_rate_pos['User_id'][(dis_rate_pos['dis_rate'] == disrate) & (dis_rate_pos['label'] == 0)]
    total = dis_rate_pos['User_id'][(dis_rate_pos['dis_rate'] == disrate)].sum()
    prob = pos / total
    dis_rate.append(prob.values[0])
    
    weekday = offline['weekd'][i]
    pos = week_pos['User_id'][(week_pos['weekd'] == weekday) & (week_pos['label'] == 0)]
    total = week_pos['User_id'][(week_pos['weekd'] == weekday)].sum()
    prob = pos / total
    week.append(prob.values[0])
    
    
    
offline['distance'] = distance
offline['dis_rate'] = dis_rate
offline['weekd'] = week

offline.to_csv('../data/distance_sample_train.csv',index=False)
