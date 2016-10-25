import os
import pandas as pd
import numpy as np
       
os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample.csv')
test = pd.read_csv('../data/test_sample.csv')

def get_week_prob(week_day,prob):
    mon = 0
    tue = 0
    wed = 0
    thu = 0
    fri = 0
    sat = 0
    sun = 0
    if week_day == 0:
        mon = prob
    elif week_day == 1:
        tue = prob
    elif week_day == 2:
        wed = prob
    elif week_day == 3:
        thu = prob
    elif week_day == 4:
        fri = prob
    elif week_day == 5:
        sat = prob
    else:
        sun = prob
    return mon,tue,wed,thu,fri,sat,sun


def get_disrate_prob(discount,prob):
    manjian = 0
    disrate = 0
    if len(discount.split(":"))>= 2:
        manjian = prob
    else:
        disrate = prob
    return manjian , disrate

def get_prob(offline,feat):
    distance_pos = offline['User_id'].groupby([offline[feat],offline['label']]).count().reset_index()
    distance_prob = []
    distance = []
    dis_uq = np.unique(offline[feat])
    for dist in dis_uq:
        neg = distance_pos['User_id'][(distance_pos[feat] == dist) & (distance_pos['label'] == 0)]
        total = distance_pos['User_id'][(distance_pos[feat] == dist)].sum()
        if neg.size > 0:
            prob = neg / total
            distance_prob.append(prob.values[0])
        else:
            distance_prob.append(0)
        distance.append(dist)
        
    dis_df = pd.DataFrame()
    dis_df[feat] = distance
    dis_df["prob"] = distance_prob
    return dis_df


def get_total_prob(offline):
    distance_pos = offline['User_id'].groupby(offline['label']).count().reset_index()
    neg = distance_pos['User_id'][ (distance_pos['label'] == 0)]
    total = distance_pos['User_id'].sum()
    prob = neg / total
    return prob.values[0]


distance_df = get_prob(offline,'distance')
disrate_df = get_prob(offline,'dis_rate')
week_df = get_prob(offline,'weekd')
merchant_df = get_prob(offline,'Merchant_id')
def  get_sample(base_sample):
    distance = []
    week = []
    merchant = []
    man_jian = []
    dis_rate = []
    monday = []
    tuesday = []
    wedsday = []
    thusday = []
    friday = []
    sataday = []
    sunday = []
    #base_sample.shape[0]
    #base_sample = base_sample.iloc[:30]
    for i in xrange(base_sample.shape[0]):
        if i % 10000 == 0:
            print(i)
        dist = offline['distance'][i]
        disrate = offline['dis_rate'][i]
        weekday = offline['weekd'][i]
        merch= offline['Merchant_id'][i]
        
        distance_prob = distance_df['prob'][(distance_df['distance'] == dist)]
        distance.append(distance_prob.values[0])
        
        week_prob = week_df['prob'][(week_df['weekd'] == weekday)].values[0]
        mon,tue,wed,thu,fri,sat,sun = get_week_prob(weekday,week_prob)
        monday.append(mon)
        tuesday.append(tue)
        wedsday.append(wed)
        thusday.append(thu)
        friday.append(fri)
        sataday.append(sat)
        sunday.append(sun)
        
        disrate_prob = disrate_df['prob'][(disrate_df['dis_rate'] == disrate)].values[0]
        manjian , disrate_ = get_disrate_prob(disrate,disrate_prob)
        man_jian.append(manjian)
        dis_rate.append(disrate_)
        
        merchant_prob = merchant_df['prob'][(merchant_df['Merchant_id'] == merch)].values[0]
        merchant.append(merchant_prob)
    base_sample['distance'] = distance
    base_sample['man_jian'] = man_jian
    base_sample['dis_rate'] = dis_rate
    base_sample['mon'] = monday
    base_sample['tue'] = tuesday
    base_sample['wed'] = wedsday
    base_sample['thu'] = thusday
    base_sample['fri'] = friday
    base_sample['sta'] = sataday
    base_sample['sun'] = sunday
    base_sample['merchant'] = merchant
    return base_sample
  

train_sample = get_sample(offline)
train_sample.to_csv('../data/sample_train_1025_2.csv',index=False)

test_sample = get_sample(test)
test_sample.to_csv('../data/sample_test_1025_2.csv',index=False)
