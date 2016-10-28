import os
import pandas as pd
import numpy as np
       
os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample_1027.csv')
test = pd.read_csv('../data/base_sample_test.csv')
origin = pd.read_csv('../data/ccf_offline_stage1_train.csv')


origin = origin[["'User_id'","'Merchant_id'","'Date'"]][origin["'Date'"] != 'null']
merchant_freq = origin[["'User_id'"]].groupby(origin["'Merchant_id'"]).count().reset_index()
user_freq = origin[["'Merchant_id'"]].groupby(origin["'User_id'"]).count().reset_index()

#merchant_freq["'User_id'"][merchant_freq["'Merchant_id'"] == 1].values[0]


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


def get_discount_rate(discount):
    manjian_rate = 0
    disrate = 0
    man = 0
    jian = 0
    if len(discount.split(":"))>= 2:
        items = discount.split(":")
        man = items[0]
        jian = items[1]
        manjian_rate = 1 - float(jian) / float(man)
    else:
        disrate = float(discount)
    return man,jian, manjian_rate , disrate



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
        pos = distance_pos['User_id'][(distance_pos[feat] == dist) & (distance_pos['label'] == 1)]
        total = distance_pos['User_id'][(distance_pos[feat] == dist)].sum()
        if pos.size > 0:
            prob = pos / total
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
    pos = distance_pos['User_id'][ (distance_pos['label'] == 1)]
    total = distance_pos['User_id'].sum()
    prob = pos / total
    return prob.values[0]


distance_df = get_prob(offline,'distance')
disrate_df = get_prob(offline,'dis_rate')
week_df = get_prob(offline,'weekd')
merchant_df = get_prob(offline,'Merchant_id')
#print distance_df
#print disrate_df
#print week_df

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
    man_jan_rate = []
    dis_rate_rate = []
    man = []
    jian = []
    user = []
    #base_sample.shape[0]
    #base_sample = base_sample.iloc[:30]
    for i in xrange(base_sample.shape[0]):
        if i % 10000 == 0:
            print(i)
        dist = offline['distance'][i]
        disrate = offline['dis_rate'][i]
        weekday = offline['weekr'][i]
        merch= offline['Merchant_id'][i]
        userid = offline['User_id'][i]
        
        distance_prob = distance_df['prob'][(distance_df['distance'] == dist)]
        distance.append(distance_prob.values[0])
        week_prob = week_df['prob'][(week_df['weekd'] == str(weekday))].values[0]
      
        mon,tue,wed,thu,fri,sat,sun = get_week_prob(weekday,week_prob)
        monday.append(mon)
        tuesday.append(tue)
        wedsday.append(wed)
        thusday.append(thu)
        friday.append(fri)
        sataday.append(sat)
        sunday.append(sun)
        
        #disrate_prob = disrate_df['prob'][(disrate_df['dis_rate'] == disrate)].values[0]
        #manjian , disrate_ = get_disrate_prob(disrate,disrate_prob)
        #man_jian.append(manjian)
        #dis_rate.append(disrate_)
        
        man_,jian_, manjian_rate, disrate_rate = get_discount_rate(disrate)
        man.append(man_)
        jian.append(jian_)
        man_jan_rate.append(manjian_rate)
        dis_rate_rate.append(disrate_rate)
        
        
        #merchant_prob = merchant_df['prob'][(merchant_df['Merchant_id'] == merch)].values[0]
        #merchant.append(merchant_prob)
        merchant_prob = merchant_freq["'User_id'"][merchant_freq["'Merchant_id'"] == merch]
        if merchant_prob.size > 0 :
            merchant.append(merchant_prob.values[0])
        else:
            merchant.append(0)
            
        user_prob = user_freq["'Merchant_id'"][user_freq["'User_id'"] == userid]
        if user_prob.size > 0 :
            user.append(user_prob.values[0])
        else:
            user.append(0)   
            
    base_sample['distance'] = distance
    #base_sample['man_jian'] = man_jian
    #base_sample['dis_rate'] = dis_rate
    base_sample['man'] = man
    base_sample['jian'] = jian
    base_sample['man_jian_rate'] = man_jan_rate
    base_sample['dis_rate_rate'] = dis_rate_rate
    base_sample['mon'] = monday
    base_sample['tue'] = tuesday
    base_sample['wed'] = wedsday
    base_sample['thu'] = thusday
    base_sample['fri'] = friday
    base_sample['sta'] = sataday
    base_sample['sun'] = sunday
    base_sample['merchant'] = merchant
    base_sample['user'] = user
    return base_sample
  

#train_sample = get_sample(offline)
#train_sample.to_csv('../data/sample_train_1028.csv',index=False)

test_sample = get_sample(test)
test_sample.to_csv('../data/sample_test_1028.csv',index=False)

