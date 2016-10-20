import os
import pandas as pd
import numpy as np
from datetime import datetime
import datetime


def discount_rate(discount):
    manjian = 0
    disrate = 0
    if len(discount.split(":"))>= 2:
        items = discount.split(":")
        man = items[0]
        jian = items[1]
        manjian = float(jian) / float(man)
    else:
        disrate = float(discount)
    return manjian , disrate


def date2int(datestr):
    datestr = str(datestr)
    year = int(datestr[0:4])
    month = int(datestr[4:6])
    day = int(datestr[6:8])
    return year,month,day


def date_to_week2(datestr):
    mon = 0
    tue = 0
    wed = 0
    thu = 0
    fri = 0
    sat = 0
    sun = 0
    year,month,day = date2int(datestr) 
    dt= datetime.datetime(year,month,day)
    week_day =  datetime.datetime.weekday(dt)
    if week_day == 0:
        mon = 1
    elif week_day == 1:
        tue = 1
    elif week_day == 2:
        wed = 1
    elif week_day == 3:
        thu = 1
    elif week_day == 4:
        fri = 1
    elif week_day == 5:
        sat = 1
    else:
        sun = 1
    return mon,tue,wed,thu,fri,sat,sun



def date_to_week(datestr):
    mon = 0
    tue = 0
    wed = 0
    thu = 0
    fri = 0
    sat = 0
    sun = 0
    year,month,day = date2int(datestr) 
    dt= datetime.datetime(year,month,day)
    week_day =  datetime.datetime.weekday(dt)
    return week_day



def date_interval(date1,date2):
    year1,month1,day1 = date2int(date1)
    year2,month2,day2 = date2int(date2)
    d1 = datetime.datetime(year1,month1,day1)
    d2 = datetime.datetime(year2,month2,day2)
    interval = (d1 -d2).days
    return interval

offline = pd.read_csv('../data/ccf_offline_stage1_train.csv')
user = []
merchant =[]
coupon = []
reciver_date = []
man_jian = []
dis_rate = []
monday = []
tuesday = []
wedsday = []
thusday = []
friday = []
sataday = []
sunday = []
weekd = []
distance = []
label = []


for i in xrange(offline.shape[0]):
    if  offline["'Coupon_id'"][i] != 'null':
        user.append(offline["'User_id'"][i])
        merchant.append(offline["'Merchant_id'"][i])
        coupon.append(offline["'Coupon_id'"][i])
        reciver_date.append(offline["'Date_received'"][i])
        manjian,disrate = discount_rate(offline["'Discount_rate'"][i])
        man_jian.append(manjian)
        dis_rate.append(disrate)
        week = date_to_week(offline["'Date_received'"][i])
        weekd.append(week)
        #monday.append(mon)
        #tuesday.append(tue)
        #wedsday.append(wed)
        #thusday.append(thu)
        #friday.append(fri)
        #sataday.append(sat)
        #sunday.append(sun)
        if offline["'Distance'"][i] == 'null':
            distance.append('null')
        else:
            distance.append(float(offline["'Distance'"][i]))
            
        if  offline["'Date'"][i] != 'null' and date_interval(offline["'Date'"][i],offline["'Date_received'"][i]) <= 15 :
            label.append(1)
        else:
            label.append(0)
                              
                              

label_id = pd.DataFrame()
label_id['label'] = label
label_id['User_id'] = user
label_id['Merchant_id'] = merchant
label_id['Coupon_id'] = coupon
label_id['Date_received'] = reciver_date
label_id['man_jian'] = man_jian
label_id['dis_rate'] = dis_rate
label_id['weekd']  = weekd
#label_id['mon'] = monday
#label_id['tue'] = tuesday
#label_id['wed'] = wedsday
#label_id['thu'] = thusday
#label_id['fri'] = friday
#label_id['sta'] = sataday
#label_id['sun'] = sunday
label_id['distance'] = distance

label_id.to_csv('../data/base_sample.csv',index=False)
