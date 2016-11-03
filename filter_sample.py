import os
import pandas as pd
import numpy as np
       
os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample_1027.csv')
test = pd.read_csv('../data/base_sample_test.csv')
origin = pd.read_csv('../data/ccf_offline_stage1_train.csv')


offline = offline.drop_duplicates() 
filter_offline = offline['label'].groupby([offline['User_id'],offline['Merchant_id'],offline['Coupon_id'],offline['Date_received']]).max().reset_index()
label_train_offline = filter_offline[filter_offline.Date_received < 20160601]
label_valid_offline = filter_offline[filter_offline.Date_received >= 20160601]
offline.to_csv('../data/base_sample_filter_duplicate.csv',index=False)
label_train_offline.to_csv('../data/label_train_offline',index=True)
label_valid_offline.to_csv('../data/label_valid_offline',index=True)
