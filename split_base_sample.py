import os
import pandas as pd
import numpy as np
       
os.chdir("/data0/syy/coupon/tianchi-coupon")
os.listdir(os.getcwd())
offline = pd.read_csv('../data/base_sample_1027.csv')
test = pd.read_csv('../data/base_sample_test.csv')
origin = pd.read_csv('../data/ccf_offline_stage1_train.csv')



base_sample_before = offline[offline.Date_received < 20160601]
base_sample_after = offline[offline.Date_received >= 20160601]
base_sample_before.to_csv('../data/base_sample_before.csv',index=False)
base_sample_after.to_csv('../data/base_sample_after.csv',index=False)
