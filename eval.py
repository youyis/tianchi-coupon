import numpy as np
from sklearn import metrics
import os
import pandas as pd

       
os.chdir("/data0/syy/coupon/tianchi-coupon")

coupon = np.unique(valid['Coupon_id'])
valid = pd.read_csv('../data/valid_out.csv')

def get_average_auc(result):
    auc_list =[]
    for i in xrange(coupon.shape[0]):
        y_true = valid['label'][valid['Coupon_id'] == coupon[i]]
        y_preds = valid['preds'][valid['Coupon_id'] == coupon[i]]
        if np.unique(y_true).size <= 1:
            continue
        auc = roc_auc_score(y_true,y_preds)
        auc_list.append(auc)
        print coupon[i], auc
    return np.average(auc_list)

get_average_auc(valid)
