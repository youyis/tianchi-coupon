import pandas as pd
import numpy as np
from sklearn.cross_validation  import  train_test_split
import xgboost as xgb


def get_params():
    params = {}
    params["objective"] = "reg:linear"
    params["booster"] = "gbtree"
    params["eta"] = 0.1
    params["min_child_weight"] = 1 
    #params["eval_meric"] = 'auc'
    params["subsample"] = 1 
    params["colsample_bytree"] = 1 
    params["silent"] = 1
    params["max_depth"] = 5 
    plst = list(params.items())
    return plst


# train sample
train_sample =  pd.read_csv('../data/label_id.csv')
drop_col = ['User_id','Merchant_id','Coupon_id','Date_received']
sample_drop = train_sample.drop(drop_col,axis=1)
feature,label = sample_drop.iloc[:,1:],sample_drop.iloc[:,0]
train_feat,valid_feat,train_label,valid_label = train_test_split(feature,label,test_size = 0.2, random_state=0)


#test sample
test_sample =  pd.read_csv('../data/test_sample.csv')
test_feat = test_sample.drop(drop_col,axis=1)



xgtrain = xgb.DMatrix(train_feat, train_label)
xgvalid = xgb.DMatrix(valid_feat, valid_label)
xgtest =  xgb.DMatrix(test_feat)


# xgboost parameter
plst = get_params()
plst += [('eval_metric', 'auc')]
print(plst)
watchlist = [(xgvalid, 'valid'),(xgtrain,'train')]
xgb_num_rounds = 70
model = xgb.train(params = plst, dtrain = xgtrain, evals = watchlist,num_boost_round = xgb_num_rounds)

# feature importance 
importance = model.get_fscore()
tuples = [(k, importance[k]) for k in importance]
tuples = sorted(tuples, key=lambda x: -x[1])
for k,v in tuples:
    print '%s\t%d' % (k,v)
    
    
#xgb.plot_importance(model)
valid_preds = model.predict(xgvalid, ntree_limit=model.best_iteration)
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)


preds_out = pd.DataFrame()
preds_out['User_id'] = test_sample['User_id']
preds_out['Merchant_id']  =  test_sample['Merchant_id'] 
preds_out['Coupon_id'] =  test_sample['Coupon_id']
preds_out['Date_received'] =  test_sample['Date_received']
preds_out['preds'] = test_preds

preds_res = preds_out['preds'].groupby([preds_out['User_id'],preds_out['Coupon_id'],preds_out['Date_received']]).max()
preds_out.to_csv('../data/preds_out.csv',index=False)

preds_res.to_csv('../data/preds_res.csv',index=True)


print(test_preds)

