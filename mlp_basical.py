import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import scores
#用户数量，项目数量
user_count = 822
item_count = 435
sheet_names= ['no disrupt','disrupt']
writer = pd.ExcelWriter('mlp_index.xlsx')
z = 1
path = 1
#从excel文件中整理出Predict_,最终结果为(305784,)的ndarray
predict_ = pd.read_excel('y_predict_mlp.xlsx',index_col=0)
predict_ = predict_.values
predict_ = predict_.reshape((user_count*item_count,))
#从txt
# 从excel文件中整理出test,最终结果为(305784,)的ndarray
test = pd.read_excel('y_test_mlp1.xlsx',index_col=0)
instance_count = np.sum(np.sum(test))
test = test.values
test = test.reshape((user_count*item_count,))
auc_score = roc_auc_score(test, predict_)
print('AUC:     {}'.format(auc_score))
# Top-K evaluation
topks = [1,5,10,20]
MRRS,HRS,Precs,Recas,NDCGs = [auc_score],[],[],[],[]
for topk in topks:
    MRR,HR,Prec,Reca,NDCG = scores_Ridge.topK_scores(test, predict_, topk, user_count, item_count,instance_count)
    MRRS.append(MRR),HRS.append(HR),Precs.append(Prec),Recas.append(Reca),NDCGs.append(NDCG)
index = pd.DataFrame(MRRS+HRS+Precs+Recas+NDCGs).T
index.columns = ['AUC', 'MRR@1', 'MRR@5', 'MRR@10', 'MRR@20', 'HR@1', 'HR@5', 'HR@10', 'HR@20', 'Prec@1', 'Prec@5',
                     'Prec@10', 'Prec@20', 'Reca@1', 'Reca@5', 'Reca@10', 'Reca@20', 'NDCG@1', 'NDCG@5', 'NDCG@10',
                     'NDCG@20']
index.to_excel(writer)
writer._save()
print(index)

