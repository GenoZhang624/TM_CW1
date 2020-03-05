import torch
import numpy as np
import pandas as pd

def get_accuracy_bow(model,loader):
    y_preds = list()
    y_real = list()
    with torch.no_grad():
        for x, y in loader:
            y_preds.extend(model(x).argmax(dim=1).numpy().tolist())
            y_real.extend(y)
    return np.sum(np.array(y_preds)==y_real)/len(y_real),y_real,y_preds

def get_accuracy_bilstm(model,loader):
    y_preds = list()
    y_real = list()
    with torch.no_grad():
        for x, y, lengths in loader:
            y_preds.extend(model(x,lengths).argmax(dim=1).numpy().tolist())
            y_real.extend(y)
    return np.sum(np.array(y_preds)==y_real)/len(y_real),y_real,y_preds

def get_accuracy_test(model,model_type,x,y,lengths):
    with torch.no_grad():
        if model_type=='bow':
            y_preds = model(x).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y)
        if model_type=='bilstm':
            y_preds = model(x,lengths).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y)

def get_confusion_matrix(y_real,y_preds,size):
    mat = np.zeros( (size,size) )
    for i in range(len(y_real)):
        if y_real[i]==y_preds[i]:
            mat[y_real[i]-1][y_real[i]-1] += 1
        else:
            mat[y_real[i]-1][y_preds[i]-1] += 1
    return pd.DataFrame(mat)

def get_micro_f1(conf_mat):
    mat = np.array(conf_mat)
    tp,fp,fn = list(),list(),list()
    for i in range(np.size(mat,0)):
        tp.append(mat[i][i])
        fp.append(np.sum(mat[:][i]) - mat[i][i] )
        fn.append(np.sum(mat[i][:]) - mat[i][i] )
    tp_sum = np.sum(np.array(tp))
    fp_sum = np.sum(np.array(fp))
    fn_sum = np.sum(np.array(fn))
    precision = tp_sum/(tp_sum+fp_sum)
    recall = tp_sum/(tp_sum+fn_sum)
    return 2*precision*recall/(precision+recall)

def get_macro_f1(conf_mat):
    mat = np.array(conf_mat)
    precision,recall = list(),list()
    for i in range(np.size(mat,0)):
        tp=mat[i][i]
        fp=np.sum(mat[:][i]) - mat[i][i]
        fn=np.sum(mat[i][:]) - mat[i][i]
        if tp!=0 or fp!=0:
            precision.append(tp/(tp+fp))
        if tp!=0 or fn!=0:
            recall.append(tp/(tp+fn))
    pre_avg = np.mean(np.array(precision))
    rec_avg = np.mean(np.array(recall))
    return 2*pre_avg*rec_avg/(pre_avg+rec_avg)