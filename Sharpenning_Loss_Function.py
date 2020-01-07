# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:32:50 2020

@author: Sara
"""
from keras import backend as K

def Sharpenning_Loss(Lambda):

  def F_and_MAE(y_true, y_pred):
    
    beta2 = K.constant(0.3, dtype='float32')
    eps   = K.constant(1e-22, dtype='float32')
    
    
    y_true = K.cast(K.argmax(y_true,axis=-1), K.floatx())
    y_true = K.expand_dims(y_true,axis=-1)
    y_pred = y_pred[...,1:2]    

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    
    num_M_and_GT = K.sum(y_pred * y_true , axis=-1)
    num_M        = K.sum(y_pred , axis=-1)
    num_GT       = K.sum(y_true , axis=-1)
    
    prec_for_all_images   = num_M_and_GT / (num_M + eps)
    recall_for_all_images = num_M_and_GT / (num_GT+ eps)        
    
    
    Mean_Prec_all   = K.mean(prec_for_all_images)
    Mean_Recall_all = K.mean(recall_for_all_images)
    F_Loss = 1-(( (1+beta2)*Mean_Prec_all*Mean_Recall_all ) / ( (beta2*Mean_Prec_all)+Mean_Recall_all+eps ))
    
    return F_Loss + Lambda *K.mean(K.abs(y_pred-y_true), axis=-1) 

  return F_and_MAE 