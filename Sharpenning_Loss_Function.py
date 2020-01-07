from keras import backend as K

'''Implementation of the proposed Sharpening Loss function.
   The Sharpening Loss results in sharper foreground objects and less blurry predictions.
   Moreover, as shown in the paper, the Sharpening Loss outperforms the Cross-entropy loss
   by a significant margin in saliency detetction task'''

def Sharpenning_Loss(Lambda):

  def L_F_and_L_MAE(y_true, y_pred):
    
    beta2 = K.constant(0.3, dtype='float32')
    eps   = K.constant(1e-22, dtype='float32')
    
    y_true = K.cast(K.argmax(y_true,axis=-1), K.floatx())
    y_true = K.expand_dims(y_true,axis=-1)
    y_pred = y_pred[...,1:2]    

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    
    P = K.sum(y_pred * y_true , axis=-1) / (K.sum(y_pred , axis=-1) + eps)
    R = K.sum(y_pred * y_true , axis=-1) / (K.sum(y_true , axis=-1) + eps)        
    
    L_F = 1-(((1+beta2)*K.mean(P)*K.mean(R)) / ((beta2*K.mean(P))+K.mean(R)+eps))
    
    L_MAE = K.mean(K.abs(y_pred-y_true), axis=-1)
    
    return L_F + Lambda * L_MAE

  return L_F_and_L_MAE 
