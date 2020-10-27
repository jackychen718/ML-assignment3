from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA,FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from util import read_breast_cancer,read_spam
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
tf.keras.backend.set_floatx('float64')

def spam_PCA():
    test_feature,test_label,train_feature,train_label=read_spam()
    org_pca_train_feature=train_feature-train_feature.mean(axis=0)
    org_pca_test_feature=test_feature-train_feature.mean(axis=0)
    pca=PCA()
    pca_train_feature=pca.fit_transform(org_pca_train_feature)
    pca_test_feature=pca.transform(org_pca_test_feature)
    return pca_test_feature,test_label.reshape((-1,1)),pca_train_feature,train_label.reshape((-1,1))

def spam_ICA():
    test_feature,test_label,train_feature,train_label=read_spam()
    ica=FastICA(max_iter=1000)
    ica_train_feature=ica.fit_transform(train_feature)
    ica_test_feature=ica.transform(test_feature)
    return ica_test_feature,test_label.reshape((-1,1)),ica_train_feature,train_label.reshape((-1,1))

def spam_RP():
    test_feature,test_label,train_feature,train_label=read_spam()
    #Random Projection
    my_rp=GaussianRandomProjection(n_components=57)
    rp_train_feature=my_rp.fit_transform(train_feature)
    rp_test_feature=my_rp.transform(test_feature)
    return rp_test_feature,test_label.reshape((-1,1)),rp_train_feature,train_label.reshape((-1,1))

def spam_VT():
    test_feature,test_label,train_feature,train_label=read_spam()
    #feature selection    
    std_arr=train_feature.std(axis=0)
    n=len(std_arr)//4
    std_arr.sort()
    thresh=std_arr[n-1]
    my_vt=VarianceThreshold(threshold=thresh**2-1.e-6)
    vt_train_feature=my_vt.fit_transform(train_feature)
    vt_test_feature=my_vt.transform(test_feature)
    return vt_test_feature,test_label.reshape((-1,1)),vt_train_feature,train_label.reshape((-1,1))

def nn_train(test_feature,test_label,train_feature,train_label,name='original spam'):
    model=tf.keras.Sequential()
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1))
    train_iter=tf.data.Dataset.from_tensor_slices((train_feature,train_label)).shuffle(4000).batch(100)
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.15)
    loss_list=[]
    train_acc_list=[]
    test_acc_list=[]
    for epoch in range(100):
        epoch_loss=0
        for feature,label in train_iter:
            with tf.GradientTape() as tape:
                l_once=loss(label,model(feature,training=True))
            grads=tape.gradient(l_once,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            epoch_loss+=(l_once/40)
        loss_list.append(epoch_loss)
        train_pred=model(train_feature)>0.
        test_pred=model(test_feature)>0.
        train_acc_list.append(accuracy_score(train_label[:,0],train_pred[:,0]))
        test_acc_list.append(accuracy_score(test_label[:,0],test_pred[:,0]))
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(np.arange(1,101),loss_list)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('loss versus epochs for '+name)
    plt.subplot(122)
    plt.plot(np.arange(1,101),train_acc_list,label='train')
    plt.plot(np.arange(1,101),test_acc_list,label='test')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('accuracy versus epochs for '+name)
    plt.legend()
    plt.show()
    

if __name__=='__main__':
    test_feature,test_label,train_feature,train_label=read_spam()
    pca_test_feature,pca_test_label,pca_train_feature,pca_train_label=spam_PCA()
    ica_test_feature,ica_test_label,ica_train_feature,ica_train_label=spam_ICA()
    rp_test_feature,rp_test_label,rp_train_feature,rp_train_label=spam_RP()
    vt_test_feature,vt_test_label,vt_train_feature,vt_train_label=spam_VT()
    scaler=StandardScaler() 
    #original data
    train_feature=scaler.fit_transform(train_feature)
    test_feature=scaler.transform(test_feature)
    nn_train(test_feature,test_label.reshape((-1,1)),train_feature,train_label.reshape((-1,1)),name='original spam')
    #pca data
    train_feature=scaler.fit_transform(pca_train_feature)
    test_feature=scaler.transform(pca_test_feature)
    nn_train(test_feature,pca_test_label.reshape((-1,1)),train_feature,pca_train_label.reshape((-1,1)),name='pca spam')
    #ica data
    train_feature=scaler.fit_transform(ica_train_feature)
    test_feature=scaler.transform(ica_test_feature)
    nn_train(test_feature,ica_test_label.reshape((-1,1)),train_feature,ica_train_label.reshape((-1,1)),name='ica spam')
    #rp data
    train_feature=scaler.fit_transform(rp_train_feature)
    test_feature=scaler.transform(rp_test_feature)
    nn_train(test_feature,rp_test_label.reshape((-1,1)),train_feature,rp_train_label.reshape((-1,1)),name='rp spam')
    #my selection
    train_feature=scaler.fit_transform(vt_train_feature)
    test_feature=scaler.transform(vt_test_feature)
    nn_train(test_feature,vt_test_label.reshape((-1,1)),train_feature,vt_train_label.reshape((-1,1)),name='vt spam')



    

    



