from sklearn.preprocessing import StandardScaler,OneHotEncoder 
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from util import read_spam
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
tf.keras.backend.set_floatx('float64')

def kmean_spam():
    test_feature,test_label,train_feature,train_label=read_spam(101)
    kmeans = KMeans(n_clusters = 20).fit(train_feature)
    train_feature_new=kmeans.predict(train_feature)
    test_feature_new=kmeans.predict(test_feature)
    onehot=OneHotEncoder(sparse=False)
    train_feature_onehot=onehot.fit_transform(train_feature_new.reshape((-1,1)))
    test_feature_onehot=onehot.transform(test_feature_new.reshape((-1,1)))
    scaler=StandardScaler()
    train_feature_norm=scaler.fit_transform(train_feature_onehot)
    test_feature_norm=scaler.transform(test_feature_onehot)
    return test_feature_norm,test_label.reshape((-1,1)),train_feature_norm,train_label.reshape((-1,1))

def em_spam():
    test_feature,test_label,train_feature,train_label=read_spam(101)
    gmm=GaussianMixture(n_components=11).fit(train_feature)
    train_feature_new=gmm.predict(train_feature)
    test_feature_new=gmm.predict(test_feature)
    onehot=OneHotEncoder(sparse=False)
    train_feature_onehot=onehot.fit_transform(train_feature_new.reshape((-1,1)))
    test_feature_onehot=onehot.transform(test_feature_new.reshape((-1,1)))
    scaler=StandardScaler()
    train_feature_norm=scaler.fit_transform(train_feature_onehot)
    test_feature_norm=scaler.transform(test_feature_onehot)
    return test_feature_norm,test_label.reshape((-1,1)),train_feature_norm,train_label.reshape((-1,1))
    
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
    test_feature_km,test_label_km,train_feature_km,train_label_km=kmean_spam()
    test_feature_gmm,test_label_gmm,train_feature_gmm,train_label_gmm=em_spam()
    nn_train(test_feature_km,test_label_km,train_feature_km,train_label_km,name='kmeans spam')
    nn_train(test_feature_gmm,test_label_gmm,train_feature_gmm,train_label_gmm,name='gmm spam')

