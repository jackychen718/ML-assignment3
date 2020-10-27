from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import numpy as np
from util import read_breast_cancer,read_spam
import matplotlib.pyplot as plt

def kmean_cluster(name='breast cancer',n_max=20):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    sse = []
    for k in range(1, n_max+1):
        kmeans = KMeans(n_clusters = k).fit(feature)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(feature)
        curr_sse = 0
        for i in range(len(feature)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.square(feature[i]-curr_center).sum()
        sse.append(curr_sse)
    #plt.figure(figsize=(20,10))
    plt.plot(np.arange(1,n_max+1),sse)
    plt.xlabel('k value')
    plt.xticks(np.arange(1,n_max+1))
    plt.ylabel('sse')
    plt.title('sse vs k value for '+name)
    plt.show()
    return sse

def tsne_show(name='breast cancer',perplexity=100):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    new_feature=my_tsne.fit_transform(feature)
    print('perplexity:',perplexity)
    print('iteration number:',my_tsne.n_iter_)
    new_feature_pos=new_feature[label==1]
    new_feature_neg=new_feature[label==0]
    plt.figure(figsize=(10,10))
    plt.scatter(new_feature_pos[:,0],new_feature_pos[:,1],color='r',label='pos')
    plt.scatter(new_feature_neg[:,0],new_feature_neg[:,1],color='b',label='neg')
    plt.legend()
    plt.title('tsne for '+name)
    plt.show()
    
def gaussian_mixture_cluster(name='breast cancer',n_max=20):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    bic_list=[]
    for k in range(1,n_max+1):
        gmm=GaussianMixture(n_components=k)
        gmm.fit(feature)
        bic_list.append(gmm.bic(feature))
    #plt.figure(figsize=(10,6))
    plt.plot(np.arange(1,n_max+1),bic_list)
    plt.xlabel('k value')
    plt.ylabel('bic score')
    plt.xticks(np.arange(1,n_max+1))
    plt.title('bic score for '+name)
    plt.show()
    return np.argmin(np.array(bic_list))+1,min(bic_list)

def cluster_tsne(name='breast cancer',k=3,perplexity=250):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    new_feature=my_tsne.fit_transform(feature)
    kmeans = KMeans(n_clusters=k).fit(feature)
    pred_clusters = kmeans.predict(feature)
    plt.figure(figsize=(10,10))
    for i in range(k):
        sub_feature=new_feature[pred_clusters==i]
        plt.scatter(sub_feature[:,0],sub_feature[:,1])
    plt.title('tsne display using k-mean with '+str(k)+' clusters'+' for '+name)
    plt.show()
        
def em_tsne(name='breast cancer',k=10,perplexity=250):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    new_feature=my_tsne.fit_transform(feature)
    gmm=GaussianMixture(n_components=k)
    pred=gmm.fit_predict(feature)
    plt.figure(figsize=(10,10))
    for i in range(k):
        sub_feature=new_feature[pred==i]
        plt.scatter(sub_feature[:,0],sub_feature[:,1])
    plt.title('TSNE display after EM '+str(k)+' cluster for '+name)
    plt.show()
    
    

if __name__=='__main__':
    kmean_cluster(n_max=20)
    kmean_cluster('spam',20)
    tsne_show(perplexity=250)
    tsne_show(name='spam',perplexity=1200)
    cluster_tsne(name='breast cancer',k=3,perplexity=250)
    cluster_tsne(name='spam',k=20,perplexity=1200)
    n_component,bic_val=gaussian_mixture_cluster(n_max=20)
    print('best K for bic score of breast cancer dataset',n_component,bic_val)
    n_component,bic_val=gaussian_mixture_cluster('spam',n_max=20)
    print('best K for bic score of spam dataset',n_component,bic_val)
    em_tsne(name='breast cancer',k=13,perplexity=250)
    em_tsne(name='spam',k=11,perplexity=1200)
    

        

    