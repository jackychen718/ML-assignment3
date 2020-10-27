from sklearn.manifold import TSNE
import numpy as np
from util import read_breast_cancer,read_spam
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold


def PCA_DR(name='breast cancer'):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    feature=feature-feature.mean(axis=0)
    pca=PCA()
    new_feature=pca.fit_transform(feature)
    plt.bar(np.arange(1,len(feature[0])+1),pca.singular_values_/pca.singular_values_.sum())
    plt.title('percentile of sigular value distribution')
    plt.show()
    pos_feature=new_feature[label==1]
    neg_feature=new_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.legend()
    plt.title(name+' two dimension display using PCA')
    plt.show()
    return new_feature,label

def TSNE_PCA(name='breast cancer',perplexity=250):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    feature=feature-feature.mean(axis=0)
    pca=PCA()
    new_feature=pca.fit_transform(feature)
    print('data shape after PCA:',new_feature.shape)
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    tsne_feature=my_tsne.fit_transform(new_feature)
    print('iteration num:',my_tsne.n_iter_)
    pos_feature=tsne_feature[label==1]
    neg_feature=tsne_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('tsne display after PCA for '+name)
    plt.show()
        
    
def ICA_DR(name='breast cancer'):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    ica=FastICA(n_components=2,max_iter=1000)
    new_feature=ica.fit_transform(feature)
    print(new_feature.shape)
    #print(ica.n_iter_)
    print('mean kurtosis is:',kurtosis(new_feature).mean())
    pos_feature=new_feature[label==1]
    neg_feature=new_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('Two dimension display after ICA for '+name)
    plt.show()
    
    
def best_k_for_ica(name='breast cancer'):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    best_k=0
    n=len(feature[0])
    kur_list=[]
    for n_com in range(2,n+1):
        ica=FastICA(n_components=n_com,max_iter=1000)
        new_feature=ica.fit_transform(feature)
        kur=kurtosis(new_feature).mean()
        kur_list.append(kur)
    plt.plot(np.arange(2,n+1),kur_list)
    plt.xlabel('k value')
    plt.ylabel('kurtosis value')
    plt.title('mean kurtosis versus k value for '+name)
    plt.show()
    best_k=np.argmax(kur_list)+2
    print('best k for '+name+' dataset is:',best_k)
    
    
def TSNE_ICA(name='breast cancer',perplexity=250):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    ica=FastICA(max_iter=1000)
    new_feature=ica.fit_transform(feature)
    print('ICA iteration:',ica.n_iter_)
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    tsne_feature=my_tsne.fit_transform(new_feature)
    print('TSNE iteration:',my_tsne.n_iter_)
    pos_feature=tsne_feature[label==1]
    neg_feature=tsne_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('tsne display after ICA for '+name)
    plt.show()
    
def RP(name='breast cancer'):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    my_rp=GaussianRandomProjection(2,random_state=100)
    new_feature=my_rp.fit_transform(feature)
    pos_feature=new_feature[label==1]
    neg_feature=new_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('two dimension display \nusing Random Projection for '+name)
    plt.show()
    
def TSNE_RP(name='breast cancer',perplexity=250,n_dim=9):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    my_rp=GaussianRandomProjection(n_components=n_dim)
    new_feature=my_rp.fit_transform(feature)
    print('projection feature shape:',new_feature.shape)
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    tsne_feature=my_tsne.fit_transform(new_feature)
    print('TSNE iter:',my_tsne.n_iter_)
    pos_feature=tsne_feature[label==1]
    neg_feature=tsne_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('TSNE display using Random Projection for '+name)
    plt.show()
    
def VT(name='breast cancer'):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    std_arr=feature.std(axis=0)
    std_arr.sort()
    thresh=std_arr[-2]
    my_vt=VarianceThreshold(threshold=thresh**2-1.e-6)
    new_feature=my_vt.fit_transform(feature)
    pos_feature=new_feature[label==1]
    neg_feature=new_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('two dimension selection display for '+name)
    plt.show()

def TSNE_VT(name='breast cancer',perplexity=250):
    test_feature,test_label,feature,label=read_breast_cancer()
    if name=='spam':
        test_feature,test_label,feature,label=read_spam()
    std_arr=feature.std(axis=0)
    n=len(std_arr)//4
    std_arr.sort()
    thresh=std_arr[n-1]
    my_vt=VarianceThreshold(threshold=thresh**2-1.e-6)
    new_feature=my_vt.fit_transform(feature)
    print(new_feature.shape)
    my_tsne=TSNE(perplexity=perplexity,n_iter=10000)
    tsne_feature=my_tsne.fit_transform(new_feature)
    print('TSNE iter:',my_tsne.n_iter_)
    pos_feature=tsne_feature[label==1]
    neg_feature=tsne_feature[label==0]
    plt.scatter(pos_feature[:,0],pos_feature[:,1],color='r',label='pos')
    plt.scatter(neg_feature[:,0],neg_feature[:,1],color='b',label='neg')
    plt.title('TSNE display after feature selection for '+name)
    plt.show()
    

if __name__=='__main__':
    #PCA_DR(name='breast cancer')
    #PCA_DR(name='spam')
    #TSNE_PCA(name='breast cancer',perplexity=250)
    #TSNE_PCA(name='spam',perplexity=1200)
    #ICA_DR(name='breast cancer')
    #ICA_DR(name='spam')
    #best_k_for_ica(name='breast cancer')
    #best_k_for_ica(name='spam')
    #TSNE_ICA(name='breast cancer',perplexity=250)
    #TSNE_ICA(name='spam',perplexity=1200)
    #RP(name='breast cancer')
    #RP(name='spam')
    #TSNE_RP(name='breast cancer',perplexity=250,n_dim=9)
    #TSNE_RP(name='spam',perplexity=1200,n_dim=57)
    #VT(name='breast cancer')
    #VT(name='spam')
    TSNE_VT(name='breast cancer',perplexity=250)
    TSNE_VT(name='spam',perplexity=1200)
    
    