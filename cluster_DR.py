from util import read_breast_cancer,read_spam
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA,FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

def breast_cancer_cluster_after_DR():
    test_feature,test_label,feature,label=read_breast_cancer()
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(feature)
    true_label=kmeans.predict(feature)
    
    #PCA feature
    org_pca_feature=feature-feature.mean(axis=0)
    pca=PCA()
    pca_feature=pca.fit_transform(org_pca_feature)
    
    #ICA feature
    ica=FastICA(max_iter=1000)
    ica_feature=ica.fit_transform(feature)
 
    #Random Projection
    my_rp=GaussianRandomProjection(n_components=9)
    rp_feature=my_rp.fit_transform(feature)
    
    #my selection
    std_arr=feature.std(axis=0)
    n=len(std_arr)//4
    std_arr.sort()
    thresh=std_arr[n-1]
    my_vt=VarianceThreshold(threshold=thresh**2-1.e-6)
    my_feature=my_vt.fit_transform(feature)
    
    #predict DR dataset
    pca_pred=kmeans.fit(pca_feature).predict(pca_feature)
    ica_pred=kmeans.fit(ica_feature).predict(ica_feature)
    rp_pred=kmeans.fit(rp_feature).predict(rp_feature)
    my_pred=kmeans.fit(my_feature).predict(my_feature)
    
    kmean_nmi_list=[]
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,pca_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,ica_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,rp_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,my_pred))
    
    '''
    we also need to use EM algorithm to analyze
    '''
    gmm_nmi_list=[]
    gmm=GaussianMixture(n_components=13)
    true_gmm=gmm.fit_predict(feature)
    
    pca_pred_gmm=gmm.fit_predict(pca_feature)
    ica_pred_gmm=gmm.fit_predict(ica_feature)
    rp_pred_gmm=gmm.fit_predict(rp_feature)
    my_pred_gmm=gmm.fit_predict(my_feature)
    
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,pca_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,ica_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,rp_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,my_pred_gmm))
    print(kmean_nmi_list)
    print(gmm_nmi_list)
    plt.bar(['pca','ica','rp','vt'],kmean_nmi_list)
    plt.ylabel('nmi')
    plt.title('NMI score for Kmean cluster')
    plt.show()
    plt.bar(['pca','ica','rp','vt'],gmm_nmi_list)
    plt.ylabel('nmi')
    plt.title('NMI score for GMM cluster')
    plt.show()
    

def spam_cluster_after_DR():
    test_feature,test_label,feature,label=read_spam()
    kmeans = KMeans(n_clusters = 20)
    kmeans.fit(feature)
    true_label=kmeans.predict(feature)
    
    #PCA feature
    org_pca_feature=feature-feature.mean(axis=0)
    pca=PCA()
    pca_feature=pca.fit_transform(org_pca_feature)
    
    #ICA feature
    ica=FastICA(max_iter=1000)
    ica_feature=ica.fit_transform(feature)
 
    #Random Projection
    my_rp=GaussianRandomProjection(n_components=57)
    rp_feature=my_rp.fit_transform(feature)
    
    #my selection
    std_arr=feature.std(axis=0)
    n=len(std_arr)//4
    std_arr.sort()
    thresh=std_arr[n-1]
    my_vt=VarianceThreshold(threshold=thresh**2-1.e-6)
    my_feature=my_vt.fit_transform(feature)
    
    #predict DR dataset
    pca_pred=kmeans.fit(pca_feature).predict(pca_feature)
    ica_pred=kmeans.fit(ica_feature).predict(ica_feature)
    rp_pred=kmeans.fit(rp_feature).predict(rp_feature)
    my_pred=kmeans.fit(my_feature).predict(my_feature)
    
    kmean_nmi_list=[]
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,pca_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,ica_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,rp_pred))
    kmean_nmi_list.append(normalized_mutual_info_score(true_label,my_pred))
    
    '''
    we also need to use EM algorithm to analyze
    '''
    gmm_nmi_list=[]
    gmm=GaussianMixture(n_components=11)
    true_gmm=gmm.fit_predict(feature)
    
    pca_pred_gmm=gmm.fit_predict(pca_feature)
    ica_pred_gmm=gmm.fit_predict(ica_feature)
    rp_pred_gmm=gmm.fit_predict(rp_feature)
    my_pred_gmm=gmm.fit_predict(my_feature)
    
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,pca_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,ica_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,rp_pred_gmm))
    gmm_nmi_list.append(normalized_mutual_info_score(true_gmm,my_pred_gmm))
    print(kmean_nmi_list)
    print(gmm_nmi_list)
    plt.bar(['pca','ica','rp','vt'],kmean_nmi_list)
    plt.ylabel('nmi')
    plt.title('NMI score for Kmean cluster')
    plt.show()
    plt.bar(['pca','ica','rp','vt'],gmm_nmi_list)
    plt.ylabel('nmi')
    plt.title('NMI score for GMM cluster')
    plt.show()

    
if __name__=='__main__':
    breast_cancer_cluster_after_DR()
    spam_cluster_after_DR()
    
    
    