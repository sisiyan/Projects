#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:28:33 2018

@author: syan
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:11:02 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, cluster_f1score, myGMM,nn_arch,nn_reg, plot_KMcluster, plot_EMcluster
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


out = "./Part3/ICA/"
np.random.seed(0)

loans = pd.read_csv('./lending_club_scaled.csv')
loans_X = loans.drop('safe_loans',1).copy().values
loans_Y = loans['safe_loans'].copy().values

blocks = pd.read_csv('./page-blocks.csv')
class1 = blocks[blocks['Class']==1]
class1 = class1.sample(frac = 0.1, random_state = 1)
blocks_balanced = class1
for n in range(2,6):
    blocks_balanced = blocks_balanced.append(blocks[blocks['Class']==n]) 
blocks_X = blocks_balanced.drop('Class',1).copy().values
blocks_Y = blocks_balanced['Class'].copy().values
blocks_X= StandardScaler().fit_transform(blocks_X)
print blocks_X.shape
#%% ICA for Part 2 for Loans

dim = 22
ica = FastICA(n_components=dim,random_state=10)

loansX2 = ica.fit_transform(loans_X)
loans2 = pd.DataFrame(np.hstack((loansX2,np.atleast_2d(loans_Y).T)))
cols = list(range(loans2.shape[1]))
cols[-1] = 'Class'
loans2.columns = cols
#loans2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

#%% Clustering on ICA transformed data
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

clusters =  [2,3,4,5,8,12,15,18,21,25]

loans_km_acc = []
loans_gmm_acc = []
loans_km_score = []
loans_gmm_score = []
loans_km_ami = []
loans_gmm_ami = []
loans_km_silhouette = []
loans_gmm_silhouette = []

for k in clusters:
    km.set_params(n_clusters=k)
    km.fit(loansX2)
    loans_km_acc.append(cluster_acc(loans_Y,km.predict(loansX2)))
    loans_km_score.append(km.score(loansX2))
    loans_km_ami.append(ami(loans_Y,km.predict(loansX2)))
    loans_km_silhouette.append(silhouette_score(loansX2, km.predict(loansX2)))
    
    gmm.set_params(n_components=k)
    gmm.fit(loansX2)
    loans_gmm_acc.append(cluster_acc(loans_Y,gmm.predict(loansX2)))
    loans_gmm_score.append(gmm.score(loansX2))
    loans_gmm_ami.append(ami(loans_Y,gmm.predict(loansX2)))
    loans_gmm_silhouette.append(silhouette_score(loansX2, gmm.predict(loansX2)))
    
loans_df= pd.DataFrame({'Kmeans acc': loans_km_acc, 'GMM acc': loans_gmm_acc,\
           'Kmeans score': loans_km_score, 'GMM score': loans_gmm_score,\
           'Kmeans ami': loans_km_ami, 'GMM ami': loans_gmm_ami,\
           'km avg silhouette': loans_km_silhouette, 'GMM avg silhouette':loans_gmm_silhouette },\
           index = clusters)
loans_df.to_csv(out+'clustering_loans_ica.csv')

ax = loans_df.plot(y=['Kmeans score'],\
                      title = "K-means Clustering: SSE",\
                      xlim = (0,25), style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("SSE")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part3_ica_SSEvsK.png')

ax = loans_df.plot(y=['GMM score'],\
                      title = "EM Clustering: LP",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("Log probability")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part3_ica_LPvsK.png')

ax = loans_df.plot(y=['km avg silhouette', 'GMM avg silhouette'],\
                      title = "Clustering: Average sihouette",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("The silhouette values")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part3_ica_silhouettevsK.png')

ax = loans_df.plot(y=['Kmeans ami', 'GMM ami'],\
                      title = "Clustering: Adjusted mutual info score",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("AMI")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part3_ica_AMIvsK.png')

ax = loans_df.plot(y=['Kmeans acc', 'GMM acc'],\
                      title = "ICA + Clustering: Accuracy for Risky/Safe Loans",\
                      xlim = (0,25), ylim=(0.5, 0.68), style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("Accuracy")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part3_ica_ACCvsK.png')
plt.close('all')
#%%Plot clusters for Loans

tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(loansX2)

n_clusters = 18
km = kmeans(random_state=5, n_clusters= n_clusters)
km.fit(loansX2)
cluster_labels = km.predict(loansX2)

colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Loans ICA k-means, k = 18")        
plt.savefig(out+"loans_ICA_km_clusters.png", dpi = 300)      


n_clusters = 18
gmm = GMM(random_state=5, n_components= n_clusters) 
gmm.fit(loansX2)
cluster_labels = gmm.predict(loansX2)
colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Loans ICA EM, k = 18")             
plt.savefig(out+"loans_ICA_em_clusters.png", dpi = 300)      
 

#%% ICA transform Pageblocks


dim = 4
ica = FastICA(n_components=dim,random_state=10)
blocksX2 = ica.fit_transform(blocks_X)
blocks2 = pd.DataFrame(np.hstack((blocksX2,np.atleast_2d(blocks_Y).T)))
cols = list(range(blocks2.shape[1]))
cols[-1] = 'Class'
blocks2.columns = cols
#blocks2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)

#%% Clustering on Pageblocks
clusters2 =  [5,6,7,8,10,12,15,20,25,30,35,40]

blocks_km_f1 = []
blocks_gmm_f1 = []
blocks_km_score = []
blocks_gmm_score = []
blocks_km_ami = []
blocks_gmm_ami = []
blocks_km_silhouette = []
blocks_gmm_silhouette = []


for k in clusters2:
    km.set_params(n_clusters=k)
    km.fit(blocksX2)
    blocks_km_f1.append(cluster_f1score(blocks_Y,km.predict(blocksX2)))
    blocks_km_score.append(km.score(blocksX2))
    blocks_km_ami.append(ami(blocks_Y,km.predict(blocksX2)))
    blocks_km_silhouette.append(silhouette_score(blocksX2, km.predict(blocksX2)))
    
    gmm.set_params(n_components=k)
    gmm.fit(blocksX2)
    blocks_gmm_f1.append(cluster_f1score(blocks_Y,gmm.predict(blocksX2)))
    blocks_gmm_score.append(gmm.score(blocksX2))
    blocks_gmm_ami.append(ami(blocks_Y,gmm.predict(blocksX2)))
    blocks_gmm_silhouette.append(silhouette_score(blocksX2, gmm.predict(blocksX2)))
    
    
blocks_df_pca= pd.DataFrame({'Kmeans f1': blocks_km_f1, 'GMM f1': blocks_gmm_f1,\
           'Kmeans score': blocks_km_score, 'GMM score': blocks_gmm_score,\
           'Kmeans ami': blocks_km_ami, 'GMM ami': blocks_gmm_ami,\
           'Kmeans avg silhouette': blocks_km_silhouette, 'GMM avg silhouette': blocks_gmm_silhouette},\
           index = clusters2)
blocks_df_pca.to_csv(out+'clustering_pageBlocks_pca.csv')

#%% Plot for scores
ax = blocks_df_pca.plot(y=['Kmeans score'],\
                      title = "Pageblock K-means Clustering ICA: SSE",\
                      style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("SSE")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblock_part3_ica_SSEvsK.png')

ax = blocks_df_pca.plot(y=['GMM score'],\
                      title = "Pageblock EM Clustering ICA: LP",\
                      style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("Log probability")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblock_part3_ica_LPvsK.png')

ax = blocks_df_pca.plot(y=['Kmeans avg silhouette', 'GMM avg silhouette'],\
                      title = "Pageblock Clustering ICA: Average sihouette",\
                      style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("The silhouette values")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblock_part3_ica_silhouettevsK.png')

ax = blocks_df_pca.plot(y=['Kmeans ami', 'GMM ami'],\
                      title = "Pageblock Clustering ICA: Adjusted mutual info score",\
                      style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("AMI")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblock_part3_ica_AMIvsK.png')

ax = blocks_df_pca.plot(y=['Kmeans f1', 'GMM f1'],\
                      title = "ICA + Clustering: Accuracy for Pageblocks",\
                      ylim= (0.5, 0.95), style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("f1 weighted score")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblock_part3_ica_ACCvsK.png')
plt.close('all')
#%% Plot clusters
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(blocksX2)
n_clusters = 8
km = kmeans(random_state=5, n_clusters= n_clusters)
km.fit(blocksX2)
cluster_labels = km.predict(blocksX2)

colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")  
plt.title("Pageblocks ICA k-means, k = 8")           
plt.savefig(out+"Pageblocks_ICA_km_clusters.png", dpi = 300)  

n_clusters = 25
gmm = GMM(random_state=5, n_components= n_clusters) 
gmm.fit(blocksX2)
cluster_labels = gmm.predict(blocksX2)

colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Pageblocks ICA EM, k = 25")           
plt.savefig(out+"Pageblocks_ICA_em_clusters.png", dpi = 300)      
#%% Part 5

out = "./Part5/"
dim2 = 4
ica = FastICA(n_components=dim2,random_state=10)
blocksX_ica = ica.fit_transform(blocks_X)


km = kmeans(random_state=5, n_clusters=8)
km.fit(blocksX_ica)
cluster_labels = km.predict(blocksX_ica)
cluster_labels = pd.get_dummies(cluster_labels)
blocksX_km = np.hstack((blocksX_ica,cluster_labels))

grid ={'NN__alpha':nn_reg}   
mlp = MLPClassifier(solver='lbfgs', activation='logistic', 
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
start = time()
gs.fit(blocksX_km,blocks_Y)
print "Run time: %.2f seconds" %(time() - start)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_ica_km_NN.csv')

blocksX_ori_km = np.hstack((blocks_X,cluster_labels))
grid ={'NN__alpha':nn_reg}   
mlp = MLPClassifier(solver='lbfgs', activation='logistic', 
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
start = time()
gs.fit(blocksX_ori_km,blocks_Y)
print "Run time: %.2f seconds" %(time() - start)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_ori_ica_km_NN.csv')


gmm = GMM(random_state=5, n_components = 25)
gmm.fit(blocksX_ica)
cluster_labels = gmm.predict(blocksX_ica)
cluster_labels = pd.get_dummies(cluster_labels)
blocksX_gm = np.hstack((blocksX_ica,cluster_labels))

grid ={'NN__alpha':nn_reg}   
mlp = MLPClassifier(solver='lbfgs', activation='logistic', 
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
start = time()
gs.fit(blocksX_gm,blocks_Y)
print "Run time: %.2f seconds" %(time() - start)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_ica_em_NN.csv')

blocksX_ori_gm = np.hstack((blocks_X,cluster_labels))
grid ={'NN__alpha':nn_reg}   
mlp = MLPClassifier(solver='lbfgs', activation='logistic', 
                     hidden_layer_sizes=(50,),
                    max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
start = time()
gs.fit(blocksX_ori_gm,blocks_Y)
print "Run time: %.2f seconds" %(time() - start)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Pageblocks_ori_ica_em_NN.csv')