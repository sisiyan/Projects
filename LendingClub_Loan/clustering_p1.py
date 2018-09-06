#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:56:41 2018

@author: syan
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, cluster_f1score, myGMM,plot_silhouette,plot_KMcluster,plot_EMcluster
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import sys

#%%
#out = './{}/'.format(sys.argv[1])
out = "./Part1/"
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

np.random.seed(0)
loans = pd.read_csv('./lending_club_scaled.csv')
loans_X = loans.drop('safe_loans',1).copy().values
loans_Y = loans['safe_loans'].copy().values


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
    km.fit(loans_X)
    loans_km_acc.append(cluster_acc(loans_Y,km.predict(loans_X)))
    loans_km_score.append(km.score(loans_X))
    loans_km_ami.append(ami(loans_Y,km.predict(loans_X)))
    loans_km_silhouette.append(silhouette_score(loans_X, km.predict(loans_X)))
    
    gmm.set_params(n_components=k)
    gmm.fit(loans_X)
    loans_gmm_acc.append(cluster_acc(loans_Y,gmm.predict(loans_X)))
    loans_gmm_score.append(gmm.score(loans_X))
    loans_gmm_ami.append(ami(loans_Y,gmm.predict(loans_X)))
    loans_gmm_silhouette.append(silhouette_score(loans_X, gmm.predict(loans_X)))
    
loans_df= pd.DataFrame({'Kmeans acc': loans_km_acc, 'GMM acc': loans_gmm_acc,\
           'Kmeans score': loans_km_score, 'GMM score': loans_gmm_score,\
           'Kmeans ami': loans_km_ami, 'GMM ami': loans_gmm_ami,\
           'km avg silhouette': loans_km_silhouette, 'GMM avg silhouette':loans_gmm_silhouette },\
           index = clusters)
loans_df.to_csv(out+'clustering_loans.csv')
#loans_score_df= pd.DataFrame(data = [loans_km_score, loans_gmm_score], index = clusters, columns = ['Kmeans', 'GMM'])

#%% Plot t-SNE plot for Loans

tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(loans_X)

n_clusters = 15
km = kmeans(random_state=5, n_clusters= n_clusters)
km.fit(loans_X)
cluster_labels = km.predict(loans_X)

colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Loans k-means, k = 15")        
plt.savefig(out+"loans_km_clusters.png", dpi = 300)      


n_clusters = 15
gmm = GMM(random_state=5, n_components= n_clusters) 
gmm.fit(loans_X)
cluster_labels = gmm.predict(loans_X)
colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Loans EM, k = 15")             
plt.savefig(out+"loans_em_clusters.png", dpi = 300)   

#%% Clustering on Pageblocks

blocks = pd.read_csv('page-blocks.csv')
# The dataset is not balance, almost 90% of the data is in Class 1.
# Sample 10% of data from Class1 to get a more balanced dataset     
class1 = blocks[blocks['Class']==1]
class1 = class1.sample(frac = 0.1, random_state = 1)
# Combine the sampled class1 and other classes
blocks_balanced = class1
for n in range(2,6):
    blocks_balanced = blocks_balanced.append(blocks[blocks['Class']==n]) 

blocks_X = blocks_balanced.drop('Class',1).copy().values
blocks_Y = blocks_balanced['Class'].copy().values

blocks_X= StandardScaler().fit_transform(blocks_X)

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
    km.fit(blocks_X)
    blocks_km_f1.append(cluster_f1score(blocks_Y,km.predict(blocks_X)))
    blocks_km_score.append(km.score(blocks_X))
    blocks_km_ami.append(ami(blocks_Y,km.predict(blocks_X)))
    blocks_km_silhouette.append(silhouette_score(blocks_X, km.predict(blocks_X)))
    
    gmm.set_params(n_components=k)
    gmm.fit(blocks_X)
    blocks_gmm_f1.append(cluster_f1score(blocks_Y,gmm.predict(blocks_X)))
    blocks_gmm_score.append(gmm.score(blocks_X))
    blocks_gmm_ami.append(ami(blocks_Y,gmm.predict(blocks_X)))
    blocks_gmm_silhouette.append(silhouette_score(blocks_X, gmm.predict(blocks_X)))
    
    
blocks_df= pd.DataFrame({'Kmeans f1': blocks_km_f1, 'GMM f1': blocks_gmm_f1,\
           'Kmeans score': blocks_km_score, 'GMM score': blocks_gmm_score,\
           'Kmeans ami': blocks_km_ami, 'GMM ami': blocks_gmm_ami,\
           'Kmeans avg silhouette': blocks_km_silhouette, 'GMM avg silhouette': blocks_gmm_silhouette},\
           index = clusters2)
blocks_df.to_csv(out+'clustering_pageBlocks.csv')

#%% Plot t-SNE plot for Pageblocks


tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(blocks_X)

n_clusters = 12
km = kmeans(random_state=5, n_clusters= n_clusters)
km.fit(blocks_X)
cluster_labels = km.predict(blocks_X)

colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Pageblock k-means, k = %d" % n_clusters)        
plt.savefig(out+"pageblock_km_clusters.png", dpi = 300)      


n_clusters = 6
gmm = GMM(random_state=5, n_components= n_clusters) 
gmm.fit(blocks_X)
cluster_labels = gmm.predict(blocks_X)
colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
plt.scatter(tsne_results[:,0], tsne_results[:,1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
plt.xlabel("x-tsne")
plt.ylabel("y-tsne")
plt.title("Pageblock EM, k = %d" % n_clusters)             
plt.savefig(out+"pageblock_em_clusters.png", dpi = 300)   


#%% Plot scores and accuracy

ax = loans_df.plot(y=['Kmeans score'],\
                      title = "K-means Clustering: SSE",\
                      xlim = (0,25), style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("SSE")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part1_SSEvsK.png')

ax = loans_df.plot(y=['GMM score'],\
                      title = "EM Clustering: LP",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("Log probability")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part1_LPvsK.png')

ax = loans_df.plot(y=['km avg silhouette', 'GMM avg silhouette'],\
                      title = "Clustering: Average sihouette",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("The silhouette values")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part1_SihouettEvsK.png')

ax = loans_df.plot(y=['Kmeans ami', 'GMM ami'],\
                      title = "Clustering: Adjusted mutual info score",\
                      xlim = (0,25),style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("AMI")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part1_AMIvsK.png')

ax = loans_df.plot(y=['Kmeans acc', 'GMM acc'],\
                      title = "Clustering: Accuracy for Risky/Safe Loans",\
                      xlim = (0,25), ylim= (0.5, 0.68), style = '.-')
ax.set_xlabel("k")
ax.set_ylabel("Accuracy")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Loans_part1_ACCvsK.png')

ax2 = blocks_df.plot(y=['Kmeans score'],\
                      title = "K-means Clustering: SSE",\
                      style = '.-')
ax2.set_xlabel("k")
ax2.set_ylabel("SSE")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblocks_part1_SSEvsK.png')

ax2 = blocks_df.plot(y=['GMM score'],\
                      title = "EM Clustering: LP",\
                      style = '.-')
ax2.set_xlabel("k")
ax2.set_ylabel("Log probability")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblocks_part1_LPvsK.png')

ax2 = blocks_df.plot(y=['Kmeans avg silhouette', 'GMM avg silhouette'],\
                      title = "Clustering: Average sihouette",\
                      style = '.-')
ax2.set_xlabel("k")
ax2.set_ylabel("The silhouette values")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblocks_part1_SihouettEvsK.png')

ax2 = blocks_df.plot(y=['Kmeans ami', 'GMM ami'],\
                      title = "Clustering: Adjusted mutual info score",\
                      style = '.-')
ax2.set_xlabel("k")
ax2.set_ylabel("AMI")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblocks_part1_AMIvsK.png')

ax2 = blocks_df.plot(y=['Kmeans f1', 'GMM f1'],\
                      title = "Clustering: Accuracy for Pageblocks",\
                      ylim= (0.5, 0.95), style = '.-')
ax2.set_xlabel("k")
ax2.set_ylabel("f1 weighted score")  
#ax.legend(bbox_to_anchor=(1.5, 1))
plt.savefig(out+'Pageblocks_part1_ACCvsK.png')

 

