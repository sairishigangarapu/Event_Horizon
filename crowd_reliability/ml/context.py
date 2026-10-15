"""Causal topology features and training-only cluster routing for tree experts."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
CONTEXT_VERSION='observed_topology_v1'

def context_matrix(base,graph):
    base=np.asarray(base,dtype=np.float32)
    if base.shape!=(graph.num_nodes,18):raise ValueError('Context requires [zones,18] observed features.')
    # Neighbor values are weighted by physical edge width/distance AND presence masks.
    sums=np.zeros((len(base),6));weights=np.zeros_like(sums)
    edge_summary=np.zeros((len(base),4));degree=np.zeros(len(base))
    for ei,(u,v) in enumerate(graph.edge_index.T.tolist()):
        attrs=graph.edge_attr[ei].numpy();weight=max(.01,float(attrs[0]))/(1+max(0,float(attrs[1])))
        mask=base[u,12:18];sums[v]+=base[u,:6]*mask*weight;weights[v]+=mask*weight
        edge_summary[v]+=attrs;degree[v]+=1
    neighbor=np.divide(sums,weights,out=np.zeros_like(sums),where=weights>0)
    coverage=(weights>0).astype(float)
    edge_summary/=np.maximum(1,degree[:,None])
    return np.concatenate([base,neighbor,coverage,graph.node_static.numpy(),edge_summary],axis=1).astype(np.float32)

class ContextExpertRouter:
    def fit(self,X,targets,n_clusters=3,seed=42,min_samples=40,max_depth=6):
        from crowd_reliability.ml.tree_bank import DecisionTreeBank
        X=np.asarray(X);self.schema=CONTEXT_VERSION
        if len(X)<2 or not np.isfinite(X).all():raise ValueError('Need finite context training samples.')
        self.scaler=StandardScaler().fit(X)
        k=min(n_clusters,len(np.unique(X,axis=0)),max(1,len(X)//min_samples))
        self.clusterer=KMeans(n_clusters=k,random_state=seed,n_init=10).fit(self.scaler.transform(X))
        self.radii={};self.experts={};self.fallback=DecisionTreeBank(max_depth,seed);self.fallback.fit(X,targets)
        distances=self.clusterer.transform(self.scaler.transform(X))
        for cluster in range(k):
            mask=self.clusterer.labels_==cluster
            self.radii[cluster]=float(np.quantile(distances[mask,cluster],.99))+1e-6
            if mask.sum()>=min_samples:
                bank=DecisionTreeBank(max_depth,seed);bank.fit(X[mask],{key:val[mask] for key,val in targets.items()});self.experts[cluster]=bank
        return self
    def predict(self,X,exit_flags):
        scaled=self.scaler.transform(X);dist=self.clusterer.transform(scaled);ids=dist.argmin(1)
        values=[];masks=[];routed=[]
        for i,cluster in enumerate(ids):
            unknown=dist[i,cluster]>self.radii[int(cluster)]
            bank=self.fallback if unknown else self.experts.get(int(cluster),self.fallback)
            value,mask=bank.predict_zone(X[i],bool(exit_flags[i]));values.append(value);masks.append(mask);routed.append(-1 if unknown else int(cluster))
        return np.array(values),np.array(masks),routed
