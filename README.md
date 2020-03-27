# DP-WMM
Variational Inference Hierarchical Dirichlet process Mixture Models of Watson Distributions
---

### File
datas  # container of data  
result # container figure of dataset  
config # the hyper parameters of dataset  
model # dp-wmm model code  
utils # some util function  
train # training code  

---
### Requirements
matplotlib==3.1.1  
numpy==1.16.4  
pandas==0.23.2  
scipy==1.1.0  
sklearn==0.21.3  

---
### Run hdp-wmm
__params:__  
-name dataset name  
-lp Load hyper parameter or not 
-verbose print information or not  

-k first truncation of model  
-t second truncation of model  
-tau stick hyper params of fist level  
-gamma stick hyper params of second level  
-th second level threshold of converge   
-mth the threshold of Cluster number  
-sm second level max iteration  
-m max iterations of training  

__example:__  
python train.py -name big_data -lp 1 -verbose 1 -k 8 -t 50 -tau 1 -gamma 1 -th 1e-7 -mth 0.01 -sm -1 -m 700
