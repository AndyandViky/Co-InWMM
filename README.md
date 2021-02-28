# CDP-WMM
Paper for: Axial Data Modeling with Collapsed Nonparametric Watson Mixture Models and Its Application to Depth Image Analysis
---

### File
    datas   # container of data  
    result  # container figure of dataset  
    config  # the hyper parameters of dataset  
    model   # dp-wmm model code  
    utils   # some util function  
    train   # training code  

---
### Requirements
    matplotlib ==3.1.1  
    numpy      ==1.16.4  
    pandas     ==0.23.2  
    scipy      ==1.1.0  
    sklearn    ==0.21.3  

---
### Run cdp-wmm
__params:__  

    -name    dataset name  
    -lp      Load hyper parameter or not  
    -verbose print information or not  

    -t       truncation of model  
    -gamma   stick hyper params
    -mth     the threshold of Cluster number  
    -m       max iterations of training  

__example:__  
python train.py -name syn_data2 -lp 1 -verbose 1 -t 10 -gamma 1 -mth 0.01 -m 100

### Reference

    @InProceedings{YANG2020,
    author="Yang, Lin; Liu, Yuhang and Fan, Wentao",
    title="Axial Data Modeling with Collapsed Nonparametric Watson Mixture Models and Its Application to Depth Image Analysis",
    booktitle="Pattern Recognition and Computer Vision",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="17--28",
    }
