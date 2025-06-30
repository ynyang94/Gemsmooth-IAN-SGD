# import necessary packages
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pandas import DataFrame
import csv
from fractions import Fraction
from pandas.plotting import scatter_matrix
import scipy.stats
from scipy.stats.mstats import winsorize
import random
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
# from torchvision.io import read_image
# from torchvision.models import resnet18, ResNet18_Weights
import copy
import os
# import cvxpy as cp
import pdb

# import resnet18_file
# from resnet18_file import ResNet18

# Define function L and its gradient
# loss function and its expression of gradient.
def L_dL(X_train,y_train,X_test,y_test,xmodel,eta,batch,psi="chi-square",psi_grad="chi-square",\
         lambda0=0.01,is_grad=True,coeff_penalty=0.1,eps_penalty=1.0):
    if isinstance(psi, list):
        if psi[0]=="smoothCVaR":
            alpha=psi[1]
            def psi(t):                
                if type(t)==torch.Tensor:
                    rr=t.clone()
                    idp=(t>0)
                    idn=~idp
                    rr[idn]=torch.log(1-alpha+alpha*torch.exp(t[idn]))
                    rr[idp]=torch.log((1-alpha)*torch.exp(-t[idp])+alpha)+t[idp]
                    return rr/alpha
                else:
                    rr=t.copy()
                    idp=(t>0)
                    idn=~idp
                    rr[idn]=np.log(1-alpha+alpha*np.exp(t[idn]))
                    rr[idp]=np.log((1-alpha)*np.exp(-t[idp])+alpha)+t[idp]
                    return rr/alpha
            def psi_grad(t):
                if type(t)==torch.Tensor:
                    rr=t.clone()
                    idp=(t>0)
                    idn=~idp
                    expt=torch.exp(t[idn])
                    rr[idn]=expt/(1-alpha+alpha*expt)
                    expt=torch.exp(-t[idp])
                    rr[idp]=1/((1-alpha)*expt+alpha)
                else:
                    rr=t.copy()
                    idp=(t>0)
                    idn=~idp
                    expt=np.exp(t[idn])
                    rr[idn]=expt/(1-alpha+alpha*expt)
                    expt=np.exp(-t[idp])
                    rr[idp]=1/((1-alpha)*expt+alpha)
                return rr
    elif psi=="chi-square":
        def psi(t):
            if type(t)==np.ndarray:
                return np.maximum(t/2+1,np.zeros_like(t))**2-1
            else:
                return torch.maximum(t/2+1,torch.zeros_like(t))**2-1   
        def psi_grad(t):
            if type(t)==np.ndarray:
                return np.maximum(t/2+1,np.zeros_like(t))
            else:
                return torch.maximum(t/2+1,torch.zeros_like(t))
    elif psi=="KL":
        def psi(t):
            if type(t)==np.ndarray:
                return np.exp(t)-1
            else:
                return torch.exp(t)-1   
        def psi_grad(t):
            if type(t)==np.ndarray:
                return np.exp(t)
            else:
                return torch.exp(t)

    if type(batch)==np.ndarray:
        batch=batch.reshape(-1)

    R=(X_train[batch].dot(xmodel)-y_train[batch])
    l=(R**2)/2+coeff_penalty*np.log(1+np.abs(xmodel)/eps_penalty).sum()
    in_psi=(l-eta)/lambda0
    LL=lambda0*(psi(in_psi).mean())+eta
    #Log-sum penalty: https://arxiv.org/abs/2103.02681
    if is_grad:
        psi_grad_vec=psi_grad(in_psi)
        eta_grad=psi_grad_vec.mean()
        x_grad=((R*psi_grad_vec).reshape(-1,1)*X_train[batch]).mean(axis=0)\
            +coeff_penalty*eta_grad*np.sign(xmodel)/(np.abs(xmodel)+eps_penalty)
        eta_grad=1-eta_grad
        return LL,x_grad,eta_grad
    return LL
    
#Spider-DRO algorithm     
def Spider_DRO(X_train,y_train,X_test,y_test,total_iters,epoch_vt,epoch_momentum,gamma,S0,S01,S1,\
               beta=0,x0=None,eta0=0.1,normalize_power=1.0,grad_max=0,clip_constant=0,epoch_eval=1,psi="chi-square",\
                   psi_grad="chi-square",lambda0=0.01,eval_stepsize=0.1,eval_thr=1e-7,eval_maxiter=1e+4,is_eval_psi=True,\
                       penalty_hyps={"coeff_penalty":0.1,"eps_penalty":1.0},indepedent_sampling=False,print_progress=False):
    # X_train, y_train, X_test, y_test: train and test data
    # total_iters: iteration number, epoch_vt: gradient update (should set be 1 in usual case)
    # epoch_momemtum: epoch where acceleration is used, gamma: learning rate; S0, S1: batch size of spider. (S01: independet sample size)
    # beta: acceleration parameter; x0: initialization, normalize power: "beta" in our paper, a.k.a. normalization constant
    # eta0: dual variable initialization.
    # grad_max: cliping threshold. epoch_eval: the epoch number doing evaluation.
    # psi, psi_grad = DRO robust function.
    # lambda0: DRO objective hyper-parameter, refer to https://arxiv.org/abs/2110.12459 
    n_train,d=X_train.shape
    n_test=X_test.shape[0]  
    assert d==X_test.shape[1],"X_train and X_test should have the same dimensionality."
    assert n_train==y_train.shape[0], "X_train and y_train should have the same number of samples."
    assert n_test==y_test.shape[0], "X_test and y_test should have the same number of samples."  
    
    grad_max*=grad_max
    # choices of robust function.
    if isinstance(psi, list):
        if psi[0]=="smoothCVaR":
            alpha=psi[1]
            def psi(t):
                if type(t)==torch.Tensor:
                    rr=t.clone()
                    idp=(t>0)
                    idn=~idp
                    rr[idn]=torch.log(1-alpha+alpha*torch.exp(t[idn]))
                    rr[idp]=torch.log((1-alpha)*torch.exp(-t[idp])+alpha)+t[idp]
                else:
                    rr=t.copy()
                    idp=(t>0)
                    idn=~idp
                    rr[idn]=np.log(1-alpha+alpha*np.exp(t[idn]))
                    rr[idp]=np.log((1-alpha)*np.exp(-t[idp])+alpha)+t[idp]
                return rr/alpha
                
            def psi_grad(t):
                if type(t)==torch.Tensor:
                    rr=t.clone()
                    idp=(t>0)
                    idn=~idp
                    expt=torch.exp(t[idn])
                    rr[idn]=expt/(1-alpha+alpha*expt)
                    expt=torch.exp(-t[idp])
                    rr[idp]=1/((1-alpha)*expt+alpha)
                else:
                    rr=t.copy()
                    idp=(t>0)
                    idn=~idp
                    expt=np.exp(t[idn])
                    rr[idn]=expt/(1-alpha+alpha*expt)
                    expt=np.exp(-t[idp])
                    rr[idp]=1/((1-alpha)*expt+alpha)
                return rr
    elif psi=="chi-square":
        def psi(t):
            if type(t)==np.ndarray:
                return np.maximum(t/2+1,np.zeros_like(t))**2-1
            elif type(t)==torch.Tensor:
                return torch.maximum(t/2+1,torch.zeros_like(t))**2-1   
        def psi_grad(t):
            if type(t)==np.ndarray:
                return np.maximum(t/2+1,np.zeros_like(t))
            elif type(t)==torch.Tensor:
                return torch.maximum(t/2+1,torch.zeros_like(t))
    elif psi=="KL":
        def psi(t):
            if type(t)==np.ndarray:
                return np.exp(t)-1
            else:
                return torch.exp(t)-1   
        def psi_grad(t):
            if type(t)==np.ndarray:
                return np.exp(t)
            else:
                return torch.exp(t)

    if x0 is None:
        x0=np.random.normal(size=d)   #Initialize (primal) model parameter
    wt_x=copy.deepcopy(x0)
    wt_eta=eta0 # Initialize (dual) model parameter.
    L_set=[]
    Psi_set=[]
    # test_acc_set=[]
    iters_set=[]
    complexity_set=[]
    wold_x=wold_eta=mt_x=mt_eta=None
    complexity=0
    for k in range(total_iters):
        if k%epoch_eval==0:
            if print_progress:
                print("Evaluating "+str(k)+"-th iteration")
            grad=eval_thr+1
            
            R=(X_train.dot(wt_x)-y_train) #l2 loss
            l_full=(R**2)/2+penalty_hyps['coeff_penalty']*np.log(1+np.abs(wt_x)/penalty_hyps['eps_penalty']).sum() # l2 loss with penality
            #Log-sum penalty: https://arxiv.org/abs/2103.02681
            L_now=lambda0*np.mean(psi((l_full-wt_eta)/lambda0))+wt_eta # DRO dual objective: refer to https://arxiv.org/abs/2110.12459 
            # pdb.set_trace()
            L_set+=[L_now]
            if np.isnan(L_now) or np.isinf(L_now):
                return L_set,Psi_set,iters_set,complexity_set,wt_x,wt_eta
            # search for optimal eta.
            if is_eval_psi:
                eta_opt=wt_eta
                obj_min=np.inf
                eta_iter=0
                while abs(grad)>=eval_thr and eta_iter<=eval_maxiter:
                    # corresponds to $\mathbb{E}_{\xi}\Psi'((l(x)-\eta)/\lambda)
                    psi_input=(l_full-eta_opt)/lambda0
                    obj=lambda0*psi(psi_input).mean()+eta_opt
                    obj_min=min(obj,obj_min)
                    # gradient w.r.t eta.
                    grad=1-psi_grad(psi_input).mean()
                    eta_opt-=grad*eval_stepsize
                    if print_progress:
                        print("eta="+str(eta_opt)+"; grad="+str(grad)+"; obj="+str(obj))
                    eta_iter+=1
                Psi_set+=[obj_min]
                # print("L="+str(L_now)+", Psi="+str(obj_min)+", test accuracy="+str(test_acc))
            else:
                # print("L="+str(L_now)+", test accuracy="+str(test_acc))
                if print_progress:
                    print("L="+str(L_now))
            iters_set+=[k]
            complexity_set+=[complexity]
            if k%200 == 0:
                print('objective value at %d is %f'%(k,L_now))
            #End of evaluation

        # gradient update w.r.t x.
        if print_progress:
            print("Updating "+str(k)+"-th iteration")
        if indepedent_sampling == False:
            if k%epoch_vt==0:
                complexity+=S0
                batch=np.random.choice(n_train, S0, replace=False)
                _,vt_x,vt_eta=L_dL(X_train,y_train,X_test,y_test,wt_x,wt_eta,batch,psi,psi_grad,lambda0,is_grad=True,\
                                coeff_penalty=penalty_hyps['coeff_penalty'],eps_penalty=penalty_hyps['eps_penalty'])
            else:
                complexity+=S1
                batch=np.random.choice(n_train, S1, replace=False)
                _,gt_x,gt_eta=L_dL(X_train,y_train,X_test,y_test,wt_x,wt_eta,batch,psi,psi_grad,lambda0,is_grad=True,\
                               coeff_penalty=penalty_hyps['coeff_penalty'],eps_penalty=penalty_hyps['eps_penalty'])
                _,gold_x,gold_eta=L_dL(X_train,y_train,X_test,y_test,wold_x,wold_eta,batch,psi,psi_grad,lambda0,is_grad=True,\
                               coeff_penalty=penalty_hyps['coeff_penalty'],eps_penalty=penalty_hyps['eps_penalty'])            

                vt_x+=gt_x-gold_x
                vt_eta+=gt_eta-gold_eta # SPIDER update
        # I-NSGD here
        else:
            avaliable_indices = np.arange(n_train)
            if k%epoch_vt==0:
                complexity+=S0
                batch=np.random.choice(n_train, S0, replace=False)
                _,vt_x,vt_eta=L_dL(X_train,y_train,X_test,y_test,wt_x,wt_eta,batch,psi,psi_grad,lambda0,is_grad=True,\
                                coeff_penalty=penalty_hyps['coeff_penalty'],eps_penalty=penalty_hyps['eps_penalty'])
                complexity += S01
                independet_batch = np.random.choice(avaliable_indices, S01, replace=False)
                _,vt_x_independent,vt_eta_independent = L_dL(X_train,y_train,X_test,y_test,wt_x,wt_eta,independet_batch,psi,psi_grad,lambda0,is_grad=True,\
                                coeff_penalty=penalty_hyps['coeff_penalty'],eps_penalty=penalty_hyps['eps_penalty'])
        
        if (beta==0) | (k%epoch_momentum==0):
            mnext_x=vt_x.copy()
            mnext_eta=vt_eta # 1 dimension
        else:
            mnext_x=beta*mt_x+(1-beta)*vt_x
            mnext_eta=beta*mt_eta+(1-beta)*vt_eta # acceleration.
        
        wold_x=copy.deepcopy(wt_x)
        wold_eta=wt_eta
        mt_x=mnext_x.copy()
        mt_eta=mnext_eta
        if indepedent_sampling == True:
            mnext_x_independent = vt_x_independent.copy()
            mnext_eta_independent = vt_eta_independent # 1 dimension.
        
        if normalize_power==0: # correponds to case like SGD.
            coeff=gamma
        else: # corresponds to case like normalized SGD, clipped SGD, 
            norm_sq=max(np.sqrt(np.sum(mnext_x**2)+mnext_eta**2)+clip_constant,grad_max)
            coeff=gamma/(norm_sq**(normalize_power))
            if indepedent_sampling == True: # tailored for indepedent normalized SGD.
                norm_sq=max(2*np.sqrt(np.sum(mnext_x_independent**2)+mnext_eta_independent**2)+clip_constant,grad_max)
                coeff=gamma/(norm_sq**(normalize_power))
        wt_x=wt_x-coeff*mnext_x
        wt_eta=wt_eta-coeff*mnext_eta
    return L_set,Psi_set,iters_set,complexity_set,wt_x,wt_eta
    # return L_set,Psi_set,test_acc_set,iters_set,complexity_set

def num2str_neat(num):
    a=Fraction(num)
    if abs(a.numerator)>100:
        a=Fraction(num).limit_denominator()
        return(str(a.numerator)+'/'+str(a.denominator))
    return str(num)

#https://www.telusinternational.com/insights/ai-data/article/10-open-datasets-for-linear-regression
# Find data for regresion


# Get data
#Life expectancy data: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
#Python code on life expectancy data: https://thecleverprogrammer.com/2021/01/06/life-expectancy-analysis-with-python/ 
life_expectancy = pd.read_csv("Life Expectancy Data.csv") #reading the file
life_expectancy.head()
life_expectancy.rename(columns = {" BMI " :"BMI", 
                              "Life expectancy ": "Life_expectancy",
                              "Adult Mortality":"Adult_mortality",
                              "infant deaths":"Infant_deaths",
                              "percentage expenditure":"Percentage_expenditure",
                              "Hepatitis B":"HepatitisB",
                              "Measles ":"Measles",
                              "under-five deaths ": "Under_five_deaths",
                              "Total expenditure":"Total_expenditure",
                              "Diphtheria ": "Diphtheria",
                              " thinness  1-19 years":"Thinness_1-19_years",
                              " thinness 5-9 years":"Thinness_5-9_years",
                              " HIV/AIDS":"HIV/AIDS",
                              "Income composition of resources":"Income_composition_of_resources"}, inplace = True)

#Fill in missing values with the corresponding column' median
life_expectancy.reset_index(inplace=True)
life_expectancy.groupby('Country').apply(lambda group: group.interpolate(method= 'linear'))
imputed_data = []
for year in list(life_expectancy.Year.unique()):
    year_data = life_expectancy[life_expectancy.Year == year].copy()
    for col in list(year_data.columns)[4:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().median()).copy()
    imputed_data.append(year_data)
life_expectancy = pd.concat(imputed_data).copy()

#winsorizing columns
life_expectancy = life_expectancy[life_expectancy['Infant_deaths'] < 1001]
life_expectancy = life_expectancy[life_expectancy['Measles'] < 1001]
life_expectancy = life_expectancy[life_expectancy['Under_five_deaths'] < 1001]

life_expectancy.drop(['BMI'], axis=1, inplace=True)
life_expectancy['log_Percentage_expenditure'] = np.log(life_expectancy['Percentage_expenditure'])
life_expectancy['log_Population'] = np.log(life_expectancy['Population'])
life_expectancy['log_GDP'] = np.log(life_expectancy['GDP'])
life_expectancy = life_expectancy.replace([np.inf, -np.inf], 0)
life_expectancy['log_Percentage_expenditure']

life_expectancy['winz_Life_expectancy'] = winsorize(life_expectancy['Life_expectancy'], (0.05,0))
life_expectancy['winz_Adult_mortality'] = winsorize(life_expectancy['Adult_mortality'], (0,0.04))
life_expectancy['winz_Alcohol'] = winsorize(life_expectancy['Alcohol'], (0.0,0.01))
life_expectancy['winz_HepatitisB'] = winsorize(life_expectancy['HepatitisB'], (0.20,0.0))
life_expectancy['winz_Polio'] = winsorize(life_expectancy['Polio'], (0.20,0.0))
life_expectancy['winz_Total_expenditure'] = winsorize(life_expectancy['Total_expenditure'], (0.0,0.02))
life_expectancy['winz_Diphtheria'] = winsorize(life_expectancy['Diphtheria'], (0.11,0.0))
life_expectancy['winz_HIV/AIDS'] = winsorize(life_expectancy['HIV/AIDS'], (0.0,0.21))
life_expectancy['winz_Thinness_1-19_years'] = winsorize(life_expectancy['Thinness_1-19_years'], (0.0,0.04))
life_expectancy['winz_Thinness_5-9_years'] = winsorize(life_expectancy['Thinness_5-9_years'], (0.0,0.04))
life_expectancy['winz_Income_composition_of_resources'] = winsorize(life_expectancy['Income_composition_of_resources'], (0.05,0.0))
life_expectancy['winz_Schooling'] = winsorize(life_expectancy['Schooling'], (0.03,0.01))

col_dict_winz = {'winz_Life_expectancy':1,'winz_Adult_mortality':2,'Infant_deaths':3,'winz_Alcohol':4,
            'log_Percentage_expenditure':5,'winz_HepatitisB':6,'Measles':7,'Under_five_deaths':8,'winz_Polio':9,
            'winz_Total_expenditure':10,'winz_Diphtheria':11,'winz_HIV/AIDS':12,'log_GDP':13,'log_Population':14,
            'winz_Thinness_1-19_years':15,'winz_Thinness_5-9_years':16,'winz_Income_composition_of_resources':17,
            'winz_Schooling':18}
X_train=np.array(life_expectancy)
y_train=X_train[:,4]
y_train=(y_train-y_train.mean())/y_train.std() 
X_train=np.delete(X_train, [1,3,4], axis=1)
X_train=X_train.astype(np.float64)
X_train=(X_train-(X_train.mean(axis=0).reshape(1,-1)))/(X_train.std(axis=0).reshape(1,-1))
n_train, d=X_train.shape

y_std=1.0
random.seed(1)
np.random.seed(1)
y_train=y_train.astype(np.float64)+np.random.normal(scale=y_std,size=n_train)

n_test=100
n_train-=n_test   #2313
X_test=X_train[n_train:(n_train+n_test)]
y_test=y_train[n_train:(n_train+n_test)]
X_train=X_train[0:n_train]
y_train=y_train[0:n_train]

#hyperparameters
num_exprs=1   #number of experiments
total_iters=50
total_iters_stoc=1001
gamma=0.05   #Stepsize
lambda0=0.01
psi="chi-square"
psi_grad="chi-square"  #["smoothCVaR",1]
penalty_hyps={"coeff_penalty":0.1,"eps_penalty":1.0}

epoch_eval=1
is_eval_psi=True
print_progress=True
eval_stepsize=0.1
eval_thr=1e-4
eval_maxiter=5000

colors=['red','black','blue','green','cyan','purple','lime','gold','darkorange']
#markers=['.','v','s','P','*','+','-','x','o']
markers=['P','v','*','s','.','+','x','-','o']
label_size=18
num_size=18
lgd_size=20
percentile=95
clip_constant = 25
clip_constant_1 = 40


random.seed(1)
np.random.seed(1)
n_train,d=X_train.shape
n_test=X_test.shape[0]
x0=np.random.normal(size=d,scale=1.0) #Initial model x
eta0=0.1                    #Initial eta

S1=128
S01=8
'''
GD type (Not used in this paper)
'''
hyps=[{'epoch_vt':1,'gamma':1e-4,'S0':n_train,'S1':None,'S01':None,'normalize_power':0,'clip_constant':0,'grad_max':0,'sampling':False,\
       'beta':0,'epoch_momentum':1,'legend':'GD','name':'GD'}]#.vsP*
hyps+=[{'epoch_vt':1,'gamma':0.2,'S0':n_train,'S1':None,'S01':None,'normalize_power':1,'clip_constant':0,'grad_max':0,'sampling':False,\
        'beta':0,'epoch_momentum':1,'legend':'Normalized GD','name':'1GD'}]
hyps+=[{'epoch_vt':1,'gamma':0.3,'S0':n_train,'S1':None,'S01':None,'normalize_power':1,'clip_constant':0,'grad_max':10.0,'sampling':False,\
        'beta':0,'epoch_momentum':1,'legend':'Clipped GD','name':'ClippedGD'}]
'''
SGD Algorithm
'''
hyps+=[{'epoch_vt':1,'gamma':4e-5,'S0':S1,'S1':None,'S01':None,'normalize_power':0,'clip_constant':0,'grad_max':0,'sampling':False,\
        'beta':0,'epoch_momentum':1,'legend':'SGD','name':'SGD'}]
"""
Normalized SGD
"""
hyps+=[{'epoch_vt':1,'gamma':0.005,'S0':S1,'S1':None,'S01':None,'normalize_power':2/3,'clip_constant':0,'grad_max':0,'sampling':False,\
        'beta':0,'epoch_momentum':1,'legend':'NSGD with '+r"$\beta = \frac{2}{3}$",'name':'NSGD'}]
"""
NSGDm
"""
hyps+=[{'epoch_vt':1,'gamma':0.005,'S0':S1,'S1':None,'S01':None,'normalize_power':2/3,'clip_constant':0,'grad_max':0,'sampling':False,'beta':1e-1,\
        'epoch_momentum':total_iters+9,'legend':'NSGDm with '+r"$\beta = \frac{2}{3}$" ,'name':'NormalizedSGDm'}]
'''
Clip SGD
'''
hyps+=[{'epoch_vt':1,'gamma':0.14,'S0':S1,'S1':None,'S01':None,'normalize_power':2/3,'clip_constant':clip_constant,'grad_max':35,'sampling':False,\
        'beta':0,'epoch_momentum':1,'legend':'Clip SGD with '+r"$\beta = \frac{2}{3}$",'name':'ClippedSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.1,'S0':S1,'S1':None,'S01':None,'normalize_power':2/3,'clip_constant':clip_constant,'grad_max':60,'sampling':False,\
        #'beta':0,'epoch_momentum':1,'legend':'Clip SGD with '+r'$\gamma = 0.1$','name':'Clip SGD'}]
        
"""
SPIDER
"""
hyps+=[{'epoch_vt':15,'gamma':0.003,'S0':n_train,'S1':S1,'S01':None,'normalize_power':2/3,'clip_constant':0,'grad_max':0, 'sampling':False,\
        'beta':0.25,'epoch_momentum':1,'legend':'SPIDER with '+r"$\beta = \frac{2}{3}$",'name':'SPIDER'}]
"""
INSGD
"""
hyps+=[{'epoch_vt':1,'gamma':0.14,'S0':S1,'S1':0,'S01':S01,'normalize_power':2/3,'clip_constant':clip_constant,'grad_max':35,'sampling':True,\
        'beta':0,'epoch_momentum':1,'legend':'I-NSGD with '+r'$\beta = \frac{2}{3}$','name':'INSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.16,'S0':S1,'S1':0,'S01':S01,'normalize_power':2/3,'clip_constant':clip_constant,'grad_max':60,'sampling':True,\
       # 'beta':0,'epoch_momentum':1,'legend':'I-NSGD with '+r'$\gamma = 0.16$','name':'INSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':0,'S01':S01,'normalize_power':1,'clip_constant':clip_constant,'grad_max':60,'sampling':True,\
       # 'beta':0,'epoch_momentum':1,'legend':'I-NSGD with ' + r'$\beta = 1$','name':'INSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':0,'S01':S01,'normalize_power':4/5,'clip_constant':clip_constant,'grad_max':60,'sampling':True,\
        #'beta':0,'epoch_momentum':1,'legend':'I-NSGD with ' + r'$\beta = \frac{4}{5}$','name':'INSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':0,'S01':S01,'normalize_power':7/10,'clip_constant':clip_constant,'grad_max':60,'sampling':True,\
        #'beta':0,'epoch_momentum':1,'legend':'I-NSGD with ' + r'$\beta = \frac{7}{10}$','name':'INSGD'}]
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':0,'S01':S01,'normalize_power':3/5,'clip_constant':clip_constant,'grad_max':60,'sampling':True,\
        #'beta':0,'epoch_momentum':1,'legend':'I-NSGD with ' + r'$\beta = \frac{3}{5}$','name':'INSGD'}]
"""
I-NSGD Ablation study on batch size
"""
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':None,'S01':8,'normalize_power':2/3,
        #'grad_max':60.0,'clip_constant':clip_constant,'beta':0,'epoch_momentum':1,'sampling':True,'legend':'I-NSGD with '+r"${B'}=8$"}]
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':None,'S01':16,'normalize_power':2/3,
        #'grad_max':60.0,'clip_constant':clip_constant,'beta':0,'epoch_momentum':1,'sampling':True,'legend':'I-NSGD with '+r"${B'}=16$" }] #"${B'}=64$,"
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':None,'S01':32,'normalize_power':2/3,
        #'grad_max':60.0,'clip_constant':clip_constant,'beta':0,'epoch_momentum':1,'sampling':True,'legend':'I-NSGD with '+r"${B'}=32$"}]#"${B'}=32$,"
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':None,'S01':64,'normalize_power':2/3,
        #'grad_max':60.0,'clip_constant':clip_constant,'beta':0,'epoch_momentum':1,'sampling':True,'legend':'I-NSGD with '+r"${B'}=64$"}] # """"
#hyps+=[{'epoch_vt':1,'gamma':0.28,'S0':S1,'S1':None,'S01':128,'normalize_power':2/3,
        #'grad_max':60.0,'clip_constant':clip_constant,'beta':0,'epoch_momentum':1,'sampling':True,'legend':'I-NSGD with '+r"${B'}=128$"}]# ","
#hyps+=[{'epoch_vt':1,'gamma':0.5,'S0':S1,'S1':None,'S01':4,'normalize_power':2/3,



random.seed(2024)
np.random.seed(2024)
# initiliaze as GD method
hyp=hyps[1].copy() 
num_iters_init=30

#To generate x1, eta1 via 30 iterations of normalized GD as the initialization of the stochastic algorithms
L,Psi,iters,complexities,x1,eta1=Spider_DRO(X_train,y_train,X_test,y_test,num_iters_init,hyp['epoch_vt'],hyp['epoch_momentum'],\
               hyp['gamma'],hyp['S0'],hyp['S01'],hyp['S1'],hyp['beta'],x0,eta0,hyp['normalize_power'],hyp['grad_max'],hyp['clip_constant'],epoch_eval=1,psi="chi-square",\
                   psi_grad="chi-square",lambda0=0.01,eval_stepsize=0.1,eval_thr=1e-7,eval_maxiter=eval_maxiter,is_eval_psi=is_eval_psi,\
                       penalty_hyps={"coeff_penalty":0.1,"eps_penalty":1.0},indepedent_sampling=False,print_progress=False)


#random.seed(2024)
np.random.seed(2024)
results=[]
num_algs=len(hyps)
for hyp_k in range(num_algs):
    hyp=hyps[hyp_k].copy()
    results+=[{}]
    print("Running "+hyp['legend']+" algorithm.")
    if hyp_k<=2:
        results[hyp_k]['L'],results[hyp_k]['Psi'],results[hyp_k]['iters'],results[hyp_k]['complexities'],_,_=\
            Spider_DRO(X_train,y_train,X_test,y_test,total_iters,hyp['epoch_vt'],hyp['epoch_momentum'],\
                       hyp['gamma'],hyp['S0'],hyp['S01'],hyp['S1'],hyp['beta'],x0,eta0,hyp['normalize_power'],hyp['grad_max'],hyp['clip_constant'],
                       epoch_eval=1,psi="chi-square",psi_grad="chi-square",lambda0=0.01,eval_stepsize=0.1,\
                       eval_thr=1e-7,eval_maxiter=eval_maxiter,is_eval_psi=is_eval_psi,penalty_hyps={"coeff_penalty":0.1,"eps_penalty":1.0},indepedent_sampling=False,print_progress=False)
    else:
        results[hyp_k]['L'],results[hyp_k]['Psi'],results[hyp_k]['iters'],results[hyp_k]['complexities'],_,_=\
            Spider_DRO(X_train,y_train,X_test,y_test,total_iters_stoc,hyp['epoch_vt'],hyp['epoch_momentum'],\
                       hyp['gamma'],hyp['S0'],hyp['S01'],hyp['S1'],hyp['beta'],x1,eta1,hyp['normalize_power'],hyp['grad_max'],hyp['clip_constant'],
                       epoch_eval=1,psi="chi-square",psi_grad="chi-square",lambda0=0.01,eval_stepsize=0.1,\
                       eval_thr=1e-7,eval_maxiter=eval_maxiter,is_eval_psi=is_eval_psi,penalty_hyps={"coeff_penalty":0.1,"eps_penalty":1.0},
                       indepedent_sampling=hyp['sampling'],print_progress=False)

xlabels={'iters':'Iteration t','complexities':'Sample Complexity'}
ylabels={'L':r'$L(x_t,\eta_t)$','Psi':r'$\Psi(x_t)$'}
folder_final='DRO_results/'
if not os.path.isdir(folder_final):
    os.makedirs(folder_final)

y_type='Psi'
x_type='iters'
plt.figure(figsize=(8,6))
for hyp_k in range(3):
    hyp=hyps[hyp_k].copy()
    x_plot=np.array(results[hyp_k][x_type])
    plt.plot(x_plot,results[hyp_k][y_type],color=colors[hyp_k],label=hyp['legend'],\
             )#marker=markers[hyp_k],markevery=int(len(x_plot)/(hyp_k+6))
plt.legend(prop={'size':lgd_size},loc='upper right',ncol = 1)
plt.xlabel(xlabels[x_type])
plt.ylabel(ylabels[y_type])
plt.rc('axes', labelsize=label_size)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=num_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=num_size)    # fontsize of the tick labels
plt.xlabel(xlabels[x_type])
plt.ylabel(ylabels[y_type])
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.savefig(folder_final+'GDs_'+y_type+'VS'+x_type+'.png',dpi=200)
plt.close()

x_type='complexities'
x_max_stoc=np.min([results[k]['complexities'][-1] for k in [3,4,5,6,7,8]]) # remember to change it
plt.figure(figsize=(8,6))
for hyp_k in [3,4,5,6,7,8]: # remeber to change it
    hyp=hyps[hyp_k].copy()
    x_plot=np.array(results[hyp_k][x_type])
    if x_type=='complexities':
        index=(x_plot<=x_max_stoc)
        x_plot=x_plot[index]
        plt.plot(x_plot,np.reshape(results[hyp_k][y_type],-1)[index],linewidth = 2.5,color=colors[hyp_k],label=hyp['legend'],\
                 )#marker=markers[hyp_k-3],markevery=int(len(x_plot)/(hyp_k+6))
    else:
        plt.plot(x_plot,results[hyp_k][y_type],color=colors[hyp_k],label=hyp['legend'],\
                 marker=markers[hyp_k-3],markevery=int(len(x_plot)/(hyp_k+6)))
plt.legend(prop={'size':lgd_size},loc='upper right',bbox_to_anchor=(1.03, 1.03),ncol = 1)
plt.xlabel(xlabels[x_type])
plt.ylabel(ylabels[y_type])
plt.grid(True)
plt.rc('axes', labelsize=label_size)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=num_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=num_size)    # fontsize of the tick labels
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
if x_type=='complexities':
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig(folder_final+'stochastic_'+y_type+'VS'+x_type+'.png',dpi=200)
plt.close()
        

hyp_txt=open(folder_final+'hyperparameters.txt','w')
k=0
for hyp in hyps:
    hyp_txt.write('Hyperparameter '+str(k)+':\n')
    k+=1
    for hyp_name in list(hyp.keys()):
        hyp_txt.write(hyp_name+':'+str(hyp[hyp_name])+'\n')
    hyp_txt.write('\n\n')
hyp_txt.close()

