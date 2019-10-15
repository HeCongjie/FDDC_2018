import pandas as pd

import numpy as np
import scipy as sp


from sklearn import linear_model
from sklearn.linear_model import LassoLarsCV,LinearRegression
import time

from math import sqrt
import warnings
import re

def del_zero_col(df):
    a_list=list(df.columns)
    names = [item for item in a_list if item not in inter_col]
    for nm in names:
        test=df[nm]
        if  set(test)=={0}:
            df=df.drop(nm,1)
    return df
def del_duplicate_rep(df):
    tick=df.sort_values(['END_DATE','PUBLISH_DATE'])
    row=len(tick)
    tick.index=range(row)
    delect=[]
    for i in range(row-1):
        if tick.END_DATE[i]==tick.END_DATE[i+1]:
            delect.append(i)
    tick=tick.drop(delect)
    tick.index=range(len(tick))
    return tick


def to_season(df):
    #columns need to transform to season
    to_q=[nm for nm in df.columns if nm in is_cf_col]
    #columns needn't to transform to season
    oth=[q for q in df.columns if q not in to_q ]
    dirs={'A':4,'Q3':3,'S1':2,'Q1':1}
    rep_type=df.REPORT_TYPE
    df1=df[oth]
    data=df[to_q]
    #check report type
    loc=len(data)-1
    while loc>0:
        if (dirs[rep_type[loc]]==1)|(dirs[rep_type[loc]]-dirs[rep_type[loc-1]]==1):
            loc-=1
        else:
            break
        
    df1=df1[loc:]
    data=data[loc:]
    row,col=len(data),len(data.columns)
    data.index=range(row)
    df1.index=range(row)
    data_q=np.zeros((row,col))
    rep_type1=df1.REPORT_TYPE
    for i in range(row):
        for j in range(col):
            if (dirs[rep_type1[i]]==1)|(i==0):
                data_q[i][j]=data.iloc[i,j]
            elif dirs[rep_type1[i]]-dirs[rep_type1[i-1]]==1:
                data_q[i][j]=data.iloc[i,j]-data.iloc[i-1,j] 
                
    df_q=pd.DataFrame(data_q,columns=to_q)
    result=pd.concat([df1,df_q],axis=1,join_axes=[df1.index])
    if result.REPORT_TYPE[0]!='Q1':
        result=result[1:]
        result.index=range(len(result))
    return result

#delect error data
def del_error_col(df):
    a_list=list(df.columns)
    names = [item for item in a_list if item not in inter_col]
    for nm in names:
        test=df[nm]
        if  len(set(test))<4:
            df=df.drop(nm,1)
    return df
 
def select_cut_point(df):
    all_point=range(3,len(df.index)-5)
    revenue=df.REVENUE
    dict={}
    for i in all_point:

        left_1=revenue.values[i]-revenue.values[i-1]
        right_1=revenue.values[i+1]-revenue.values[i]
        left_2=revenue.values[i-1]-revenue.values[i-2]
        right_2=revenue.values[i+2]-revenue.values[i+1]
        left_3=revenue.values[i-2]-revenue.values[i-3]
        right_3=revenue.values[i+3]-revenue.values[i+2]

        left=left_1
        right=right_1
        if ((left_2<0) == (left_1<0)):
            left=(left_1+left_2)
            if ((left_3<0) == (left_2<0)):
                left=(left_1+left_2+left_3)
        if ((right_2<0) == (right_1<0)):
            right=(right_1+right_2)
            if ((right_3<0) == (right_2<0)):
                right=(right_1+right_2+right_3)
        dict[i]=abs(left-right)

    mean_s=np.mean(list(dict.values()))
    cut_down=0
    if mean_s*2<max(dict.items(), key=lambda x:x[1])[1]:
        cut_down=max(dict.items(), key=lambda x:x[1])[0]
    return cut_down

def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)

def Error(rgt_df):
    x_list=rgt_df.index.tolist()
    z_n=np.polyfit(x_list, rgt_df.REVENUE, 4)
    p_n=np.poly1d(z_n)
    Error=R22(p_n(x_list),rgt_df.REVENUE)
    #plt.plot(x_list,p_3(x_list))
    #plt.plot(x_list,rgt_df.REVENUE,color='green')
    #plt.show()
    return Error
 
#trend_cut
def lindistance(a):
    n=len(a)
    an=[]
    for j in range(n):
        aj=j*(a[n-1]-a[0])/(n-1)+a[0]
        an.append(aj-a[j])  
    return an

def trend_cut(rev): 
    smooth1=rev.rolling(3).mean().dropna(0)
    smooth2=smooth1.rolling(3).mean().dropna(0)
    s1=np.array(rev)/rev.max()
    max_s=smooth2.max()
    a=np.array(smooth2)/max_s
    #k：截取点调整；flat:营收值稳定不变；thredhold:折线拟合阈值;degree:判断转折的角度；distan:判断转折的另外一个参数
    k=2
    #flat,thredhold,degree,distan=0.16,0.13725,0.3725,0.2365
    #flat,thredhold,degree,distan=0.315306, 0.135714, 0.269696, 0.265306
    flat,thredhold,degree,distan=0.315306, 0.135714, 0.261879, 0.2642953
    n=len(smooth2)
    an=lindistance(a)    
    p=[0,n-1]
    max_s=an[1]
    pk=1
    for j in range(2,n-1):    
        if (an[j]*an[j-1]>0)&(abs(an[j])>abs(max_s)):
            pk=j
            max_s=an[j]          
        if an[j]*an[j-1]<0:
            if (thredhold<abs(max_s))&(pk not in p):
                p.insert(0,pk)                 
        if (j==n-2)&(pk not in p):
             if thredhold<abs(max_s):
                p.insert(0,pk)                            
    p=sorted(p)
    kv=[]
    for j in range(1,len(p)):   
        kj=(a[p[j]]-a[p[j-1]])
        kv.append(kj)
    
    point=[] 
    if len(p)==2:result=0
    elif ((abs(s1[-1]-s1[0])<flat)|(abs(a[-1]-a[0])<flat))&(len(p)>2):
        p.pop(0)
        if p[-1]-p[-2]<6:p[-2]=p[-2]-2
        p.pop()
        point=[cp+k for cp in p]
    
    else:
        result=0
        if abs(kv[0])<0.05:
            p.pop(0)
            if len(p)==2:
                if p[1]-p[0]<6:result=p[0]
                else:result=p[0]+k 
        nn=len(p)-1
        while nn-2>=0:
            deg=np.arctan(kv[nn-1])-np.arctan(kv[nn-2])
            distance=lindistance(an[p[nn-2]:p[nn]+1])      
            if (abs(deg)>degree)&(abs(distance[p[nn-1]-p[nn-2]])>distan) :
                result=p[nn-1]+k
                if p[-1]-p[nn-1]<6:
                    result=p[-1]-2
                break
            else:
                if nn==2:
                    if p[0]==0:result=0
                    else:reslut=p[0]+k
            nn-=1 
        point.append(result)
    if 0 not in point:
        point.append(0)
    return sorted(point)

#Normalize columns in x and labels
def normalize(xList,labels):    
    nrows = len(xList)
    ncols = len(xList[0])
    #calculate means and variances
    xMeans = []
    xSD = []
    for i in range(ncols):
        col = [xList[j][i] for j in range(nrows)]
        mean = sum(col)/nrows
        xMeans.append(mean)
        colDiff = [(xList[j][i] - mean) for j in range(nrows)]
        sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
        stdDev = sqrt(sumSq/nrows)
        xSD.append(stdDev)    
    #use calculate mean and standard deviation to normalize xList
    xNormalized = []
    for i in range(nrows):
        rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)    
    #Normalize labels
    meanLabel = sum(labels)/nrows
    sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)    
    labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]    
    #normalized lables
    Y = np.array(labelNormalized)
    #Normlized Xss
    X = np.array(xNormalized)
    return X,Y

def choose_result(df,cut):
    #版本3
    num=len(cut)
    n,pct_min=0,0.6
    q1_all,q2_all=[],[]
    for i in range(num):
        df1=df[cut[i]:].reset_index(drop=True)
        df1=del_error_col(df1)
        lp=lasso_predict(df1)
        pct=abs(lp[1]/lp[0]-1)
        if i==0:
            pct_min=pct
            q1_true,q1,q2=lp[0],lp[1],lp[2]
        else:
            if pct<=pct_min:
                pct_min=pct
                q1,q2=lp[1],lp[2]
        if (lp[2]/lp[0]>1)&(lp[1]>0):
            q1_all.append(lp[1])
            q2_all.append(lp[2])
            n+=1   
    
    #表决
    thred=0.7
    q1_all=sorted(q1_all)
    q2_all=sorted(q2_all)
    if n>=3:
        q1_max=max(q1_all)
        q1_norm=q1_all/q1_max
        q2_max=max(q2_all)
        q2_norm=q2_all/q1_max
        cmin1,cmin2,cmax1,cmax2=0,0,0,0
        for i in range(n):
            if q1_norm[i]<thred:cmin1+=1     
            else:cmax1+=1
            if q2_norm[i]<thred:cmin2+=1
            else:cmax2+=1
        if cmin1>cmax1:q1=(sum(q1_all)-q1_all[-1])/(n-1)
        elif cmin1<cmax1:q1=(sum(q1_all)-q1_all[0])/(n-1)
        else:q1=np.mean(q1_all)
        if cmin2>cmax2:q2=q(sum(q2_all)-q2_all[-1])/(n-1)
        elif cmin2<cmax2:q2=(sum(q2_all)-q2_all[0])/(n-1)
        else:q2=np.mean(q2_all)
    elif n>0:
        q1=sum(q1_all)/n
        q2=sum(q2_all)/n        
    return [round(q1_true,2),round(q1,2),round(q2,2)]

def lasso_predict(df):
    K=0.65306122
    a_list=list(df.columns)
    names = [item for item in a_list if item not in inter_col]
    xdata=df[names]
    labels=np.array(df.REVENUE[1:],dtype=np.float64)
    xList=np.array(xdata[:-1],dtype=np.float64)
    X,Y=normalize(xList,labels)
    X[np.isnan(X)]=0
    #give value of cv
    n=len(df)
    if n>8:
        cv=8
    else:
        cv=n-2
    #Call LassoCV from sklearn.linear_model
    X=np.nan_to_num(X)
    Rev_Model = LassoLarsCV(cv=cv).fit(X, Y)
    alphas, coefs, _  = linear_model.lasso_path(X, Y,  return_models=False)
    nattr, nalpha = coefs.shape
    #find coefficient ordering
    nzList = []
    for iAlpha in range(1,nalpha):
        coefList = list(coefs[: ,iAlpha])
        nzCoef = [index for index in range(nattr) if coefList[index] != 0.0]
        for q in nzCoef:
            if not(q in nzList):
                nzList.append(q)
    #find coefficients corresponding to best alpha value. alpha value corresponding to
    #normalized X and normalized Y is Rev_Model.alpha_
    alphaStar =Rev_Model.alpha_
    indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
    indexStar = max(indexLTalphaStar)
    #here's the set of coefficients to deploy
    coefStar = list(coefs[:,indexStar])
    #The coefficients on normalized attributes give another slightly different ordering
    absCoef = [abs(a) for a in coefStar]
    #sort by magnitude
    coefSorted = sorted(absCoef, reverse=True)
    idxCoefSize = [absCoef.index(a) for a in coefSorted if not(a == 0.0)]
    vari_nm= [xdata.columns[idxCoefSize[i]] for i in range(len(idxCoefSize))]
    
    #use variables in vari_nmto regress
    feat=min(len(vari_nm),int(K*len(df)))
    vari_nm=vari_nm[:feat]
    y=np.array(xdata.REVENUE[1:],dtype=np.float64)
    x=np.array(xdata[vari_nm][:-1],dtype=np.float64)
    xpred=np.array(xdata[vari_nm][-2:])
    rev_q1_true=np.float(xdata.REVENUE[-1:])/1000000
    X1,Y1=normalize(x,y)
    reg = linear_model.LassoLarsIC(criterion='aic')
    reg.fit(X1,Y1)
    coefs=reg.coef_
    score=reg.score(X1,Y1)
    vari_nm1=[vari_nm[i] for i in range(len(vari_nm)) if coefs[i]!=0]
    if (len(vari_nm1)>1)&(score>0.412626):
        x=np.array(xdata[vari_nm1][:-1],dtype=np.float64)
        xpred=np.array(xdata[vari_nm1][-2:])
    linreg=LinearRegression()
    linreg.fit(x, y)
    score=linreg.score(x,y)
    rev_q1=linreg.predict(xpred)[0]/1000000
    rev_q2=linreg.predict(xpred)[1]/1000000+rev_q1_true
    return  [rev_q1_true,rev_q1,rev_q2]





if __name__=='__main__':
    print('Start')
    warnings.filterwarnings('ignore')
    #非数值列
    inter_col=['PARTY_ID','TICKER_SYMBOL','EXCHANGE_CD','PUBLISH_DATE','END_DATE_REP','END_DATE','REPORT_TYPE','FISCAL_PERIOD','MERGED_FLAG']
    #IS和CF表的所有columns
    sheets=[4,5,6,7,8,9,10,11]
    data_dict=pd.read_excel('/home/fddc1_data/financial_data/Data Dictionary.xlsx',sheet_name=sheets)
    is_cf=[]
    for i in sheets:
        is_cf+=list(data_dict[i].iloc[4:,0])
    is_cf_col=[nm for nm in set(is_cf) if nm not in inter_col]

    BS= pd.read_excel('/home/fddc1_data/financial_data/Balance Sheet.xls',sheet_name=[0,1,2,3])
    IS=pd.read_excel('/home/fddc1_data/financial_data/Income Statement.xls',sheet_name=[0,1,2,3])
    CF=pd.read_excel('/home/fddc1_data/financial_data/Cashflow Statement.xls',sheet_name=[0,1,2,3])
    print('Read Done')

    #合并成一个表,按TICKER_SYMBOL分组
    all_bs=pd.concat([BS[0], BS[1],BS[2],BS[3]], axis=0,sort=False,ignore_index=True)
    all_is=pd.concat([IS[0], IS[1],IS[2],IS[3]], axis=0,sort=False,ignore_index=True)
    all_cf=pd.concat([CF[0], CF[1],CF[2],CF[3]], axis=0,sort=False,ignore_index=True)
    col_on=['PARTY_ID','TICKER_SYMBOL','EXCHANGE_CD','PUBLISH_DATE','END_DATE_REP','END_DATE','REPORT_TYPE','FISCAL_PERIOD','MERGED_FLAG']
    all_ticker=all_bs.merge(all_is,on=col_on,how='inner')
    all_ticker=all_ticker.merge(all_cf,on=col_on,how='inner').fillna(0)
    all_ticker_group=all_ticker.groupby(['TICKER_SYMBOL'])

    #获取需要预测的股票的代码
    comp_sub=pd.read_csv('/home/fddc1_data/predict_list.csv',header=None)[0]
    comp_sub_id=[int(re.sub("\D", "", comp_sub[i])) for i in range(len(comp_sub))] 

    predict=[]
    for nm,group in all_ticker_group:
        if nm in comp_sub_id:
            print(nm)
            df=del_duplicate_rep(group)
            df=to_season(df) 
            df=del_zero_col(df)
            cut=trend_cut(df.REVENUE) 
            lp=choose_result(df,cut) 
            if lp[2]<0:
                cutpoint=select_cut_point(df)
                if (cutpoint not in cut):cut.append(cutpoint)
                lp=choose_result(df,cut)
            symbol=['%06d.%s'%(df.TICKER_SYMBOL[0],df.EXCHANGE_CD[0])]
            predict.append(symbol+lp)  
        
    predicted=pd.DataFrame(predict,columns=['TICKER_SYMBOL','REVENUE_Q1_True','REVENUE_Q1','REVENUE_Q2']).sort_values(by=['TICKER_SYMBOL']).reset_index(drop=True)
    #predicted.to_csv('D:FDDC/REV_predicted_v0.csv',index=False)
    keynm=['TICKER_SYMBOL','REVENUE_Q2']
    localtime = time.localtime(time.time())
    predicted[keynm].to_csv('/home/47_151/submit/submit_%d%02d%02d_%02d%02d%02d.csv' % (localtime[0],localtime[1],localtime[2],localtime[3],localtime[4],localtime[5]),index=False)
    print('Well Done')

