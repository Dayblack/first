#!/usr/bin/env python
# coding: utf-8

# # 大型购物中心的销售额预测
# 
# *数据集描述*
# 
# **数据**  为2013年美国十家不同城市的商店共1559种产品的销售数据。
# 
# **目的**  是要建立商品在实际商店的销售额预测模型，帮助各地购物中心实现利商品销售润最大化
# 
# **模型建立假设**  为不同地区的消费者具有商品选择的偏好性

# In[ ]:


import pandas as pd
import numpy as np
#read file:
train = pd.read_csv(r"C:\Users\ASUS\Downloads\train_02.csv")
test = pd.read_csv(r"C:\Users\ASUS\Downloads\test_02.csv")

#将训练数据与测试数据合并到一起，做特征工程
train["source"]="train"
test["source"]="test"

data = pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,data.shape)
print(data.head())


# ## DEA
# 

# 检验哪些列存在缺失值

# In[ ]:


data.apply(lambda x:sum(x.isnull()))


# 查看基本的统计变量

# In[ ]:


data.describe()


# **问题1**：Item_Visibility,可见度最小值不可能为0
# 
# **问题2**：对Outlet_Establishment_Year该列最好的表示方法是建立几年而非几几年建立，同时将该列转化为时间格式
# 
# **问题3**：Item_Outlet_Sales与Item_Weight由于缺失值而少

# 查看分类变量

# In[ ]:


data.describe(include="all")
data.apply(lambda x:len(x.unique()))
data.info()


# 过滤分类变量,过滤掉'Item_Identifier','Outlet_Identifier','source'
# 
# 分类别打印频率

# In[ ]:


categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=="object"]
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())


# 使用describe()直接对离散变量进行数据描述

# **针对分类数据分析结论如下：**
# 
# **问题4**：在Item_Fat_Content列，LF、reg、low fat，需要进行合并
# 
# **问题5**：对于Item_Type，分类条目过多，需要进行分类缩减
# 
# **问题6**：对于Outlet_Type，Supermarket Type2 、Supermarket Type3 需要考虑是否相近来判断是否合并
# 
# **问题
# 
# 

# ## 数据清洗
# 
# **包括缺失值处理、异常值处理、离群值处理**

# In[ ]:


"""解决问题3"""
#第一步，输入缺失值Item_Weight and Outlet_Size,使用各商品的平均重量进行填充
item_avg_weight = data.pivot_table(values="Item_Weight",index="Item_Identifier")
#第二步，获取在Item_Weight中缺失值的位置
miss_bool = data["Item_Weight"].isnull()
#第三步，根据Item_Identifier对缺失值进行填充
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print('Final #missing: %d'% sum(data['Item_Weight'].isnull()))
#-------------------------------------------------------------------------------------
#scipy.stats.mode寻找出现次数最多的成员，返回传入数组/矩阵中最常出现的成员以及出现的次数
from scipy.stats import mode
#第一步，预先处理数据
outlet_size_mode = data.pivot_table(values='Outlet_Size', index='Outlet_Type',aggfunc=(lambda x:mode(x)[0]))
print(outlet_size_mode)

#第二步，创建缺失值布尔值
miss_bool = data['Outlet_Size'].isnull() 

#第三步，进行替换
data.loc[miss_bool,"Outlet_Size"] = data.loc[miss_bool,"Outlet_Type"].apply(lambda x: outlet_size_mode[x])
print(sum(data["Outlet_Size"].isnull()))


# ## 特征工程

# In[ ]:


#解决问题6
"""使用两样本的独立T检验，检验Supermarket Type2 and Type3销售额是否有显著差异,从而判断是否将两类型合并为一个类型
需要满足假设前提条件1..两超市销售额服从正态分布2.进行方差齐次检验
检验假设如下：
H0:Supermarket Type2 and Type3均值之差为0
H1:Supermarket Type2 and Type3均值之差不为0
"""
#正态分布假设检验
from scipy.stats import normaltest
data1 = data[data["Item_Outlet_Sales"].isnull() == False].loc[:,["Outlet_Type","Item_Outlet_Sales"]]
test_report=[]
def testreport(data_cl):
    for i in range(3):
        one = "Supermarket Type" + str(i+1)
        test_report.append(normaltest(data_cl[data_cl.loc[:,"Outlet_Type"] == one]["Item_Outlet_Sales"])[1])
        alpha = 1e-3
        if test_report[i] < alpha:
            print(one,"检验结果不符合正态分布")
        else:
            print(one,"检验结果符合正态分布") 
testreport(data1)
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')
"""检验结果均未通过正态检验，由于后续涉及到线性回归模型等其他模型，因此转换销售额偏态分布数据为正态分布数据对于建模准确有提升"""


#对每种Supermarket Type销售额进行分布情况展示
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["YouYuan"]
mpl.rcParams["axes.unicode_minus"] = False
plt.figure(1,figsize=(20,10))
for i in range(3):
    moe = "Supermarket Type" + str(i+1)
    data_update = data1[data1.loc[:,"Outlet_Type"] == moe]["Item_Outlet_Sales"].reset_index(drop=True)
    plt.figure(1)
    ax = plt.subplot(1,3,i+1)
    plt.ylabel("Supermarket Type%d" %i,size=20,family="Time New Roman")
    plt.xlabel("sale",size=15,family="Time New Roman")
    labels = ax.get_yticklabels() + ax.get_xticklabels()
    plt.tick_params(labelsize=15)
    [label.set_fontname("Time New Roman") for label in labels]
    plt.hist(data_update,bins=len(data_update)//10,color=np.random.choice(["r","m","c","g"],1,replace=False))
plt.subplot(1,3,2)
font_dict={"fontsize":40,
            "fontweight":9.0,
            "color":"g"}
plt.title("销售额直方图",fontdict=font_dict,loc="center")


#将三种分布图在同一个图中显示，对比差异
import seaborn as sns
sns.set(color_codes=True)
plt.figure(figsize=(20, 10))
for i in range(3):
    moe = "Supermarket Type" + str(i+1)
    data_update = data1[data1.loc[:,"Outlet_Type"] == moe]["Item_Outlet_Sales"].reset_index(drop=True)
    sns.distplot(data_update, kde=True,hist=True,bins=len(data_update)//10,fit=norm)
"""三种类型超市销售额分布均为不同程度的右偏分布"""


#三种分布图与正态分布的差异
import seaborn as sns
from scipy.stats import norm, skew
sns.set(color_codes=True)
plt.figure(figsize=(60, 60))

lambda_list = dict()
for i in range(3):
    moe = "Supermarket Type" + str(i+1)
    data_update = data1[data1.loc[:,"Outlet_Type"] == moe]["Item_Outlet_Sales"].reset_index(drop=True)
    plt.subplot(3,3,1+3*i)
    sns.distplot(data_update, kde=True,hist=True,bins=len(data_update)//10,fit=norm)
    (mu,sigma)=norm.fit(data_update)
    plt.legend(['$\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu,sigma)],loc='best')
    plt.ylabel('Frequency')
    plt.subplot(3,3,2+3*i)
    res=stats.probplot(data_update,plot=plt)
    plt.suptitle('Before')
    print("-"*50)
    print(f"Skewness of saleprice: {data_update.skew()}")
    print(f"Kurtosis of saleprice: {data_update.kurt()}")

    #进行Box-Cox变换
    print("Boc-Cox变换")
    data_update,lambda_=stats.boxcox(data_update)
    lambda_list.update({moe:lambda_})
    print(lambda_)
    plt.subplot(3,3,3+3*i)
    res=stats.probplot(data_update,plot=plt)
    plt.suptitle('After')
    print(f"Skewness of saleprice: {pd.Series(data_update).skew()}")
    print(f"Kurtosis of saleprice: {pd.Series(data_update).kurt()}")

#销售额正态分布转化完成，lambda值分别为0.37076703594771304、0.30825879264840556、0.37076703594771304方便后续逆转换
#转换公式为：Y = (Y^lambda_-1)/lambda_
#添加一列，表示经过COX_BOX正态分布转换后的销售额
#新增一列各种类超市正态分布销售额Item_Outlet_Sales_boxcox
data_index =[]
for i in range(3):
    moe = "Supermarket Type" + str(i+1)
    data_update = data1[data1.loc[:,"Outlet_Type"] == moe]["Item_Outlet_Sales"]

    #进行Box-Cox变换
    print(len(data_update))
    print("Boc-Cox变换")
    data_index.append(data_update.index.to_list())
    data_update,lambda_=stats.boxcox(data_update)
    data.ix[data_index[i],"Item_Outlet_Sales_boxcox"] = data_update 
    

data_G_index = data[(data["Outlet_Type"]=="Grocery Store")&(np.isnan(data["Item_Outlet_Sales"])==False)].index
data.ix[data_G_index,"Item_Outlet_Sales_boxcox"] =stats.boxcox(data.ix[data_G_index,"Item_Outlet_Sales"])[0]
#处理完毕


# In[ ]:


#解决问题1
"""Item_Visibility将0改为产品的平均可见度"""
#确定产品的平均可见性
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
#获取可见度为0的索引
miss_bool = (data['Item_Visibility'] == 0)
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

"""创建在不同商店商品的重要性，根据visibility_avg来创建"""
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())


# In[ ]:


# data["Item_Visibility"].replace(0,numpy.NaN,inplace=True)
# 利用groupby().transform()
# data.groupby("Item_Identifier").transform(lambda x: x.fillna(x.mean()))


# In[ ]:


#解决问题5
"""根据FD、DR或NC开头进行创建新类别"""
#提取
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#增加
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[ ]:


#解决问题2
"""数据集为2013年收集的"""
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[ ]:


#解决问题4
"""使用replace对重义词进行合并"""
print("Original Categories:")
print(data['Item_Fat_Content'].value_counts())

print("\nModified Categories:")
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({"LF":"Low Fat",
                                                             "reg":"Regular",
                                                             "low fat":"Low Fat"})
print(data['Item_Fat_Content'].value_counts())

#完善问题4
"""根据问题五的处理，我们应该将NC修改为Non-Edible"""
data.loc[data['Item_Type_Combined']=="Non-Consumable","Item_Fat_Content"] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[ ]:


"""将所有的类别的名义变量转换为数值型变量"""
#进行One-Hot-Coding，首先从sklearn预处理模块将所有分类变编码为数字
from sklearn.preprocessing import LabelEncoder
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
#进行One-Hot-Coding
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
data.dtypes
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[ ]:


data.head()


# In[ ]:


"""导出数据"""
#删除已转换的列
data.drop(['Item_Type','Outlet_Establishment_Year','Item_Outlet_Sales'],axis=1,inplace=True)
#分割为测试集与训练集
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
#删除多余的列
test.drop(['Item_Outlet_Sales_boxcox','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#储存
train.to_csv(r"C:\Users\ASUS\Downloads\train_modified.csv",index=False)
test.to_csv(r"C:\Users\ASUS\Downloads\test_modified.csv",index=False)


# In[ ]:


data.head()


# # 建立模型

# In[ ]:


#选用Item_Outlet_Sales_boxcox总体平均值作为基准线
mean_sales = train['Item_Outlet_Sales_boxcox'].mean()

#创建一个提交ID与预测结果的dataframe
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales_boxcox'] = mean_sales

base1.to_csv(r"C:\Users\ASUS\Downloads\alg0.csv",index=False)


# In[ ]:


#Define target and ID columns:
target = 'Item_Outlet_Sales_boxcox'
IDcol = ['Item_Identifier','Outlet_Identifier']
#导入交叉验证
from sklearn import metrics
from sklearn.model_selection import cross_val_score
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    
    #根据train数据调整算法结构
    alg.fit(dtrain[predictors], dtrain[target])
    
    
    #对训练集进行预测:
    #得到当前超参模型对应的的预测结果
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    #建立交叉验证集，模型评估准则使用MSE，评估模型超参数（也就是该模型）在交叉验证集上的得分:
    #作用：得分与接下来几个模型进行对比，作为模型好坏比较的一种测量，根据得分高低帮助我们选择合适的模型
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring="neg_mean_squared_error")
    cv_score = np.sqrt(np.abs(cv_score))
    
    #打印模型报告结果:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #最后在test上最终预测：（不参与任何训练过程）
    dtest[target] = alg.predict(dtest[predictors])
    
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# In[ ]:


#建立OLS线性回归模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]

# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
#提取出线性模型的各系数
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


# In[ ]:


"""对于系数结果，可以发现大小差异非常大，容易使模型偏向过高的一些特征，使得模型存在过拟合
系数量级差异非常之大，就可以怀疑OLS回归过度拟合一些噪音点使用带正则化的最小二乘法-岭回归，L2正则"""


# In[ ]:


#岭回归模型
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

"""岭回归后的系数平稳，但两者交叉验证得分并没有太大差别，考虑使用决策树模型进行拟合"""


# In[ ]:


#随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
#n_estimators：回归树的个数
#max_depth：最大树深
#n_jobs：拟合与预测CPU核数
#max_features：允许单个决策树使用特征的最大数量
#对于bagging来说，个体学习器应该具有独立性，遵循好而不同的准则
#oob_score:随机森林交叉验证方法,使用袋外样本估计未知数据R^2


#网格参数调优
from sklearn.model_selection import GridSearchCV
rfr = RandomForestRegressor()
parameters = {
    "max_features":[0.4,0.6,0.8,0.9],
    "max_depth":[4,5,6],
    "n_estimators":[200,300,400,500],
    "min_samples_leaf":[80,100,150],
    "n_jobs":[-1],
    "random_state":[50]
}
clf = GridSearchCV(estimator=rfr,param_grid=parameters)


target = 'Item_Outlet_Sales_boxcox'
IDcol = ['Item_Identifier','Outlet_Identifier']
predictors = [x for x in train.columns if x not in [target]+IDcol]
clf.fit(train[predictors], train[target])

print("最优分数： %.4lf" %clf.best_score_)
print("最优参数：", clf.best_params_)


# ### 最优参数结果：
# **最优分数**(MSE)： 
# 
# 0.8386
# 
# **最优参数**：
# 
# {'max_depth': 6, 'max_features': 0.9, 'min_samples_leaf': 80, 'n_estimators': 200, 'n_jobs': -1, 'random_state': 50}

# In[ ]:


alg5 = RandomForestRegressor(n_estimators=200,max_depth=6, min_samples_leaf=80,n_jobs=-1,max_features=0.9)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

coef5.sort_values(ascending = False)
#可以发现，Outlet_Type_0、Item_MRP、Outlet_Type_3、Outlet_5 、Outlet_Years、Item_Visibility_MeanRatio
#每个元素值为正，并且总和为 1.0。一个元素的值越高，其对应的特征对预测函数的贡献越大

