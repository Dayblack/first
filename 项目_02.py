#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#read file:
train = pd.read_csv(r"C:\Users\ASUS\Downloads\train_02.csv")
test = pd.read_csv(r"C:\Users\ASUS\Downloads\test_02.csv")


# In[2]:


#将训练数据与测试数据合并到一起，做特征工程
train["source"]="train"
test["source"]="test"

data = pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,data.shape)


# In[3]:


#检验那些列存在缺失值
data.apply(lambda x:sum(x.isnull()))


# In[4]:


#查看基本的统计变量
data.describe()


# In[5]:


"""
问题1：Item_Visibility,可见度最小值不可能为0
问题2：对Outlet_Establishment_Year该列最好的表示方法是建立几年而非几几年建立
问题3：Item_Outlet_Sales与Item_Weight由于缺失值而少
"""
data.head()


# In[6]:


#查看分类变量
data.apply(lambda x:len(x.unique()))


# In[7]:


#过滤分类变量,过滤掉'Item_Identifier','Outlet_Identifier','source'
#分类别打印频率
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=="object"]
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
for col in categorical_columns:
    print('\nFrequency of Categories for varible %s'%col)
    print(data[col].value_counts())


# In[8]:


#尝试使用describe()直接对离散变量进行数据描述
#describe(),参数include默认描述数值型统计量
data.describe(include="all")


# In[9]:


"""针对分类数据结论如下：
问题4：在Item_Fat_Content列，LF、reg、low fat，需要进行合并
问题5：对于Item_Type，分类条目过多，需要进行分类缩减
问题6：对于Outlet_Type，Supermarket Type2 、Supermarket Type3 需要考虑是否相近来判断是否合并
"""


# In[10]:


"""数据清洗"""
"""通常来讲，包括缺失值处理、异常值处理、离群值处理"""
#解决问题3
#第一步，输入缺失值Item_Weight and Outlet_Size
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
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]))
print(outlet_size_mode)

#第二步，创建缺失值布尔值
miss_bool = data['Outlet_Size'].isnull() 

#第三步，进行替换
data.loc[miss_bool,"Outlet_Size"] = data.loc[miss_bool,"Outlet_Type"].apply(lambda x: outlet_size_mode[x])
print(sum(data["Outlet_Size"].isnull()))


# In[11]:


"""特征工程"""
#解决问题6
"""使用两样本的独立T检验，检验Supermarket Type2 and Type3销售额是否有显著差异
假设1.两超市销售额服从正态分布2.进行方差齐次检验
H0:Supermarket Type2 and Type3均值之差为0
H1:Supermarket Type2 and Type3均值之差不为0
"""
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')
#改进  进行Supermarket Type1、Supermarket Type2、Supermarket Type3的销售额方差分析


# In[12]:


#解决问题1
"""Item_Visibility将0改为产品的平均可见度"""
#确定产品的平均可见性
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
#获取可见度为0的索引
miss_bool = (data['Item_Visibility'] == 0)
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


# In[13]:


# data["Item_Visibility"].replace(0,numpy.NaN,inplace=True)
# #利用groupby().transform()
# data.groupby("Item_Identifier").transform(lambda x: x.fillna(x.mean()))


# In[14]:


"""创建在不同商店商品的重要性，根据visibility_avg来创建"""
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())
#改进  可以尝试更多有用特征的创建


# In[15]:


#解决问题5
"""根据FD、DR或NC开头进行创建新类别"""
#提取
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#增加
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()
#后期改进， 可以根据销售额组合来分类为不同的类别


# In[16]:


#解决问题2
"""数据集为2013年收集的"""
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[17]:


#解决问题4
"""使用replace对重义词进行合并"""
print("Original Categories:")
print(data['Item_Fat_Content'].value_counts())

print("\nModified Categories:")
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({"LF":"Low Fat",
                                                             "reg":"Regular",
                                                             "low fat":"Low Fat"})
print(data['Item_Fat_Content'].value_counts())


# In[18]:


#完善问题4
"""根据问题五的处理，我们应该将NC修改为'Non-Edible'"""
data.loc[data['Item_Type_Combined']=="Non-Consumable","Item_Fat_Content"] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[19]:


"""将所有的类别的名义变量转换为数值型变量"""
#进行One-Hot-Coding，首先从sklearn预处理模块将所有分类变编码为数字
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
data['Outlet']
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
#进行One-Hot-Coding，


# In[20]:


data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
data.dtypes
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[21]:


"""导出数据"""
#删除已转换的列
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#分割为测试集与训练集
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]
#删除多余的列
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#储存
train.to_csv(r"C:\Users\ASUS\Downloads\train_modified.csv",index=False)
test.to_csv(r"C:\Users\ASUS\Downloads\test_modified.csv",index=False)


# In[22]:


"""建立模型"""

#选用Item_Outlet_Sales总体平均值作为基准线
mean_sales = train['Item_Outlet_Sales'].mean()

#创建一个提交ID与预测结果的dataframe
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#
base1.to_csv(r"C:\Users\ASUS\Downloads\alg0.csv",index=False)


# In[38]:


#Define target and ID columns:
target = 'Item_Outlet_Sales'
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


# In[39]:


#建立线性回归模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]

# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
#提取出线性模型的各系数
coef1 = pd.Series(alg1.coef_, predictors).sort_values()


# In[40]:


coef1.plot(kind='bar', title='Model Coefficients')


# In[47]:


#对于系数结果，可以发现大小差异非常大，容易使模型偏向过高的一些特征，使得模型存在过拟合
#系数量级差异非常之大，就可以怀疑OLS回归过度拟合一些噪音点
#使用带正则化的最小二乘法-岭回归，L2正则


# In[48]:


#岭回归模型
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')


# In[ ]:




