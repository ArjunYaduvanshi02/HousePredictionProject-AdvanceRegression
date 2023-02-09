import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestRegressor
import pickle
import sklearn
df=pd.read_csv("train.csv")
print(df.shape)
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
print(df.shape)
df.drop(['Id'],axis=1,inplace=True)
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)
print(df.shape)
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond','ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
         'BsmtFinType2','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir','Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


def category_onehot_multcols(multcolumns):
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df.drop([fields], axis=1, inplace=True)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
main_df=df.copy()
test_df=pd.read_csv('formulatedtest.csv')
print(test_df.shape)
final_df=pd.concat([df,test_df],axis=0)
print(final_df['SalePrice'])
print(final_df.shape)
final_df=category_onehot_multcols(columns)
print(final_df.shape)
final_df =final_df.loc[:,~final_df.columns.duplicated()]
print(final_df.shape)
df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]
print(df_Train.shape)
y_train=df_Train['SalePrice']
X_train=df_Train.drop(['SalePrice'],axis=1)
clf=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0,max_iter=1000).fit(X_train,y_train)
clf.fit(X_train,y_train)
final_file="finalized.pkl"
pickle.dump(clf,open(final_file,'wb'))
y_pred=clf.predict(df_Test.drop(['SalePrice'],axis=1))
print(y_pred)
pred=pd.DataFrame(y_pred)
submission=pd.read_csv('sample_submission.csv')
dataset=pd.concat([submission['Id'],pred],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission.csv',index=False)