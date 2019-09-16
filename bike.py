import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import missingno as msno
import datetime as dt

# %matplotlib inline
plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False
train = pd.read_csv('bike\\train.csv', parse_dates=['datetime'])
test = pd.read_csv('bike\\test.csv', parse_dates=['datetime'])
# print(train)
# print(train.shape)
# print(train.info())
# print(train.head())
# print(train.temp)
# print(train.temp.describe())
# print(train.isnull().sum())


# msno.matrix(train,figsize=(12,5))
# plt.show()

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['weekday'] = train['datetime'].dt.weekday
# for i in train.groupby(train['weekday']):
#     if i[0]==5 or i[0]==6 :
#         train['weekend']=i[0]
# print(train)
train['hour'] = train['datetime'].dt.hour

# figure, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)

# sns.barplot(data=train,x='year',y='count',ax=ax1)
# sns.barplot(data=train,x='month',y='count',ax=ax2)
# sns.barplot(data=train,x='day',y='count',ax=ax3)
# sns.barplot(data=train,x='weekday',y='count',ax=ax4)
# #sns.barplot(data=train,x='year',y='count',ax=ax1)
# plt.show()

# sns.boxplot(data=train,y='count',orient='v',ax=ax1)
# sns.boxplot(data=train,y='count',x='season',ax=ax2)
# sns.boxplot(data=train,y='count',x='hour',ax=ax3)
# sns.boxplot(data=train,y='count',x='workingday',ax=ax4)
# plt.show()

train['dayofweek'] = train['datetime'].dt.dayofweek


# print(train.info())
# print(train)
# print(train['dayofweek'].value_counts())
# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
# fig.set_size_inches(18,25)
# sns.pointplot(data=train,y='count',x='hour',ax=ax1)
# sns.pointplot(data=train,y='count',x='hour',ax=ax2,hue='workingday')
# sns.pointplot(data=train,y='count',x='hour',ax=ax3,hue='dayofweek')
# sns.pointplot(data=train,y='count',x='hour',ax=ax4,hue='weather')
# sns.pointplot(data=train,y='count',x='hour',ax=ax5,hue='season')
# plt.show()

# corrmatt = train[['temp','atemp','casual','registered','humidity','windspeed','count']]
# #print(type([[['temp','atemp','casual','registered','humidity','windspeed','count']]]))
# #print(type(['temp','atemp','casual','registered','humidity','windspeed','count']))
#
# corrmatt = corrmatt.corr()
#
# #print(corrmatt)
# mask = np.array(corrmatt)
# mask[np.tril_indices_from(mask)]=False
# #print(mask)
# fig,ax = plt.subplots()
# sns.heatmap(corrmatt,mask=mask,square=True,annot=True)
# plt.show()


# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
# sns.regplot(x='temp',y='count',ax=ax1,data=train)
# sns.regplot(x='windspeed',y='count',ax=ax2,data=train)
# sns.regplot(x='humidity',y='count',ax=ax3,data=train)
# plt.show()

def concatenate(dt):
    return '{}-{}'.format(dt.year, dt.month)


train['year-month'] = train['datetime'].apply(concatenate)
# print(train)

# fig, (ax1) = plt.subplots(ncols=1)
# #sns.barplot(data=train,ax=ax1,x='year',y='count')
# sns.barplot(data=train,ax=ax1,x='year-month',y='count')
# plt.show()

trainWithoutOutliers = train[np.abs(train['count'] - train['count'].mean() <= (3 * train["count"].std()))]


# print(trainWithoutOutliers)

# fig, axis = plt.subplots(ncols=2,nrows=2)
# sns.distplot(train['count'],ax=axis[0][0])
# stats.probplot(train['count'],dist='norm',fit=True,plot=axis[0][1])
# sns.distplot(np.log(trainWithoutOutliers['count']),ax=axis[1][0])
# stats.probplot(np.log1p(trainWithoutOutliers['count']),dist='norm',fit=True,plot=axis[1][1])
#
# plt.show()

# print(train['count'].count())


def rmsle(y, y_, convertExp=False):
    if convertExp:
        y = np.exp(y)
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


def rmsle2(predicted_values, actual_values):
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)

    # 평균을 낸다.
    mean_difference = difference.mean()

    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)

    return score

# fig,axis = plt.subplots()
# sns.barplot(data=train,x='windspeed',y='count')
# plt.show()


# fig,axis = plt.subplots(nrows=2)
# plt.sca(axis[0])
# plt.xticks(rotation=30, ha = 'right')
# plt.sca(axis[1])
# plt.xticks(rotation=30, ha = 'right')
# axis[0].set(ylabel='count',title='test')
# sns.countplot(data=train,x='windspeed',ax=axis[0])
# sns.barplot(data=train,x='windspeed',y='count',ax=axis[1])
# plt.show()


from sklearn.ensemble import RandomForestClassifier


def predict_windspeed(data):
    wind0 = data.loc[data['windspeed'] == 0]
    windnot0 = data.loc[data['windspeed'] != 0]
    #features_windspeed=['season','weather','temp','atemp','humidity','month','year']
    features_windspeed=['season','weather','humidity','month','temp','year','atemp']
    windnot0['windspeed']=windnot0['windspeed'].astype('str')
    rfModel_wind = RandomForestClassifier()
    rfModel_wind.fit(windnot0[features_windspeed],windnot0['windspeed'])
    wind0values = rfModel_wind.predict(X=wind0[features_windspeed])
    predictwind0 = wind0
    predictwindnot0 = windnot0
    predictwind0['windspeed']=wind0values
    data = predictwindnot0.append(predictwind0)
    data['windspeed'] = data['windspeed'].astype('float')
    data.reset_index(inplace=True)
    data.drop('index',inplace=True,axis=1)
    return data

train = predict_windspeed(train)


# fig,axis = plt.subplots()
# plt.sca(axis)
# plt.xticks(rotation=30)
# sns.countplot(data=train,ax=axis,x='windspeed')
#
# plt.show()


categorical_feature_names=['season','holiday','workingday','weather','dayofweek','month','year','hour']

for i in categorical_feature_names:
    train[i]= train[i].astype('category')

features=['season','weather','temp','atemp','humidity','windspeed','year','hour','dayofweek','holiday','workingday']

X_train = train[features]
#print(X_train)

#label_name = 'count'
Y_train = train['count']
#print(Y_train)

from sklearn.metrics import make_scorer

rmsle_scorer = make_scorer(rmsle)
#print(rmsle_scorer)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10,shuffle=True,random_state=0)
#print(k_fold)

from sklearn.ensemble import RandomForestRegressor


#
# max_depth_list=[]
# model = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=0)
# #print(model)
#
#
# score = cross_val_score(model,X_train,Y_train,cv=k_fold,scoring=rmsle_scorer)
# score=score.mean()
# #print('Score={0:.5f}'.format(score))


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings

# pd.options.mode.chained_assignment=None
# warnings.filterwarnings('ignore',category=DeprecationWarning)
# lModel = LinearRegression()
# Y_train_log = np.log1p(Y_train)
# lModel.fit(X_train,Y_train_log)
# predict = lModel.predict(X_train)
# print(predict)
# print(rmsle(np.exp(Y_train_log),np.exp(predict)))



from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
Y_train_log = np.log1p(Y_train)

# fig,(ax1,ax2) = plt.subplots(ncols=2)
# sns.distplot(Y_train,ax=ax1)
# sns.distplot(np.exp(predict),ax=ax2)
# plt.show()







# print(train)
# print(train.shape)
# print(train.info())
# print(train.head())
# print(train.temp)
# print(train.temp.describe())
# print(train.isnull().sum())


# msno.matrix(train,figsize=(12,5))
# plt.show()

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['weekday'] = test['datetime'].dt.weekday
# for i in test.groupby(test['weekday']):
#     if i[0]==5 or i[0]==6 :
#         test['weekend']=i[0]
# print(test)
test['hour'] = test['datetime'].dt.hour

# figure, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)

# sns.barplot(data=test,x='year',y='count',ax=ax1)
# sns.barplot(data=test,x='month',y='count',ax=ax2)
# sns.barplot(data=test,x='day',y='count',ax=ax3)
# sns.barplot(data=test,x='weekday',y='count',ax=ax4)
# #sns.barplot(data=test,x='year',y='count',ax=ax1)
# plt.show()

# sns.boxplot(data=test,y='count',orient='v',ax=ax1)
# sns.boxplot(data=test,y='count',x='season',ax=ax2)
# sns.boxplot(data=test,y='count',x='hour',ax=ax3)
# sns.boxplot(data=test,y='count',x='workingday',ax=ax4)
# plt.show()

test['dayofweek'] = test['datetime'].dt.dayofweek


# print(test.info())
# print(test)
# print(test['dayofweek'].value_counts())
# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
# fig.set_size_inches(18,25)
# sns.pointplot(data=test,y='count',x='hour',ax=ax1)
# sns.pointplot(data=test,y='count',x='hour',ax=ax2,hue='workingday')
# sns.pointplot(data=test,y='count',x='hour',ax=ax3,hue='dayofweek')
# sns.pointplot(data=test,y='count',x='hour',ax=ax4,hue='weather')
# sns.pointplot(data=test,y='count',x='hour',ax=ax5,hue='season')
# plt.show()

# corrmatt = test[['temp','atemp','casual','registered','humidity','windspeed','count']]
# #print(type([[['temp','atemp','casual','registered','humidity','windspeed','count']]]))
# #print(type(['temp','atemp','casual','registered','humidity','windspeed','count']))
#
# corrmatt = corrmatt.corr()
#
# #print(corrmatt)
# mask = np.array(corrmatt)
# mask[np.tril_indices_from(mask)]=False
# #print(mask)
# fig,ax = plt.subplots()
# sns.heatmap(corrmatt,mask=mask,square=True,annot=True)
# plt.show()


# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
# sns.regplot(x='temp',y='count',ax=ax1,data=test)
# sns.regplot(x='windspeed',y='count',ax=ax2,data=test)
# sns.regplot(x='humidity',y='count',ax=ax3,data=test)
# plt.show()

test['year-month'] = test['datetime'].apply(concatenate)
# print(test)

# fig, (ax1) = plt.subplots(ncols=1)
# #sns.barplot(data=test,ax=ax1,x='year',y='count')
# sns.barplot(data=test,ax=ax1,x='year-month',y='count')
# plt.show()

#testWithoutOutliers = test[np.abs(test['count'] - test['count'].mean() <= (3 * test["count"].std()))]


# print(testWithoutOutliers)

# fig, axis = plt.subplots(ncols=2,nrows=2)
# sns.distplot(test['count'],ax=axis[0][0])
# stats.probplot(test['count'],dist='norm',fit=True,plot=axis[0][1])
# sns.distplot(np.log(testWithoutOutliers['count']),ax=axis[1][0])
# stats.probplot(np.log1p(testWithoutOutliers['count']),dist='norm',fit=True,plot=axis[1][1])
#
# plt.show()

# print(test['count'].count())


# fig,axis = plt.subplots()
# sns.barplot(data=test,x='windspeed',y='count')
# plt.show()


# fig,axis = plt.subplots(nrows=2)
# plt.sca(axis[0])
# plt.xticks(rotation=30, ha = 'right')
# plt.sca(axis[1])
# plt.xticks(rotation=30, ha = 'right')
# axis[0].set(ylabel='count',title='test')
# sns.countplot(data=test,x='windspeed',ax=axis[0])
# sns.barplot(data=test,x='windspeed',y='count',ax=axis[1])
# plt.show()

#test = predict_windspeed(test)


# fig,axis = plt.subplots()
# plt.sca(axis)
# plt.xticks(rotation=30)
# sns.countplot(data=test,ax=axis,x='windspeed')
#
# plt.show()


categorical_feature_names=['season','holiday','workingday','weather','dayofweek','month','year','hour']

for i in categorical_feature_names:
    test[i]= test[i].astype('category')

features=['season','weather','temp','atemp','humidity','windspeed','year','hour','dayofweek','holiday','workingday']







X_test = test[features]
#print(X_test)

#label_name = 'count'
#Y_test = test['count']
#print(Y_test)

#rmsle_scorer = make_scorer(rmsle2)
#print(rmsle_scorer)

#k_fold = KFold(n_splits=10,shuffle=True,random_state=0)
#print(k_fold)

#max_depth_list=[]
#model = RandomForestRegressor(n_estimators=100,n_jobs=-1,random_state=0)
#print(model)


#score = cross_val_score(model,X_test,Y_test,cv=k_fold,scoring=rmsle_scorer)
#score=score.mean()
#print('Score={0:.5f}'.format(score))






# model.fit(X_train,Y_train)
# prediction = model.predict(X_test)
#print(prediction[0:10])

# fig, (ax1,ax2) = plt.subplots(ncols=2)
# sns.distplot(Y_train,ax=ax1,bins=50)
# sns.distplot(prediction,ax=ax2,bins=50)
# plt.show()

model.fit(X_train,Y_train_log)
predict_origin = model.predict(X_train)
score = rmsle(np.exp(Y_train_log),np.exp(predict_origin))
print(score)




prediction = model.predict(X_test)



submission = pd.read_csv('bike/sampleSubmission.csv')
#print(submission)
#print(len(submission['count']))
submission['count'] = np.exp(prediction)
#print(submission.head())
#submission.to_csv('submission_bike.csv',index=False)
submission.to_csv('score_{0:.05f}_submission_bike.csv'.format(score),index=False)