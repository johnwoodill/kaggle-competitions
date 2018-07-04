import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler


dat = pd.read_csv("/Users/john/Projects/kaggle-competitions/housing-prices/data/train.csv")
X = dat.drop(['Id'], axis = 1)    
#dat = dat[['SalePrice', 'MSSubClass', 'YrSold', 'LotFrontage', 'LotArea']]
X = pd.get_dummies(X)
X.head()
X = X.dropna()

y = X[['SalePrice']]
#X = dat.drop(['SalePrice'])
#X2 = dat[['MSSubClass', 'YrSold']]
#X = dat[['LotFrontage', 'LotArea']]
#X2 = pd.get_dummies(X2, columns=['MSSubClass', 'YrSold'])

#X = pd.concat([X, X2], axis=1)


sc = StandardScaler()
X_scale = sc.fit(X)
X = X_scale.transform(X)

#y = np.array(y).reshape(-1, 1)
y_scale = sc.fit(y)
y = y_scale.transform(y)
y
ksmod = Sequential()
ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
ksmod.add(Dense(8, activation='relu'))
ksmod.add(Dense(4, activation='relu'))
ksmod.add(Dense(1, activation='sigmoid'))
ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

valid = pd.DataFrame(columns = ['epoch', 'batch', 'loss', 'mse'])
# Find best epoch and batch size
# for i in range(1, 10):
#     for j in range(1, 5):

#         # Fit with i epoch
#         lloss = ksmod.fit(X, y, epochs=i, batch_size=j)

#         # Build outdat
#         indat = pd.DataFrame([[i, j, lloss.history['loss'][-1], lloss.history['mean_squared_error'][-1]]], columns = ['epoch', 'batch', 'loss', 'mse'])
#         valid = valid.append(indat)


ksmod.fit(X, y, epochs=100, batch_size=20)

ypred = ksmod.predict(X)
ypred
y_scale.inverse_transform(ypred)
y_scale.inverse_transform(y)

error = (y_scale.inverse_transform(y) - y_scale.inverse_transform(ypred))
print(max(error), min(error))
scores = ksmod.evaluate(X, y)
scores

#---------------------------------
# Test data
tdat = pd.read_csv("/Users/john/Projects/kaggle-competitions/housing-prices/data/test.csv")
tdat.head()

tdat = tdat[['MSSubClass', 'YrSold', 'LotFrontage', 'LotArea']]
tdat.head()

tdat = tdat.dropna()

tX2 = tdat[['MSSubClass', 'YrSold']]
tX = tdat[['LotFrontage', 'LotArea']]
tX2 = pd.get_dummies(tX2, columns=['MSSubClass', 'YrSold'])

tX = pd.concat([tX, tX2], axis=1)
tX = tX.dropna()

tX = X_scale.transform(tX)
pred = ksmod.predict(np.array(tX))
pred
y_scale.inverse_transform(pred)
