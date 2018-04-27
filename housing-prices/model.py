import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler

dat = pd.read_csv("data/train.csv")
dat = dat[['SalePrice', 'MSSubClass', 'YrSold', 'LotFrontage', 'LotArea']]
#dat = pd.get_dummies(dat, dummy_na=True)
dat.head()

dat = dat.dropna()

y = dat[['SalePrice']]
X2 = dat[['MSSubClass', 'YrSold']]
X = dat[['LotFrontage', 'LotArea']]
X2 = pd.get_dummies(X2, columns=['MSSubClass', 'YrSold'])

X = pd.concat([X, X2], axis=1)

X = X.dropna()
sc = StandardScaler()
X_scale = sc.fit(X)
X = X_scale.transform(X)

y = np.array(y).reshape(-1, 1)
y_scale = sc.fit(y)
y = y_scale.transform(y)

ksmod = Sequential()
ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
ksmod.add(Dense(8, activation='relu'))
ksmod.add(Dense(4, activation='relu'))
ksmod.add(Dense(1, activation='sigmoid'))
ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

ksmod.fit(X, y, epochs=100, batch_size = 20)

scores = ksmod.evaluate(X, y)
scores

# Test data
tdat = pd.read_csv("data/test.csv")
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
