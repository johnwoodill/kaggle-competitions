import pandas as pd; pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def lasso_regression(y, X, alpha, models_to_plot={}):

    #Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X, y)
    y_pred = lassoreg.predict(X)

    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(X, y_pred)
        plt.plot(X, y,'.')
        plt.title('Plot for alpha: %.3g'%alpha)

    #Return the result in pre-defined format
    rss = sum((y_pred - y.values)**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

dat = pd.read_csv("/Users/john/Projects/kaggle-competitions/housing-prices/data/train.csv")

#X = dat.drop(['Id'], axis = 1)    
X = dat[['SalePrice', 'MSSubClass', 'YrSold', 'LotFrontage', 'LotArea']]
X2 = dat.drop('SalePrice', 1)
X2 = dat.drop('Id', 1)
y = dat[['SalePrice']]

X2 = pd.get_dummies(X2)
X2 = X2.dropna()
lassoreg = Lasso(alpha=0.3, normalize=True, max_iter=1e5)
lassoreg.fit(X2, y)

X.head()
X = X.dropna()

#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(y, X, alpha_lasso[i], models_to_plot)


lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
lassoreg.fit(X2, y)
y_pred = lassoreg.predict(X)

y = X[['SalePrice']]
X = X.drop(['SalePrice'], 1)
X2 = X[['MSSubClass', 'YrSold']]
X = X[['LotFrontage', 'LotArea']]
X2 = pd.get_dummies(X2, columns=['MSSubClass', 'YrSold'])

X = pd.concat([X, X2], axis=1)


sc = StandardScaler()
X_scale = sc.fit(X)
X = X_scale.fit_transform(X)

#y = np.array(y).reshape(-1, 1)
y_scale = sc.fit(y)
y = y_scale.fit_transform(y)
y
ksmod = Sequential()
ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
ksmod.add(Dense(8, activation='relu'))
ksmod.add(Dense(4, activation='relu'))
ksmod.add(Dense(1, activation='sigmoid'))
ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

#valid = pd.DataFrame(columns = ['epoch', 'batch', 'loss', 'mse'])
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
tdat = tdat[['MSSubClass', 'YrSold', 'LotFrontage', 'LotArea']]
tdat.head()

tdat = tdat.dropna()

tX2 = tdat[['MSSubClass', 'YrSold']]
tX = tdat[['LotFrontage', 'LotArea']]
tX2 = pd.get_dummies(tX2, columns=['MSSubClass', 'YrSold'])

tX = pd.concat([tX, tX2], axis=1)
tX = tX.dropna()
tX
tX = X_scale.transform(tX)
pred = ksmod.predict(tX)
pred
y_scale.inverse_transform(pred)
