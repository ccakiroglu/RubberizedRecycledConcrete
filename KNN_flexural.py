from pandas import read_csv, DataFrame
from numpy import absolute,arange,mean,std,argsort,sqrt,array
from sklearn.model_selection import train_test_split
#from xgboost.sklearn import XGBRegressor 
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from time import time
#from metrics import *
pyplot.rc('text',usetex=True)
pyplot.rcParams.update({'font.size': 25})
pyplot.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\boldmath"
scaler = MinMaxScaler()

colnamesF=[r'$w/c$', r'$NCA\text{ }FM$', r'$NCA \text{ }[kg/m^3]$', \
       r'$RCA\text{ }FM$', r'$RCA\text{ }absorption\text{ }capacity$', r'$RCA \text{ }[kg/m^3]$', \
       r'$Replacement [\%]\text{ }RCA$', r'$NFA\text{ }FM$',  r'$NFA \text{ }[kg/m^3]$', \
       r'$CR\text{ }FM$', r'$CR \text{ }[kg/m^3]$',r'$Replacement [\%]\text{ }CR$', \
       r'Fiber\text{ }[kg/m^3]',r'$Fiber [\%]$', r'$No\text{ }Fiber$',r'$PP$', r'$Steel$' \
       r'$Steel\text{ }Tire\text{ }Wires$',r'$Age$', r'$f_f$']

Fdir='G:\\My Drive\\Papers\\2023\\RubberizedConcrete\\EXCELCSV\\Flexural2.csv'
df = read_csv(Fdir,header=0,names=colnamesF)
data=df.values
X, y = data[:, :-1], data[:, -1]
print('X.shape:', X.shape,'y.shape', y.shape)
# split into train test sets
trainStart=time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

KNNModel= KNeighborsRegressor(n_neighbors=7, p=1, leaf_size=60, weights='distance', n_jobs=-1)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['algorithm']=['auto','ball_tree', 'kd_tree', 'brute'];
#space['leaf_size']=[10,20,30,40, 50, 60]
#space['eta']=[0,0.3,0.6,0.8,1]
#space['learning_rate']=[.02, 0.1, 0.2]
#space['subsample']=[0.5]
search = GridSearchCV(KNNModel, space, n_jobs=-1, cv=cv, refit=True)
result = search.fit(X_train, y_train)
best_model = result.best_estimator_
yhat_test = best_model.predict(X_test)
yhat_train = best_model.predict(X_train)
testEnd=time()
duration=testEnd-trainStart
print('duration = ',duration)
print('MAPE train= ',mean_absolute_percentage_error(y_train, yhat_train))
print('RMSE train= ',sqrt(mean_squared_error(y_train, yhat_train)))
print('MAE train= ',mean_absolute_error(y_train, yhat_train))
print('R2 train:',r2_score(y_train, yhat_train))
print('EVS train:',explained_variance_score(y_train, yhat_train));
#print('VAF train:',VAF(y_train, yhat_train))
#print('Ef train:',NASH_SUTCLIFFE(y_train, yhat_train))#original
print('MAPE test= ',mean_absolute_percentage_error(y_test, yhat_test))
print('RMSE test= ',sqrt(mean_squared_error(y_test, yhat_test)))
print('MAE test= ',mean_absolute_error(y_test, yhat_test))
print('R2 test:',r2_score(y_test, yhat_test))#original
print('EVS test:',explained_variance_score(y_test, yhat_test));
#print('VAF test:',VAF(y_test, yhat_test))#original
#print('Ef test:',NASH_SUTCLIFFE(y_test, yhat_test))#original
print('Best parameters are',search.best_params_)#original
print('Hyperparameters: ',best_model.get_params())
fig, ax = pyplot.gcf(), pyplot.gca()
ax.scatter(yhat_train,y_train, marker='o',facecolor='steelblue',edgecolor='steelblue', label=r'$KNN\text{ }train$')
ax.scatter(yhat_test,y_test, marker='s',facecolor='cyan',edgecolor='cyan', label=r'$KNN\text{ }test$')
xk=[0,12];yk=[0,12];ykPlus10Perc=[0,13.2];ykMinus10Perc=[0,10.8];
ax.tick_params(axis='x',labelsize=16)
ax.tick_params(axis='y',labelsize=16)
ax.plot(xk,yk, color='black')
ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
ax.grid(True)
ax.set_xticks(array([0,2,4,6,8,10,12]))
ax.set_xlabel(r'$f_{f,predicted}\hspace{0.5em}[MPa]$', fontsize=17)
ax.set_ylabel(r'$f_{f,test}\hspace{0.5em}[MPa]$', fontsize=17)
pyplot.legend(loc='upper left',fontsize=12)
pyplot.tight_layout()
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\XGBoost4v.svg')
#set aspect ratio to 1
ratio = 1.0
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
pyplot.show()
