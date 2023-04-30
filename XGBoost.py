from pandas import read_csv, DataFrame
from numpy import absolute,arange,mean,std,argsort,sqrt,array
from sklearn.model_selection import train_test_split, cross_val_score
#from xgboost.sklearn import XGBRegressor 
from xgboost import XGBRegressor
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

colnamesC=[r'$Water$', r'$Cement$', r'$w/c$', r'$FM_{NCA}$', r'$NCA \text{ }[kg/m^3]$', \
       r'$FM_{RCA}$', r'$RCA\text{ }absorption\text{ }capacity$', r'$RCA \text{ }[kg/m^3]$', \
       r'$Replacement [\%]\text{ }RCA$', r'$FM_{NFA}$',  r'$NFA \text{ }[kg/m^3]$', \
       r'$FM_{CR}$', r'$CR \text{ }[kg/m^3]$',r'$Replacement [\%]\text{ }CR$', \
       r'Fiber\text{ }[kg/m^3]',r'$Fiber [\%]$', r'$No\text{ }Fiber$',r'$PP$', \
       r'$Steel$', r'$Steel\text{ }Tire\text{ }Wires$',r'$Age$', r'$f_c^{\prime}$']

Cdir='G:\\My Drive\\Papers\\2023\\RubberizedConcrete\\EXCELCSV\\Compressive.csv'
df = read_csv(Cdir,header=0,names=colnamesC)
data=df.values
X, y = data[:, :-1], data[:, -1]
print('X.shape:', X.shape,'y.shape', y.shape)
# split into train test sets
trainStart=time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

XGBModel = XGBRegressor()
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['n_estimators']=[5,10,20,50,100,200,500]
#space['max_depth']=[2, 4, 6, 8,12];
space['max_depth']=[8];
#space['min_child_weight']=[38,22,11,3,68,400];
space['min_child_weight']=[3];
#space['colsample_bytree']=[0.1,0.3,0.5,0.99];
space['colsample_bytree']=[0.5];
#space['gamma']=[0,10,20];
space['gamma']=[0]; 
#space['reg_alpha']=[0,10,20];
space['reg_alpha']=[0];
#space['reg_lambda']=[0,10,20];
space['reg_lambda']=[10];
#space['n_estimators']=[5,21,45, 67, 98, 157, 250,301,350,400,450]
#space['learning_rate']=[0.1, 0.3,0.5,0.9]
space['learning_rate']=[0.9]
space['subsample']=[0.5]
search = GridSearchCV(XGBModel, space, n_jobs=-1, cv=cv, refit=True)
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
ax.scatter(yhat_train,y_train, marker='o',facecolor='blue',edgecolor='blue', label=r'$XGBoost\text{ }train$')
ax.scatter(yhat_test,y_test, marker='s',facecolor='fuchsia',edgecolor='fuchsia', label=r'$XGBoost\text{ }test$')
xk=[0,90];yk=[0,90];ykPlus10Perc=[0,99];ykMinus10Perc=[0,81];
ax.tick_params(axis='x',labelsize=16)
ax.tick_params(axis='y',labelsize=16)
ax.plot(xk,yk, color='black')
ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
ax.grid(True)
ax.set_xticks(array([0,20,40,60,80,100]))
ax.set_yticks(array([0,20,40,60,80,100]))
ax.set_xlabel(r'$f_{c,predicted}^{\prime}\hspace{0.5em}[MPa]$', fontsize=17)
ax.set_ylabel(r'$f_{c,test}^{\prime}\hspace{0.5em}[MPa]$', fontsize=17)
pyplot.legend(loc='upper left',fontsize=12)
pyplot.tight_layout()
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\XGBoost4v.svg')
#set aspect ratio to 1
ratio = 1.0
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
pyplot.show()
