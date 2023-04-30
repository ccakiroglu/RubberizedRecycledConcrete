from pandas import read_csv
from numpy import array
import shap
from matplotlib import pyplot
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25)
pyplot.rcParams.update({'font.size': 25})
pyplot.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\boldmath"
colnamesC=[r'$w/c$', r'$FM_{NCA}$', r'$NCA \text{ }[kg/m^3]$', \
       r'$FM_{RCA}$', r'$RCA\text{ }absorption\text{ }capacity$', r'$RCA \text{ }[kg/m^3]$', \
       r'$Replacement [\%]\text{ }RCA$', r'$FM_{NFA}$',  r'$NFA \text{ }[kg/m^3]$', \
       r'$FM_{CR}$', r'$CR \text{ }[kg/m^3]$',r'$Replacement [\%]\text{ }CR$', \
       r'$Fiber\text{ }[kg/m^3]$',r'$Fiber [\%]$', r'$No\text{ }Fiber$',r'$PP$', \
       r'$Steel$', r'$Steel\text{ }Tire\text{ }Wires$',r'$Age$', r'$f_c^{\prime}$']

Cdir='G:\\My Drive\\Papers\\2023\\RubberizedConcrete\\EXCELCSV\\Compressive2.csv'
df = read_csv(Cdir,header=0,names=colnamesC)

X=df[[r'$w/c$', r'$FM_{NCA}$', r'$NCA \text{ }[kg/m^3]$', \
       r'$FM_{RCA}$', r'$RCA\text{ }absorption\text{ }capacity$', r'$RCA \text{ }[kg/m^3]$', \
       r'$Replacement [\%]\text{ }RCA$', r'$FM_{NFA}$',  r'$NFA \text{ }[kg/m^3]$', \
       r'$FM_{CR}$', r'$CR \text{ }[kg/m^3]$',r'$Replacement [\%]\text{ }CR$', \
       r'$Fiber\text{ }[kg/m^3]$',r'$Fiber [\%]$', r'$No\text{ }Fiber$',r'$PP$', \
       r'$Steel$', r'$Steel\text{ }Tire\text{ }Wires$',r'$Age$']]
y=df[[r'$f_c^{\prime}$']]
import shap
from catboost import CatBoostRegressor
from matplotlib import pyplot

# train a CatBoost model
model = CatBoostRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model)
#explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

#shap.initjs()
#shap.plots.force(shap_values[0], matplotlib=True, show=False)
pyplot.xlabel('')
#pyplot.xlabel(r'$b$')
#pyplot.ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for$')
pyplot.tight_layout()
#shap.plots.beeswarm(shap_values, show=False,color_bar_label=r'$Feature\hspace{0.5em} value$' )
shap.summary_plot(shap_values,X,show=False)
# Get the current figure and axes objects.
fig, ax = pyplot.gcf(), pyplot.gca()
pyplot.gcf().axes[-1].set_aspect('auto')
pyplot.tight_layout()
pyplot.gcf().axes[-1].set_box_aspect(25) 
pyplot.grid(True)
pyplot.show()
#shap.plots.beeswarm(shap_values)
#shap.plots.scatter(shap_values[:,r'$h$'],show=False, color=shap_values)#color_bar_labels
#fig, ax = pyplot.gcf(), pyplot.gca()
#ax.set_xlabel(r'$SHAP\hspace{0.5em} value$', fontdict={"size":15})
#ax.set_ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for\hspace{0.5em}h$', fontdict={"size":25})
#ax.set_yticks(array([-300,-200,-100,0,100,200, 300, 400,500]))
#ax.set_xticks(array([0,250,500,750,1000,1250,1500,1750,2000]))
#ax.tick_params(axis='y', labelsize=25)
#ax.tick_params(axis='x', labelsize=25)
#ax.set_xlabel(r'$h$',fontdict={"size":25})
#pyplot.savefig('G:\\My Drive\\Papers\\2022\\CylindricalWall\\IMAGES\\XGBoostSHAP.svg')
