import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import f

# Exercice 1 CORRECT

df = pd.read_csv('../../Econometrics_data/textfiles/intdef.raw', delim_whitespace=True, header=None)

fig, ax = plt.subplots()

year=df[0]
i3=df[1] # 3 mo. T bill rate
ax.plot(year,i3, label='3 mo. T bill rate')
inf=df[2] # CPI inflation rate
ax.plot(year,inf, label='CPI inflation rate')
deficit=df[5] # out - rec (deficit as % GDP)
ax.plot(year,deficit, label='out - rec (deficit as % GDP)')

leg = ax.legend();

# Exercice 2 CORRECT

n=len(inf)
inf_1=inf[0:n-1] # On ne prend pas la dernière valeur valeur (car c'est l'instant t qu'on cherche)
def_1=deficit[0:n-1]
y=i3[1:n] # On ne prend pas la première valeur
const=np.ones(n-1)
X=np.column_stack((const, inf_1,def_1))

model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

# Toutes les variables sont significatives (p valeur très faibles)

# Exercice 3 CORRECT

# Le test de causalité de Granger revient à faire un test de Fisher

u=results.resid
n=len(u)
u_1=u[0:n-1]
const=np.ones(n-1)
X=np.column_stack((const, u_1))
X=X[:,1]
y=u[1:n]

model=sm.OLS(y,list(u_1)
)
#model=sm.OLS(y,X)
results1 = model.fit()
print(results1.summary())

# Exercice 4 NON FAIT EN COURS

# Exercice 5

n=len(inf)
inf_1=inf[1:n-1] # On ne prend pas la dernière valeur (car c'est l'instant t qu'on cherche)
def_1=deficit[1:n-1]
inf_2=inf[0:n-2]
def_2=deficit[0:n-2]
y=i3[2:n] # On ne prend pas la première valeur
const=np.ones(n-2)
X=np.column_stack((const,inf_1,inf_2,def_1,def_2))

model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

d_inf=(results.params[1], results.params[2])
x=(1,2)
plt.bar(x,d_inf) # On voit que la première période est très significative

d_def=(results.params[3], results.params[4])
x=(1,2)
plt.bar(x,d_def) # On voit que la deuxième période est très significative

# Exercice 6 CORRECT

# On fait un test de Fisher en testant si Beta_1 = Beta_2 = 0

# Modèle non contraint

n=len(inf)
inf_1=inf[1:n-1]
def_1=deficit[1:n-1]
inf_2=inf[0:n-2]
def_2=deficit[0:n-2]
y=i3[2:n]
const=np.ones(n-2)
X0=np.column_stack((const,inf_1,def_1,inf_2,def_2))

model=sm.OLS(y,X0)
results = model.fit()
print(results.summary())

u=results.resid
SSR0=u.T@u

# Modèle contraint
X=np.column_stack((const,def_1,def_2))

model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

u=results.resid
SSR1=u.T@u

# Computation of Fisher stats
n,k=np.shape(X0)
F=((SSR1-SSR0)/2)/(SSR0/(n-k))
f.sf(F,2,n-k)