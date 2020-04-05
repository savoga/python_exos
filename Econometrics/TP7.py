import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('BARIUM.raw', delim_whitespace=True, header=None)
barium=df[0]

# Exercice 1

plt.plot(barium)

# La stationarité n'est pas respectée car on voit bien une tendance
# On voit qu'il y a des pics => probablement des saisonnalités
# Tendance relativement linéaire

# Exercice 2

# Retirer une tendance linéaire par régression OLS et représenter la série ajusté à la tendance o1

X = df[18] # time
X = sm.add_constant(X)
y = barium
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())
o1 = results.resid
plt.plot(o1)

# On a retiré la composante tendance

# Exercice 3

# Calculer la moyenne mobile d’ordre 12 centrée et faire un graphique de la série ajustée
# o2. Refaire ensuite une moyenne mobile d’ordre 2 et ajuster la série pour cette
# tendance. Faire un graphique de la série ajustée o3.

t2=barium.rolling(window=12).mean() # on perd 6 observations à la fin et 6 observations au début (autour de la moyenne)
plt.plot(t2)

o2=barium-t2
plt.plot(o2)

t3=t2.rolling(window=2).mean()
plt.plot(t3)

o3=barium-t3
plt.plot(o3)

# ce réajustement est celui fait pas le FMI

# Exercice 4

# A partir de la série o1, retirer les effets saisonnier par régression linéaire en prenant en
# compte des variables binaires pour les mois.

const=np.ones(len(o1))

feb=df[19]
mar=df[20]
apr=df[21]
may=df[22]
jun=df[23]
jul=df[24]
aug=df[25]
sep=df[26]
oct=df[27]
nov=df[28]
dec=df[29]

X = np.column_stack((const, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec))
#X = sm.add_constant(X)

y=o1

model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

o4 = results.resid
plt.plot(o4)

# Le résidu sera ajusté pour le terme de tendance et le terme saisonnalité

# Exercice 5 INCORRECT

# A partir de la série o3, calculer l’effet saisonnier par moyenne mobile 3 × 3. Ajuster la
# série de cet effet saisonnier et faire un graphique

t4=o3.rolling(window=3).mean() # on perd 6 observations à la fin et 6 observations au début (autour de la moyenne)
o5=o3-t4

t5=o3.rolling(window=3).mean() # on perd 6 observations à la fin et 6 observations au début (autour de la moyenne)
o5=o5-t5

plt.plot(o5)

# Exercice 6

# Faire le test de racine unitaires de la série barium avec 1 délai et avec 4 délais.
# Utiliser ensuite le critère AIC pour déterminer le nombre de lags.

from statsmodels.tsa.stattools import adfuller
adf_barium1=adfuller(barium, maxlag=1)
print(adf_barium1)
adf_barium4=adfuller(barium, maxlag=4, autolag=None)
print(adf_barium4)
adf_barium_aic=adfuller(barium, autolag='AIC')
print(adf_barium_aic)

# Exercice 7

# Faire le test de Chow d’absence de changement de structure pour deux sous-périodes
# de tailles égales (0:65; 66:130). Déterminer ensuit le point de rupture en faisant une
# boucle avec un taux de trim de 15% au début et à la fin de la période.

chempi=df[8]
gas=df[9]
rtwex=df[10]
n=len(barium)
const=np.ones(n)
y=barium
X=np.column_stack((const,chempi,gas,rtwex))
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())
u=results.resid
SSR=u.T@u