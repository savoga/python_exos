import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import f

df = pd.read_csv('HPRICE3.raw', delim_whitespace=True, header=None)

# ********* Exercice 1 CORRECT

price=df[7]/100
plt.hist(price,'auto')

s=np.shape(price)
const=np.ones(s)
age=df[1]
nbh=df[3]
inst=df[5]
rooms=df[8]
area=df[9]
land=df[10]
baths=df[11]
dist=df[12]
y81=df[15]
y=price
X=np.column_stack((const,age,nbh,inst,rooms,area,land,baths,dist,y81))
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())
s=y81==0 # prix moyen des maisons en 78
p0=np.mean(price[s])
s=y81==1 # prix moyen des maisons en 81
p1=np.mean(price[s])
print(p1-p0)

# Augmentation moyen de prix est de 440 (différence absolue entre les deux prix)
# Augmentation du prix ajusté à la qualité (issu de la régression) est 358
# Le prix moyen surestime la variation des prix

# ********* Exercice 2 CORRECT - EXERCICE IMPORTANT

u=results.resid
u2=u**2
y=u2
model=sm.OLS(y,X)
results = model.fit() # régression linéaire pour expliquer la variance
print(results.summary())
# exemple: plus on s'éloigne de l'autoroute, plus la variance des prix diminue
# exemple 2 : acheter en 1981 (x9) donne de la volatilité (cépendant la variable n'est pas significative)

# F=13.53 stat de Fisher (significativité globale) ; on rejette

# Quand p valeur trop élevée (colonne P>|t|), pas de significativité pour chaque variable explicative

# Prob(F-stat) très faible
# --> on est tout à droite de F_99 (seuil de rejet) = sur la partie très plate de la distribution
# On rejette l'hypothèse de non hétéroscédasticité, donc on a hétéroscédasticité
# => la colonne std err est trop élevée
# => toutes les students sont trop faibles

# ********* Exercice 3 INCORRECT

y=price
X=np.column_stack((const,age,nbh,inst,rooms,area,land,baths,dist,y81))
model=sm.OLS(y,X)
results=model.fit()

v=results.resid
SSR0=v.T@v
X0=X
n,k=np.shape(X0)
X=np.column_stack((const,age,nbh,inst,rooms,baths,dist,y81))
y=u2
model=sm.OLS(y,X)
results=model.fit()
v2=results.resid
SSR1=v2.T@v2
F=((SSR1-SSR0)/2) / (SSR0 / (n-k))
f.sf(F,2,n-k)

# Rappel: quand Student pas significatif, attention Fisher peut être significatif

# ********* Exercice 4 CORRECT

bath2=baths==2
bath3=baths==3
bath4=baths==4
X=np.column_stack((const,age,nbh,inst,rooms,area,land,bath2,bath3,bath4,dist,y81))
y=price
model=sm.OLS(y,X)
results=model.fit()

u=results.resid
u2=u**2
y=u2
model=sm.OLS(y,X)
results = model.fit() # régression linéaire pour expliquer la variance
print(results.summary())

# On fait la significativité par rapport à une seule salle de bain
# On rejette car la Fisher est elevée

# ********* Exercice 5 CORRECT
# En utilisant la spécification de l’exercice 4, refaire le test d’hétéroscédasticité en
# utilisant log (area) et log (land).

bath2=baths==2
bath3=baths==3
bath4=baths==4
X=np.column_stack((const,age,nbh,inst,rooms,np.log(area),np.log(land),bath2,bath3,bath4,dist,y81))
y=price
model=sm.OLS(y,X)
results=model.fit()

u=results.resid
u2=u**2
y=u2
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

# En prenant le log, le test devient davantage significatif
# On remarque que les variables qui contribuent le plus à la variance sont x5 et x6

# ********* Exercice 6 CORRECT
# En utilisant la spécification de l’exercice 5, refaire le test d’hétéroscédasticité en
# utilisant y = log (price/100).

bath2=baths==2
bath3=baths==3
bath4=baths==4
X=np.column_stack((const,age,nbh,inst,rooms,np.log(area),np.log(land),bath2,bath3,bath4,dist,y81))
y=np.log(price)
model=sm.OLS(y,X)
results=model.fit()

u=results.resid
u2=u**2
y=u2
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

# On rejette à 5% mais on ne rejette pas à 1%

# ********* Exercice 7 CORRECT ?
# Utiliser la variable land pour pondérer les observations dans la spécification de
# l’exercice 6. Refaire le test d’hétéroscédasticité.

lland=df[18]
larea=df[17]
h=np.sqrt(lland)
y=np.log(price)
X=np.column_stack((const,age,nbh,inst,rooms,larea,lland,bath2,bath3,bath4,dist,y81))
model=sm.WLS(y,X,weight=1/h)
results = model.fit()
print(results.summary())

u=results.resid
u2=u**2
y=u2
model=sm.OLS(y,X)
results = model.fit()
print(results.summary())

# On rejette toujours à 5% mais on a des résultats quand même légèrement meilleurs

# ********* Exercice 8

# Faire le graphique en nuage de point entre log (price/100) et lland. Diviser
# l’échantillon en deux groupes en fonction de lland et refaire le test
# d’hétéroscédasticité pour les deux sous-groupes.

lprice=np.log(price/100)
plt.scatter(lland,lprice)
plt.xlabel("lland")
plt.ylabel("lprice")
plt.show()