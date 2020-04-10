'''
****************** ECONOMETRIE - PARTIE 2 - SERIES TEMPORELLES ***************
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

'''
1. Importer les données du fichier quarterly.xls (corriger le problème éventuel d’observations
manquantes)
'''

df = pd.read_excel('../../Econometrics_data/quarterly.xls')
print('Existence of null values in dataframe: {}'.format(df.isnull().values.any()))
# no null value in dataframe

'''
2. Stationnariser la série de CPI en utilisant la méthode de régression qui inclue un terme de
tendance dont la forme fonctionnelle est à choisir (linéaire, quadratique, log, exponentielle,
…)
'''
cpi = df['CPI']
date = df['DATE']
plt.figure(figsize=(8, 6))
ax = plt.subplot()
myLocator = mticker.AutoLocator()
ax.xaxis.set_major_locator(myLocator)
plt.plot(date, cpi)
plt.title('CPI')
# La série n'est clairement pas stationnaire car on voit une tendance
# => la moyenne et la variance ne sont pas similaires en tout point

dcpi = np.diff(cpi) # diff donne (y(t) - y(t-1))
# la méthode des différences permet de supprimer la tendance temporelle
# intuitivement, si on a une tendance ça veut dire qu'il y a un coefficient
# de variation constant => si on le supprime, on supprime donc la tendance
plt.figure(figsize=(8, 6))
ax = plt.subplot()
myLocator = mticker.AutoLocator()
ax.xaxis.set_major_locator(myLocator)
plt.plot(date[1:], dcpi)
plt.title('Stationarisation avec la méthode des différences')

'''
3. Stationnariser la série de CPI en utilisant un moyenne mobile centrée 5x5.
'''

cpi_roll = cpi.rolling(window=5).mean() # la première cellule (indice 4) est la moyenne
# des 5 précédentes (inclusif)

cpi_mm = cpi - cpi_roll

plt.figure(figsize=(8, 6))
ax = plt.subplot()
ax.xaxis.set_major_locator(myLocator)
plt.plot(date, cpi_mm)
plt.title('Stationarisation avec la méthode moyenne mobile')

'''
4. Calculer inf, le taux d’inflation à partir de la variable CPI. Faire un graphique dans le temps de
inf. Commentez.
'''

inf = cpi.pct_change() * 100 # l'inflation est un taux de variation de l'indice CPI
inf = inf.dropna()

plt.figure(figsize=(8, 6))
ax = plt.subplot()
ax.xaxis.set_major_locator(myLocator)
plt.plot(date[1:],inf)
plt.title("Taux d" + "'" + "inflation (%)")

# On remarque des pics haussiers au moment des chocs pétroliers (1973-1979)
# On remarque un pic baissier au moment de la crise des subprimes (2008)

'''
5. Interpréter l'autocorrélogramme et l'autocorrélogramme partiel de inf. Quelle est la
différence entre ces deux graphiques ?
'''

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plot_acf(inf)
plot_pacf(inf)

# L'autocorrélogramme donne les corrélations entre X(t) et X(t-k)

# L'autocorrélogramme partiel donne les corrélations entre les résidus des régressions
# de X(t) et X(t-k) avec X(t-1) ... X(t-k+1)

'''
6. Quelle est la différence entre la stationnarité et l'ergodicité ? Pourquoi a-t-on besoin de ces
deux conditions. Expliquez le terme "spurious regression".
'''

# Stationnarité: même moyenne et variance en tout point

# Ergodicité: l'étude d'un processus peut se faire avec une seule trajectoire
# (trajectoire = un set de données)
# un autocorrélogramme décroissant est caractéristique d'un processus ergodique
# car cela montre que la correlation est faible pour un grand nombre de périodes

# Ergodicité => Stationnarité (attention réciproque fausse)

# On a besoin que la série soit stationnaire pour avoir l'ergodicité
# On a besoin de l'ergodicité pour pouvoir appliquer le théorème central limite
# et ainsi approcher l'espérance par la moyenne.
# L'ergodicité est aussi utile si on ne dispose que d'une seule variable
# -> on peut alors calculer l'autocorrelation sur cette variable à la place
# d'une correlation entre 2 variables.

# "spurious relationships": correlations fautives entre 2 process
# Process X peut être très correlé avec process Y cependant il peut
# ne pas y avoir de causalité entre les 2 process
# => la causalité vient d'une autre variable cachée
# pour détecter une régression fautive, il faut vérifier la stationarité du terme d'erreur
# intuitivement, s'il y a une tendance, c'est qu'on a mal expliqué la variable

'''
7. Faire le test Augmented Dickey Fuller pour inf en utilisant le critère AIC pour
déterminer le nombre de lags à inclure. Commenter
'''

# AIC mesure l'erreur du modèle
# On prend le score le plus bas => erreur faible

from statsmodels.tsa.ar_model import AutoReg

AIC=[]

for i in range(1,11):
    lags = np.arange(i)+1
    res = AutoReg(np.array(inf), lags = lags).fit()
    AIC.append(res.aic)

plt.figure(figsize=(8, 6))
plt.plot(range(1,11),AIC)
plt.title("AIC")

# On voit que le modèle est plus fiable avec un nombre de lag égal à 3

# Le test ADF permet de vérifier l'existence d'une racine unitaire. Si elle existe,
# alors le process n'est pas stationnaire.

from statsmodels.tsa.stattools import adfuller

res_af = adfuller(inf, maxlag=3)
print('ADF p-value: %f' % res_af[1])

# La p-valeur étant inférieure au seuil de 5%, on rejette H_0
# On ne peut donc pas confirmer la présence d'une racine unitaire
# => la série est stationnaire

'''
8. Proposer une modélisation AR(p) de inf, en utilisant tous les outils vus au cours.
'''

from statsmodels.graphics.tsaplots import acf

print('l'"autocorrélation à l'ordre 3 est {}".format(acf(inf)[3]))

'''
9. Estimer le modèle de la courbe de Philips qui explique le taux de chômage (Unemp) en
fonction du taux d’inflation courant et une constante.
'''

import statsmodels.regression.linear_model as sm

unem = df['Unemp']
y_unem = unem[1:]
const_philips = np.ones(len(y_unem))
x_philips = np.column_stack((const_philips, inf))
model = sm.OLS(y_unem,x_philips)
results_philips = model.fit()
print(results_philips.summary())

# On ne peut pas rejeter l'hypothèse de nullité du coefficient
# => la variable x1 n'est pas significative

'''
10. Tester l’autocorrélation des erreurs.
'''

# TP5 exercice 3
# Un test sur l'autocorrélation des erreurs est un car particulier
# d'un test d'hétéroscédasticité

u = np.array(results_philips.resid)
y_res = np.array(u[1:])
const = np.ones(len(y_res))
x_res = np.column_stack((const, u[:len(u)-1]))
model = sm.OLS(y_res,x_res)
results_autocorr = model.fit()
print(results_autocorr.summary())

# la p-valeur est inférieure à 5% => on peut rejeter l'hypothèse H_0
# => les erreurs sont bien autocorrélées
# => hétéroscédasticité

'''
11. Corriger l’autocorrélation des erreurs par la méthode vue en cours.
'''

def testHeteroscedasticite(x1,y,wls):
    const = np.ones(len(y))
    x = np.column_stack((const, x1))

    model = None
    if wls:
        h = np.sqrt(x1)
        model = sm.WLS(y, x, weight=1/h)
    else:
        model = sm.OLS(y,x)
    results = model.fit()

    u = np.array(results.resid)
    y_res = np.array(u[1:])
    const_res = np.ones(len(y_res))
    x_res = np.column_stack((const_res, u[:len(u)-1]))
    model = sm.OLS(y_res,x_res)
    results_heteroscedasticite = model.fit()

    return results_heteroscedasticite

# Transformation 1 : transformation en variables binaires
# => impossible car l'inflation est une variable continue

# Transformation 2 : utiliser le logarithme
log_inf = np.log(inf).fillna(-500)
print(testHeteroscedasticite(log_inf,y_unem, wls=False).summary())
# On ne peut toujours pas rejeter l'hypothèse d'homoscédasticité
# Cependant on remarque que la F-stat diminue, on peut donc conclure que
# le test est moins significatif

# Transformation 3 : Weighted Least Squares
print(testHeteroscedasticite(log_inf,y_unem, wls=True).summary())
# WLS ne change pas les résultats car il n'y a qu'une seule variable

# Transformation 4 : régressions par sous-groupes
fig = plt.figure(figsize=(8,8))
plt.scatter(inf,y_unem)
plt.ylabel("unemployment")
plt.xlabel("inflation")
plt.show()

indexes_group_1 = inf[inf >= 1].index
indexes_group_2 = inf[inf < 1].index

y_unem_1 = y_unem[indexes_group_1]
inf_1 = inf[indexes_group_1]
print(testHeteroscedasticite(inf_1,y_unem_1,wls=False).summary())

y_unem_2 = y_unem[indexes_group_2]
inf_2 = inf[indexes_group_2]
print(testHeteroscedasticite(inf_2,y_unem_2,wls=False).summary())
# L'hétéroscédasticité est toujours significative, cependant
# avec cette dernière transformation on parvient à diminuer
# sensiblement la statistique de Fisher
# Intuitivement, en séparant les variables en deux sous-groupes, on "casse"
# de la corrélation entre ces groupes

'''
12. Tester la stabilité de la relation chômage-inflation sur deux sous-périodes de taille identique.
'''

# On avait une relation non significative précédemment

idx_mid = 90 # trouvé après plusieurs essais
idx_gpe_1 = inf[:idx_mid].index
idx_gpe_2 = inf[idx_mid:].index

y_phil_gpe_1 = y_unem[idx_gpe_1]
x_phil_gpe_1 = inf[idx_gpe_1]

y_phil_gpe_2 = y_unem[idx_gpe_2]
x_phil_gpe_2 = inf[idx_gpe_2]

const_phil_1 = np.ones(len(y_phil_gpe_1))
x_phil_1 = np.column_stack((const_phil_1, x_phil_gpe_1))
model = sm.OLS(y_phil_gpe_1,x_phil_1)
results_phil_1 = model.fit()
print(results_phil_1.summary())

const_phil_2 = np.ones(len(y_phil_gpe_2))
x_phil_2 = np.column_stack((const_phil_2, x_phil_gpe_2))
model = sm.OLS(y_phil_gpe_2,x_phil_2)
results_phil_2 = model.fit()
print(results_phil_2.summary())

# On remarque qu'en séparant l'échantillon à l'indice 90 (date 1982Q3)
# la relation chômage-inflation est significative pour la première période
# et non significative pour la seconde période (à niveau 5%)

'''
13. Faites les tests changement de structure de Chow et détecter le point de rupture.
'''
from scipy.stats import f
u_phil = results_philips.resid
SSR = u_phil.T@u_phil

for i in range(20,len(inf)-10,5):
    idx_mid = i
    idx_gpe_1 = inf[:idx_mid].index
    idx_gpe_2 = inf[idx_mid:].index

    y_phil_gpe_1 = y_unem[idx_gpe_1]
    x_phil_gpe_1 = inf[idx_gpe_1]

    y_phil_gpe_2 = y_unem[idx_gpe_2]
    x_phil_gpe_2 = inf[idx_gpe_2]

    const_phil_1 = np.ones(len(y_phil_gpe_1))
    x_phil_1 = np.column_stack((const_phil_1, x_phil_gpe_1))
    model = sm.OLS(y_phil_gpe_1,x_phil_1)
    results_phil_1 = model.fit()
    u_phil_1 = results_phil_1.resid
    SSR1 = u_phil_1.T@u_phil_1

    const_phil_2 = np.ones(len(y_phil_gpe_2))
    x_phil_2 = np.column_stack((const_phil_2, x_phil_gpe_2))
    model = sm.OLS(y_phil_gpe_2,x_phil_2)
    results_phil_2 = model.fit()
    u_phil_2 = results_phil_2.resid
    SSR2 = u_phil_2.T@u_phil_2

    F=((SSR-(SSR1+SSR2))/(SSR1+SSR2))*(len(y_unem)-2*2)/2
    pval = f.sf(F,len(y_unem)-2*2,2)
    if(pval < 0.05):
        print('date:{} pval:{}'.format(date[idx_mid], pval))

# On remarque que les différents points de rupture sont cohérents avec le fait
# le premier groupe est généralement haussier (jusqu'en 1975), et le deuxième baissier
# Il y a aussi des points de rupture en 2007-2008 (crise financière)

'''
14. Estimer la courbe de Philips en supprimant l'inflation courante des variables explicatives mais
en ajoutant les délais d’ordre 1, 2, 3 et 4 de l’inflation et du chômage. Faire le test de
Granger de non causalité de l’inflation sur le chômage. Donnez la p-valeur.
'''

# Le test de Granger est un test de causalité prédictive
# On veut donc tester la significativité des variables avec délais 1, 2, 3, 4
# Cela revient à faire un test de Fisher sur ces 4 variables

# Modèle non contraint
inf_lag_4 = inf[:len(inf)-4]
inf_lag_3 = inf[1:len(inf)-3]
inf_lag_2 = inf[2:len(inf)-2]
inf_lag_1 = inf[3:len(inf)-1]
inf_lag_0 = inf[4:]
y_unem_lag = y_unem[4:]
const_philips_2 = np.ones(len(y_unem_lag))
x_philips_nc = np.column_stack((const_philips_2, inf_lag_0, inf_lag_1, inf_lag_2, inf_lag_3, inf_lag_4))
model = sm.OLS(y_unem_lag,x_philips_nc)
results_philips_nc = model.fit()
u_philips_nc = results_philips_nc.resid
SSR_NC = u_philips_nc.T@u_philips_nc

# Modèle contraint
x_philips_c = np.column_stack((const_philips_2, inf_lag_0))
model = sm.OLS(y_unem_lag,x_philips_c)
results_philips_c = model.fit()
u_philips_c = results_philips_c.resid
SSR_C = u_philips_c.T@u_philips_c

# Calcul de la p-valeur
n = len(y_unem_lag)
F = ((SSR_C-SSR_NC)/(6-2))/(SSR_NC/(n-6))
print("p-valeur pour test de Granger: {}".format(f.sf(F,6-2,n-6)))

# La p-valeur étant très faible, on peut rejeter H_0 sans risque
# Les délais 1, 2, 3 et 4 ont bien un pouvoir prédictif

'''
15. Représentez graphiquement les délais distribués et commentez. Calculer l’impact à long de
terme de l’inflation sur le chômage.
'''

fig = plt.figure(figsize=(8,8))
plt.plot(np.arange(4)+1,results_philips_nc.params[2:])

# On voit que le délai 4 impacte le plus le chômage