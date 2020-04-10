import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../../Econometrics_data/textfiles/FERTIL3.raw', delim_whitespace=True, header=None)

gfr=df[0]

# Exercice 1 - Taux de naissance pour 1000 femmes 15-44

plt.plot(gfr)
# moyennes: 103.2, 103.0, 80.7
# variances: 18, 16, 17

# => série non stationnaire au vu de la moyenne (peut être déduit en voyant le graphe)

# Exercice 2 - correction de la moyenne

lgfr=np.log(gfr)
dl=np.diff(lgfr) # diff donne (y(t) - y(t-1))/y(t-1)
year=df[2]
n=len(year)
year1=year[1:n]
plt.plot(year1,dl)

# Maintenant ça semble être bon niveau stationnarité pour la moyenne
# En revanche la variance ne donne pas de stationnarité => facile à corriger. Ici on accepte la stationnarité.


# Exercice 3 - Calculer l’auto-covariance d’ordre 1, 10 et 20 de dl

dfl=pd.DataFrame(dl) # dl = diff(gfr)
dfl_1=dfl.shift(1)

dl_1=dfl_1[0] # conversion en séries
np.cov(dl[1:n],dfl_1[1:n])
np.corrcoef(dl[1:n],dl_1[1:n])


# Incorrect
dfl=pd.DataFrame(dl) # dl = diff(gfr)
dfl_1=dfl.shift(10)

dl_1=dfl_1[0] # conversion en séries
np.cov(dl[10:n],dfl_1[10:n])
np.corrcoef(dl[10:n],dl_1[10:n])

# Exercice 4

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
acf(dl, alpha=1/np.sqrt(len(dl)))
plot_acf(dl) # la bande bleue doit normalement être constante = ce qui est acceptable pour un bruit blanc
pacf(dl)
plot_pacf(dl)

# Pas compatible avec AM(1); AR(1) ou ARMA(1,1) candidats potentiels
# Autocorrélation parielle : la barre 3 provient peut être d'une saisonnalité
# à droite : première barre yt et yt-1 en fixant toutes les autres
# à gauche : corrélation brute en t et t-1

# Exercice 5

import statsmodels.tsa.api as smt

AIC=[]
BIC=[]

for i in range(20):
    mdl = smt.AR(dl).fit(maxlag=i)
    mdl.params

    AIC.append(mdl.aic)
    BIC.append(mdl.bic)

#plt.plot(range(20),AIC)
plt.plot(range(20),BIC)

# On prend la valeur minimale pour choisir (1?)