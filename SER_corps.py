# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:41:29 2020

@author: slefebvr
"""

import openturns as ot
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

file_name="planCAO_111_11.csv"
x_tot=pd.read_csv(file_name,header=0,sep=';')
x_tot.describe()

file_name="y_phiphi_module.csv"
y_tot=pd.read_csv(file_name,header=None,sep=';')
y_tot=y_tot.T
y_tot.describe()
scaler = StandardScaler()
ynorm=scaler.fit_transform(y_tot)

pca = PCA(n_components=3)
pca.fit(ynorm)
print(pca.explained_variance_ratio_)
ypca=pca.transform(ynorm)

scaler = MinMaxScaler()
xnorm=scaler.fit_transform(x_tot)

df=pd.DataFrame(xnorm)
df=df.rename(columns={0: "Logive", 1: "Rnez", 2: "Lfente"})
df=df.rename(columns={3: "Pfente", 4: "Dengin", 5: "LBB", 6: "Pos"})
df=df.rename(columns={7: "Fleche_BA", 8: "Fleche_BF", 9: "Corde", 10: "H"})

df['y1']=ypca[:,0]
df['y2']=ypca[:,1]
df['y3']=ypca[:,2]
export_csv = df.to_csv (r'data_scaled2.csv', index = None, header=True, sep=';') 


file_name="data_scaled.csv"
xy_tot=pd.read_csv(file_name,header=0,sep=';')
xy_tot.describe()

import statsmodels.api as sm
from statsmodels.formula.api import ols

results = ols('y ~ Logive+Rnez+Lfente+Pfente+Dengin+LBB+Pos+Fleche_BA+Fleche_BF+Corde+H', data=df).fit()
aov_table = sm.stats.anova_lm(results, typ=2)
aov_table




####AS 
file_name="data_scaled2.csv"
xy_tot=pd.read_csv(file_name,header=0,sep=';')
xy_tot.describe()
y_tot = pd.concat([xy_tot.pop(x) for x in ['y1', 'y2','y3']], 1)
y_tot
distribution = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * 11)
dimension = distribution.getDimension()
distribution.setDescription(["Logive", "Rnez", "Lfente", "Pfente", "Dengin", "LBB", "Pos","Fleche_BA", "Fleche_BF", "Corde", "H"])
# Create X/Y data
ot.RandomGenerator.SetSeed(0)
size = 10000
inputDesign = ot.SobolIndicesExperiment(distribution, size, False).generate()

import sklearn.linear_model as lm
linreg = lm.LinearRegression()
from sklearn.preprocessing import PolynomialFeatures
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_inter = interaction.fit_transform(xy_tot)

linreg1 = lm.LinearRegression()
linreg1.fit(X_inter, y_tot['y1'])
linreg2 = lm.LinearRegression()
linreg2.fit(X_inter, y_tot['y2'])
linreg3 = lm.LinearRegression()
linreg3.fit(X_inter, y_tot['y3'])

import matplotlib.pyplot as plt
plt.plot(y_tot['y2'],linreg2.predict(X_inter),'o')

plt.plot(y_tot['y1'],linreg1.predict(X_inter),'o')

plt.plot(y_tot['y3'],linreg3.predict(X_inter),'o')


linreg1.score(X_inter,y_tot['y1'])
linreg2.score(X_inter,y_tot['y2'])
linreg3.score(X_inter,y_tot['y3'])

inputDesignb = interaction.fit_transform(inputDesign)

outputDesign=np.zeros((np.shape(inputDesign)[0],3))
outputDesign[:,0] = linreg1.predict(inputDesignb)
outputDesign[:,1] = linreg2.predict(inputDesignb)
outputDesign[:,2] = linreg3.predict(inputDesignb)

outputDesign=ot.Sample(outputDesign)


sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign.getMarginal([0]), size)

first_indice = sensitivityAnalysis.getFirstOrderIndices()
total_indice = sensitivityAnalysis.getTotalOrderIndices()
first_indiceIC = sensitivityAnalysis.getFirstOrderIndicesInterval()
total_indiceIC = sensitivityAnalysis.getTotalOrderIndicesInterval()

print(first_indice)
print(total_indice)
print(first_indiceIC)
print(total_indiceIC)

# Draw indices
sensitivityAnalysis.draw()

#analyse par composante
for j in range(1,3):
    sensitivityAnalysisb = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign.getMarginal([j]), size)
    bb = sensitivityAnalysisb.getFirstOrderIndices()
    tt = sensitivityAnalysisb.getTotalOrderIndices()
    first_indice=np.vstack((first_indice,bb))
    total_indice=np.vstack((total_indice,tt))


#plot
  

variables = ["Logive", "Rnez", "Lfente", "Pfente", "Dengin", "LBB", "Pos","Fleche_BA", "Fleche_BF", "Corde", "H"]
xxx = np.arange(len(variables))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(xxx - width, first_indice[0,:], width, color='red',label='Composante1')
rects2 = ax.bar(xxx , first_indice[1,:], width, color='blue',label='Composante2')
rects3 = ax.bar(xxx + width, first_indice[2,:], width, color='green',label='Composante3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Indices Principaux')
ax.set_xticks(xxx)
ax.set_xticklabels(variables)
ax.legend()



#plot
fig, ax = plt.subplots()
rects1 = ax.bar(xxx - width, total_indice[0,:], width, color='red',label='Composante1')
rects2 = ax.bar(xxx , total_indice[1,:], width, color='blue',label='Composante2')
rects3 = ax.bar(xxx + width, total_indice[2,:], width, color='green',label='Composante3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Indices Totaux')
ax.set_xticks(xxx)
ax.set_xticklabels(variables)
ax.legend()


#essai aggregation
# Compute first order indices using the Saltelli estimator
sensitivity_algorithm = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, size)
first_indiceag = sensitivity_algorithm.getAggregatedFirstOrderIndices()
total_indiceag = sensitivity_algorithm.getAggregatedTotalOrderIndices()
print(first_indiceag)
print(total_indiceag)
nr_bootstrap = 100
confidence_level = 0.95
# sensitivity_algorithm = ot.MartinezSensitivityAlgorithm(
# inputDesign, outputDesign, size)
sensitivity_algorithm.setBootstrapSize(nr_bootstrap)
sensitivity_algorithm.setConfidenceLevel(confidence_level)
sensitivity_algorithm.setUseAsymptoticDistribution(False)
interval_fo = sensitivity_algorithm.getFirstOrderIndicesInterval()
interval_to = sensitivity_algorithm.getTotalOrderIndicesInterval()
print("bootstrap intervals")
print("Aggregated first order indices interval = ", interval_fo)
print("Aggregated total order indices interval = ", interval_to)
graph = sensitivity_algorithm.draw()
graph
