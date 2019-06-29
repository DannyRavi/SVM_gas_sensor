
# -*- coding:utf-8 -*-
import csv
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import seaborn as sns


os.getcwd()
os.listdir(os.getcwd())
#
# ────────────────────────────────────────────────────────────────── I ──────────
#   :::::: C R E A T E D A T A S E T : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────
#



sensor_data = pd.read_csv('rst.csv', delimiter=';')
df = pd.DataFrame.from_dict(sensor_data)
mymap = {'acetone': 1, 'ethanol': 2}
df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)
print(sensor_data.shape)
X_All = df.iloc[:, 1:7]
X = df[['baseline_average', 'fall_time']]#? Корреляция между "baseline_average" и "baseline_average" была замечена
Y = df['gas_type']
print(df)
#
# ────────────────────────────────────────────────────────────────────────── II ──────────
#   :::::: V I S U A L I Z E P U R E D A T A : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────
#


pd.plotting.scatter_matrix(X_All, c=Y ,alpha=0.9, cmap='brg', s=50 ,label=['B','O'])
handles = [plt.plot([],[],color=plt.cm.brg(i/1.), ls="", marker=".", \
                    markersize=np.sqrt(10))[0] for i in range(2)]
labels=["acetone", "ethanol"]
plt.legend(handles, labels, loc=(1.02,0))
plt.title("All data was shown")
plt.show()
#
# ──────────────────────────────────────────────────────────── III ──────────
#   :::::: E X E C U T E S V M : :  :   :    :     :        :          :
# ──────────────────────────────────────────────────────────────────────
#


clf = svm.SVC(kernel='linear')
clf.fit(X.values, Y.values)
ax = plot_decision_regions(X=X.values, y=Y.values, clf=clf, legend=2)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, 
          ['acetone', 'ethanol'], 
           framealpha=0.3, scatterpoints=1)
plt.title('SVM Decision Регион Граница')
plt.xlabel('baseline_average')
plt.ylabel('fall_time')
plt.show()

#
# ────────────────────────────────────────────── IV ──────────
#   :::::: E N D : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────
#

