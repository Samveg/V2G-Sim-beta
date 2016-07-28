from __future__ import division

# General imports
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import V2G-Sim
v2gSimPath = os.path.join(os.getcwd(), "../../v2g_sim")
sys.path.append(os.path.join(v2gSimPath))

data = pd.read_excel('/Users/wangdai/V2G-sim2/v2g_sim/battery_degradation/forpandas.xlsx', header=None)
data = data.rename(columns={0: 'a1', 1: 'a2', 2: 'b1', 3: 'b2', 4: 'c1', 5: 'c2', 6: 'd1', 7: 'd2'})
data = data[(data.a1 < 200) & (data.a2 < 200) & (data.b1 < 200) &
            (data.b2 < 200) & (data.c1 < 200) & (data.c2 < 200) & (data.d1 < 200) & (data.d2 < 200)]
#print data
#print list(data.columns.values)
sns.boxplot(data=data)
sns.swarmplot(data=data)
plt.show()
