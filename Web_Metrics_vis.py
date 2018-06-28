import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

webMetrics = pd.read_csv('Web_Metrics_A.csv')


X = webMetrics.iloc[1100:, 2:].values
y = webMetrics.iloc[1100:, 1]


plt.plot(X)
plt.show()