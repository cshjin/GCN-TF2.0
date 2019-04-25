import matplotlib.pyplot as plt
import pandas as pd

""" plot gcn_with_alpha """
df = pd.read_csv('output_gcn_with_alpha.csv')
ax = df.boxplot(grid=False)
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Citeseer")
plt.show()


""" plot gcn_1, gcn_2, gcn_3 """
