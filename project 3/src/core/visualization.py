import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import cm

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - ADASYN")
plt.show()