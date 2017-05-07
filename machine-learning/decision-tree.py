import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO  
import pydot 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

input_file = "../sample-data/PastHires.csv"
df = pd.read_csv(input_file, header = 0)

# Map non-numerical values
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)

# Take feature columns only
features = list(df.columns[:6])

# Take "Hired" as outcome, build decision tree classifier models
y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

### Plot decision tree graph
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
pngStr = graph.create_png()

# treat the dot output string as an image file
sio = StringIO()
sio.write(pngStr)
sio.seek(0)
img = mpimg.imread(sio)

# plot the image
imgplot = plt.imshow(img, aspect='equal')
plt.show()

# Remarks on the graph
# In each decision, left is True, right is False
# "Samples" means the number under the decision
# "Values" means the result, eg. [4,5] means 4 is not hired(0) and 5 is hired (1)

### Predictions by random forest to avoid overfitting
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

# Predict employment of an employed 10-year veteran
print clf.predict([[10, 1, 4, 0, 0, 0]])
# Predict an unemployed 10-year veteran
print clf.predict([[10, 0, 4, 0, 0, 0]])