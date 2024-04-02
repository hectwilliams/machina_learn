from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
from graphviz import Source
dot_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'iris_tree.dot' ) 
dot_file_path_pdf = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'iris_tree' ) 

iris = load_iris()

x = iris['data']

y = iris['target']

feature_names = iris['feature_names']

class_names = iris['target_names']

tree_clf =  DecisionTreeClassifier(max_depth=2)

tree_clf.fit(x, y)

export_graphviz(tree_clf, out_file=dot_file_path, feature_names=feature_names[:] , class_names=class_names , rounded=True, filled=True )

Source.from_file(dot_file_path).render(filename=dot_file_path_pdf, view=True)



