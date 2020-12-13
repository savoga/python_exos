# WARNING: this implementation is for CLASSIFICATION ONLY

def label_counts(rows):
    """Counts the number of each label in a dataset"""
    counts = {}
    for row in rows:
        label = row[-1] # LABEL IS ALWAYS THE LAST COLUMN
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)
# **********************************************************


# ********************* OBJECTS TO BUILD THE TREE **********
class SplitElement:
    """Object (column, value) used to split the rows"""
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def compare(self, example):
        # Compare example with this SplitElement.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "%s %s %s?" % (
            'col:'+str(self.column), condition, str(self.value))

class Leaf:
    """count LABELS in the leaf"""
    def __init__(self, rows):
        self.predictions = label_counts(rows)

class Decision_Node:
    """Contains the SplitElement and to the two child nodes"""
    def __init__(self,
                 split_element,
                 right_branch,
                 left_branch):
        self.split_element = split_element
        self.right_branch = right_branch
        self.left_branch = left_branch
# ************************************************************

# ********************* FUNCTIONS TO BUILD THE TREE **********
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.count_level = 0

    def partition(self, rows, split_element):
        """For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows' """
        left_rows, right_rows = [], []
        for row in rows:
            if split_element.compare(row):
                right_rows.append(row)
            else:
                left_rows.append(row)
        return right_rows, left_rows

    def gini(self, rows):
        """Calculate the Gini Impurity for a list of rows.
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity"""
        counts = label_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    def info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def find_best_split(self, rows):
        """Find the best SplitElement by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0
        best_split_element = None
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1
        for col in range(n_features):
            values = set([row[col] for row in rows])
            for val in values:
                split_element = SplitElement(col, val)
                right_rows, left_rows = self.partition(rows, split_element)
                if len(right_rows) == 0 or len(left_rows) == 0:
                    continue
                gain = self.info_gain(right_rows, left_rows, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_split_element = gain, split_element
        return best_gain, best_split_element

    def build_tree(self, rows): # =fit()
        gain, split_element = self.find_best_split(rows)
        if gain == 0 or self.count_level==self.max_depth:
            return Leaf(rows)
        right_rows, left_rows = self.partition(rows, split_element)
        # Recursively build the true branch.
        self.count_level += 1
        right_branch = self.build_tree(right_rows)
        # Recursively build the false branch.
        left_branch = self.build_tree(left_rows)
        return Decision_Node(split_element, right_branch, left_branch)

# **********************************************************

def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.split_element))
    print (spacing + '--> True:')
    print_tree(node.right_branch, spacing + "  ")
    print (spacing + '--> False:')
    print_tree(node.left_branch, spacing + "  ")

'''
# ********************* COMPARISON WITH SCIKIT LEARN ************

training_data = [ # = ROWS of the dataset
    ['France', 5, 'Premium'],
    ['Switzerland', 5, 'Premium'],
    ['UAE', 10, 'Advanced'],
    ['UAE', 10, 'Advanced'],
    ['Switzerland', 5, 'Basic'],
    ['France', 12, 'Basic']
]
header = ["Country", "Wealth", "Mandate"]

my_tree = DecisionTree(max_depth=2)
print_tree(my_tree.build_tree(training_data))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree

df_training_data = pd.DataFrame(data=training_data, columns=header)
clf = DecisionTreeClassifier(random_state=0, max_depth=2)
ohe = OneHotEncoder()
X = ohe.fit_transform(df_training_data[['Country']])
X = pd.concat([pd.DataFrame(data=X.toarray()),df_training_data['Wealth']], axis=1)
y = df_training_data['Mandate']
clf.fit(X, y)
tree.plot_tree(clf)
'''