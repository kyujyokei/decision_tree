import numpy as np
import csv
import datetime


"""
loads the csv file 

@ filename: string, the filename of csv, has to include ".csv"
@ has_label: boolean, true if the data contains label to verify prediction

"""

def load_file(filename):
    data = np.genfromtxt(filename, dtype=np.str, delimiter=",")
    data = data.astype(float) # sets the type of values to float


    label = data[:, :1] # extract labels from data
    # data = np.delete(data, 0, axis=1) # delete the labels
    for i in label:
        if i[0] == 3: i[0] = 1 # changes label 3 to 1, 5 to -1
        else: i[0] = -1;

    #data = np.insert(data, 0, values=1.0, axis=1) # insert bias term
    return data


train_d = load_file("pa3_train_reduced.csv")
# print(train_d, train_l)

# creates a set of all the unique values in a certain column
def unique_vals(data, col):
    return set([d[col] for d in data])
# print(unique_vals(train_d, 0))

# counts the number of each type of examples in a data set, which means all the possible labels
def class_counts(data):
    counts = {}
    for row in data:
        label = row[0] #where the label is located
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
# print(class_counts(train_d))
class_count = class_counts(train_d)

def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)
# print(is_numeric(1), is_numeric(0.5), is_numeric("Bui"))

class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s?" % (condition, str(self.value))

# print(train_d[0])
# q = Question(0, 0)
# print(q)
# example = train_d[0]
# print(q.match(example))

# splits the data regarding the question
def partition(data, question):
    # print("partition")
    true_rows, false_rows = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
# t, f = partition(train_d, Question(0, 0))
# print(t[1])

# calculates the gini impurity
def gini(data):
    counts = class_counts(data) #a list that counts the number of all the possible labels
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(data))
        impurity -= prob_of_lbl**2
    return impurity
# no_mixing = [['Apple'], ['Apple']]
# some_mixing = [['Apple'], ['Orange']]
# print(gini(no_mixing), gini(some_mixing))

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

fruit_data = [
    ['Apple', 'Green', 3],
    ['Apple', 'Yellow', 3],
    ['Grape', 'Red', 1],
    ['Grape', 'Red', 1],
    ['Lemon', 'Yellow', 3],
]
# current_uncertainty = gini(fruit_data)
# print(current_uncertainty)
# true_row, false_row = partition(fruit_data, Question(1, 'Red'))
# print(info_gain(true_row, false_row, current_uncertainty))
# print(true_row)

def find_best_split(data):
    print("find best")
    best_gain = 0
    best_question = None
    current_uncertainty = gini(data)
    n_features = len(data[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in data])
        # print(col)
        for val in values:
            # print("     ", val)
            question = Question(col, val)

            true_rows, false_rows = partition(data, question)

            if len(true_rows) == 0 or len(false_rows) == 0: continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question
# print(len(train_d[0]) - 1)
# best_gain, best_q = find_best_split(fruit_data)
# print(best_q)

class Leaf:
    def __init__(self, data):
        self.predictions = class_counts(data)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(data, height):
    gain, question = find_best_split(data)
    print("H:", height)
    if gain == 0 or height >= 20:
        print("H:", height)
        return Leaf(data)

    true_rows, false_rows = partition(data, question)

    true_branch = build_tree(true_rows, height + 1)
    false_branch = build_tree(false_rows, height + 1)

    # if t_h >= 20 or f_h >= 20: # control the height of the tree
    #     return Leaf(data)

    return Decision_Node(question, true_branch, false_branch)

my_tree = build_tree(fruit_data, 0)

def print_tree(node, spacing=" "):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing, str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# print_tree(my_tree)

def classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# print(classify(fruit_data[0], my_tree))

def print_leaf(counts):

    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

# print(print_leaf(classify(fruit_data[0], my_tree)))

print(datetime.datetime.now())
tree = build_tree(train_d, 0)
print(datetime.datetime.now())

# for row in fruit_data:
#     print(row[0], print_leaf(classify(row, my_tree)))
