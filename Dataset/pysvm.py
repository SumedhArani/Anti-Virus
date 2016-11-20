import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.decomposition import PCA
import numpy as np
import clean

#input is a list of dictionary
input = clean.read()
#data, label  is a list of data and thier corresoponding target variables
#data is a 2D array
data, label = clean.make_list(input)
h = .02  # step size in the mesh

data = np.array(data)
label = np.array(label)
names = ["Nearest Neighbors", "Linear SVM","Decision Tree", "Random Forest", "Neural Net","Logistic Regression"]

classifiers = [KNeighborsClassifier(),SVC(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=20),MLPClassifier(solver='lbfgs', activation='tanh',alpha=1e-5, hidden_layer_sizes=(3, 370), random_state=5),linear_model.LogisticRegression(C=1e5)]
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
#linearly_separable = (X, y)
#datasets = [make_moons(noise=0.3, random_state=0),make_circles(noise=0.2, factor=0.5, random_state=1),linearly_separable]

def get_stats(classifier, res, label_test):
	
	actual_yes = list(label_test).count(-1)
	actual_no = list(label_test).count(+1)
	total = len(list(label_test))
	output = list(zip(res, label_test))
	true_pos = len(list(filter(lambda x:x[0]==-1 and x[0]==x[1], output)))
	true_neg = len(list(filter(lambda x:x[0]==1 and x[0]==x[1], output)))
	false_pos = len(list(filter(lambda x:x[0]==-1 and x[0]!=x[1], output)))
	false_neg = len(list(filter(lambda x:x[0]==1 and x[0]!=x[1], output)))
	pred_yes = list(res).count(-1)
	pred_no = list(res).count(1)

	recall = true_pos/actual_yes
	precision = true_pos/pred_yes
	accuracy = (true_pos+true_neg)/total
	misclassification_rate = (false_neg+false_pos)/total
	specificity = true_neg/actual_no

	print(classifier, "Stats: ")
	print("Accuracy: ", accuracy)
	print("Precision: ", precision)
	print("Recall: ", recall)
	print("Misclassification Rate: ", misclassification_rate)
	print("Specificity: ", specificity)

	return recall, precision, accuracy, misclassification_rate, specificity



data_train, data_test, label_train, label_test = train_test_split(data, label, 
	test_size=0.25, random_state=np.random.randint(0,10))

#K Nearest Neighbours
knn = KNeighborsClassifier()
knn.fit(data_train, label_train)
kres = knn.predict(data_test)
get_stats('K Nearest Neighbours', kres, label_test)
print('Parameters: ', knn.get_params(), end='\n\n')

#Support Vector Machines
clf = svm.SVC()
clf.fit(data_train, label_train)
cres = clf.predict(data_test)
get_stats("Support Vector Machines", cres, label_test)
print('Parameters: ', clf.get_params(), end='\n\n')

#Decision Tree
dt = tree.DecisionTreeClassifier()
dt.fit(data_train, label_train)
dres = dt.predict(data_test)
get_stats("Decision Tree", dres, label_test)
print('Parameters: ', dt.get_params(), end='\n\n')

#Neural Network
nnet = MLPClassifier(solver='lbfgs', activation='tanh', 
	alpha=1e-5, hidden_layer_sizes=(3, 370), random_state=5)
nnet.fit(data_train, label_train)
nres = nnet.predict(data_test)
get_stats("Neural Net", nres, label_test)
print('Parameters: ', nnet.get_params(), end='\n\n')

#Random Forest
rf = RandomForestClassifier(n_estimators=20)
rf.fit(data_train, label_train)
rres = rf.predict(data_test)
get_stats("Random Forest", rres, label_test)
print('Parameters: ', rf.get_params(), end='\n\n')

#Logistic regression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data_train, label_train)
lres = logreg.predict(data_test)
get_stats('Logistic Regression Classifier', lres, label_test)
print('Parameters: ', logreg.get_params(), end='\n\n')

#PCA


figure = plt.figure(figsize=(27, 9))
i = 1
X, y = data,label
  # preprocess dataset, split into training and test part
#    X, y = ds
#   X = StandardScaler().fit_transform(X)
#    data_train, data_test, label_train, label_test = \
#        train_test_split(X, y, test_size=.4, random_state=42)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)
#if ds_cnt == 0:
ax.set_title("Input data")
# Plot the training points
ax.scatter(data_train[:, 0], data_train[:, 1], c=label_train, cmap=cm_bright)
# and testing points
ax.scatter(data_test[:, 0], data_test[:, 1], c=label_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(data_train, label_train)
    score = clf.score(data_test, label_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(data_train[:, 0], data_train[:, 1], c=label_train, cmap=cm_bright)
    # and testing points
    ax.scatter(data_test[:, 0], data_test[:, 1], c=label_test, cmap=cm_bright,alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_cnt == 0:
        ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()

#Validate scores on fuzzying the dataset
#deprecated method
#try newer pdf
#can also try GridSearchCV
#kfold = cross_validation.KFold(len(data), n_folds=5)
#scores = [clf.fit(data[train], label[train]).score(data[test], label[test]) for train, test in kfold]
#print(scores)

#pca = PCA(n_components=2)
#pca.fit(data)

'''
Things that we can do-
1. ROC/AUC curve
2. Correlation coeffs
3. Comparision plots
4. Dimensionality reduction - PCA, IDWT, QDA, LDA(Refer scikit-learn)
5. Data visualisation
'''