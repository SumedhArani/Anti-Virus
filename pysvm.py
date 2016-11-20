import sklearn
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from sklearn.decomposition import PCA
import numpy as np
import clean

#input is a list of dictionary
input = clean.read()
#data, label  is a list of data and thier corresoponding target variables
#data is a 2D array
data, label = clean.make_list(input)
data = np.array(data)
label = np.array(label)

'''
#Extra trees classifier
#Tree-based estimators can be used to compute feature importances
#Which in turn can be used to discard irrelevant features 
print(data.shape)
etc = ExtraTreesClassifier()
etc = etc.fit(data, label)
model = SelectFromModel(etc, prefit=True)
data_new = model.transform(data)
print(data_new.shape)

#Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero.
#When the goal is to reduce the dimensionality of the data to use with another classifier
#they can be used along with other feature selectors to select the non-zero coefficients
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, label)
model = SelectFromModel(lsvc, prefit=True)
data_new = model.transform(data)
print(data_new.shape)
'''


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
