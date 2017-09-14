#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'exercised_stock_options',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi'] 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#bonus-salary ratio

for employee, features in my_dataset.iteritems():
	if features['bonus'] == "NaN" or features['salary'] == "NaN":
		features['bonus_salary_ratio'] = "NaN"
	else:
		features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

#fraction from poi
for employee, features in my_dataset.iteritems():
	if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
		features['from_this_person_to_poi_fraction'] = 0
	else:
		features['from_this_person_to_poi_fraction'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

#fraction to poi
for employee, features in my_dataset.iteritems():
	if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
		features['from_poi_to_this_person_fraction'] = 0
	else:
		features['from_poi_to_this_person_fraction'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

features_list+=['bonus_salary_ratio','from_this_person_to_poi_fraction','from_poi_to_this_person']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.



from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

scaler = MinMaxScaler()
kbest = SelectKBest()
dtc = DecisionTreeClassifier()

knc = KNeighborsClassifier()
gnb=GaussianNB()

#pipeline steps
steps=[
      # ('min_max_scaler',scaler),
       ('f_select',kbest),
       ('Dtree',dtc),
      # ('knb',knc),
      # ('gausian_nb',gnb)
      ]
       
pipeline=Pipeline(steps)

#parameter to search in gridsearch
parameters = {'f_select__k':[2,3,4,5,6,7,8,9,10,11],
              'Dtree__criterion': ['gini','entropy'],
              'Dtree__splitter':['best','random'],
              'Dtree__min_samples_split':[2, 10, 20],
              'Dtree__max_depth':[10,15,20,25,30],
              'Dtree__max_leaf_nodes':[5,10,30],
              'Dtree__random_state':[42]
             # 'knb__n_neighbors':[1,2,3,4,5],
             # 'knb__leaf_size':[1, 10, 30, 60],
             # 'knb__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
              }





### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

 
from sklearn.grid_search import GridSearchCV


# Cross-validation for parameter tuning in grid search 
sss = StratifiedShuffleSplit(
    labels_train,
    n_iter = 20,
    test_size = 0.5,
    random_state = 0
    )

## Create, fit, and make predictions with grid search
gs = GridSearchCV(pipeline,
	            param_grid=parameters,
	            scoring="f1",
	            cv=sss,
                  error_score=0
                )
gs=gs.fit(features_train, labels_train)



# Pick the classifier with the best tuned parameters
clf = gs.best_estimator_
print "\n", "Best parameters are: ", gs.best_params_, "\n"

labels_predictions = clf.predict(features_test)



# Print features selected and their importances

features_selected=[features_list[i+1] for i in clf.named_steps['f_select'].get_support(indices=True)]

importances = clf.named_steps['Dtree'].feature_importances_
scores = clf.named_steps['f_select'].scores_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(len(features_selected)):
    print "feature no. {}: {} ({})({})".format(i+1,features_list[indices[i]],importances[indices[i]],scores[indices[i]])


test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)