#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#2(a)
my_parser = argparse.ArgumentParser(prog='myls',
                                    description='List contents of  folder')
my_parser = argparse.ArgumentParser(prog='myls',
                                    usage='%(prog)s [options] path',
                                    description='List contents of  folder')
$ python myls.py
my_parser = argparse.ArgumentParser(description='List contents of  folder',  epilog='Enjoy your programs! :)')


# In[ ]:


#2(b)

import argparse
parser = .argparse.ArgumentParser()
parser.add_.argument('-i', action='append', nargs='+')
args = parser.parse_args()
 p = argparse.ArgumentParser()
 p.add_argument("-i", nargs=3, action='append')
_AppendAction(...)
 p.parse_args("-i a b c -i  e f -i g h i".split())
Namespace(i=[['a', 'b', 'c'], ['', 'e', 'f'], ['g', 'h', 'i']])
 p.add_argument("-i", type=lambda x: x.split(",", 2), action='append')
 print p.parse_args("-i a,b,c -i ,e -i g,h,i,j".split())
Namespace(i=[['a', 'b', 'c'], [‘e'], ['g', 'h', 'i,j']])
class TwoOrThree(argparse._AppendAction):
    def __call__(self, parser, namspace, values, option_string=None):
        if not (2 <= len(values) <= 3):
            raise argparse.ArgumentError(self,  takes 2 or 3 values, given"(option_string, len(values)))
        super(TwoOrThree, self).__call__(parser, namespace, values, option_string)

p.add_argument("-i", nargs='+', action=TwoOrThree)
[["inputt1_url", "inputt1_name", "inputt1_other"],
 ["inputt2_url", "inputt2_name", "inputt2_other"],
 ["inputt3_url", "inputt3_name"]]
["inputt1_url", "inputt1_name", "inputt1_other", "inputt2_url", "inputt2_name", "inputt2”]


# In[ ]:


#2c
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
cross_val_score(clf, iris.data, iris.target, cv=10)
...                           ……...        ……………..
array([ 1.     ,  0.96...,  0.84...,  0.96...,  0.96...,
        0.96...,  0.96...,  1.     ,  0.96...,  1.      ])


# In[ ]:


#2d

df = pd.read_csv('inputtfile.csv')

train, testt = train_test_split(df, test_size = 0.3, random_state = 4044)
val, testt = train_test_split(test, test_size = 2/3, random_state = 4044)

train.reset_index(drop = True, inplace = True)
val.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)


def regressionTree(train_df, val_df, depthParams, maxFeatParams):        
    

    modelDict = {}

    
    for depthh in depthParams:
        for max_feature in maxFeatParams:
            
            aTree = tree.DecisionTreeRegressor(max_depth = depthh, max_features = max_feature, random_state = 42).fit(train_df.drop(['y'], axis = 1), train_df['y'])

            y_predd = aTree.predict(val_df.drop(['y'], axis = 1))

            modelDict.update({str(depth) + ' ' + str(max_feature): {
                'model': aTree,
                'rmse': mean_squared_error(val_df['y'], y_pred, squaredd = False),
                'rsquared': aTree.score(val_df.drop(['y'], axis = 1), val_df['y'])
            }})

    return(modelDict)

step2out = regressionTree(train = train, val = val, depthParams = [6], maxFeatParams = [0.8, 0.6])

