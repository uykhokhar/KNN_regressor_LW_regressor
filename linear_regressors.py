
# coding: utf-8

# In[10]:

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
get_ipython().magic('matplotlib inline')
mpl.rc('figure', figsize=[12,8])  #set the default figure size

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from sklearn import svm


# In[11]:

class KNNRegressor(sklearn.base.RegressorMixin):
    
    def __init__(self, k):
        self.k = k
        super().__init__()
        
    def fit(self, X,y):
        self.X = X
        self.y = y
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(X.reshape(-1,1))
        
    def predict(self, T):
        predictions = []
        _, neighbors = self.nn.kneighbors(T)
        regressor = LinearRegression()
        for i in range(T.shape[0]):
            regressor.fit(self.X[neighbors[i]], self.y[neighbors[i]])
            predictions.append(regressor.predict([T[i]]))
        return np.asarray(predictions)


# In[29]:

#debugged with Tony
class LWRegressor(sklearn.base.RegressorMixin):
    
    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__()

    def fit(self, X, y): 
        self.X = X
        self.y = y
        #***len is 200
        #n_rows = X.shape[0] 
        #print("n_rows ", n_rows)
        #weights = []
        #for i in range(n_rows):
        #    x = X[i]
        #    w = rbf_kernel(X=self.X.reshape(-1,1), Y=x, gamma=self.gamma)
        #    weights.append(w)
        #self.weights = weights
        

    def predict(self, T):
        predictions = []
        n_rows = X.shape[0]
        for x in T:
            #w = self.weights[i]
            w = rbf_kernel(X=self.X.reshape(-1,1), Y=x, gamma=self.gamma)
            
            reg = LinearRegression()
            reg.fit(self.X, self.y, sample_weight = w.flatten())
            p = reg.predict(x)[0]
            predictions.append(p)
        return predictions


# In[30]:

def f_func(x):
        return 3.0 + 4.0 * x - 0.05 * x**2

def generate_data(size=200):
    m = size
    X = np.sort(np.random.random(m) * 100)
    y = f_func(X) + (np.random.random(m) - 0.5) * 50
    return(X,y)


# In[31]:

u = np.linspace(0,100,300)
f = f_func(u)
X, y = generate_data()
X.shape


# In[32]:

lw_reg = LWRegressor(gamma = 0.025)
lw_reg.fit(X.reshape(-1,1), y)

lw_predictions = lw_reg.predict(u.reshape(-1,1))


# In[33]:

knn_reg = KNNRegressor(5)
knn_reg.fit(X.reshape(-1,1), y)
predictions = knn_reg.predict(u.reshape(-1,1))

print(len(predictions))
print(len(lw_predictions))
print(len(u))

plt.plot(u,f, 'r', label='underlying function')
plt.scatter(X, y, s=10, color='b', alpha=0.5, label='data')
plt.plot(u,predictions, color='g', label='knn linear regression')
plt.plot(u,lw_predictions, color='b', label='lw linear regression')
plt.legend()


# ### K-fold cross validation
# modified from my decision tree code

# In[35]:

def calc_error(predictions, y):
    #check predictions for equality with y
    #increment error count if inequality
    #return error fraction
    err = 0
    total = len(predictions)

    for i in range(total):
        err += (predictions[i] - y[i])**2
    return err/total

Xs = X
ys = y

kf = KFold(n_splits=3, shuffle=True)
kf.get_n_splits(Xs)

print(kf)

knn_err_training = np.zeros(11)
knn_err_test = np.zeros(11)

#cross validation for k-nearest neighbors
for train_index, test_index in kf.split(Xs):
    
    #f is real function, predictions compared against real function f
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xs[train_index], Xs[test_index]
    y_train, y_test = ys[train_index], ys[test_index]
    f_train, f_test = f[train_index], f[test_index] #f <- from real function
    
    #error at k = 0 very high
    knn_err_training[0] = 3000000
    knn_err_test[0] = 3000000
    
    for k in range(1,10):
        #X_train = pd.DataFrame(X_train)
        #y_train = pd.DataFrame(y_train)
        #X_test = pd.DataFrame(X_test)
        #y_test = pd.DataFrame(y_test)
        
        #fit knn for k and determine error
        knn_reg = KNNRegressor(k)
        knn_reg.fit(X_train.reshape(-1,1), y_train)
        pred = knn_reg.predict(X_train.reshape(-1,1))
        knn_err_training[k] += calc_error(pred, y_train)
        
        #determine error for test set
        pred_test = knn_reg.predict(X_test.reshape(-1,1))
        knn_err_test[k] += calc_error(pred_test, f_test)
    #print(knn_err_training)
    #print(knn_err_test)

# average errors
knn_err_training /= 3
knn_err_test /= 3

print("Training Error: {}".format(knn_err_training))
print("Test Error: {}".format(knn_err_test))



# In[36]:

param_range = np.arange(0,11)
print(param_range)

train_scores_mean3 = knn_err_training
test_scores_mean3 = knn_err_test

plt.title("Validation Curve with KNN")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0, 3000)
lw = 2
plt.plot(param_range, train_scores_mean3, label="Training score",
             color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean3, label="Cross-validation score",
             color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# ## 5

# In[43]:

def plt_max_margin(X, Y, kernel, C):
    #code from http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

    # fit the model
    clf = svm.SVC(kernel=kernel, C=C)
    clf.fit(X, Y)

    # get the separating hyperplane
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.ylim(0, 7)
    plt.xlim(1,5)
    title = "kernel={},  C={}".format(kernel, C)
    plt.title(title)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()

def plt_svc(X, Y, kernel, C): #for non-linear kernals 
    #code from http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

    # fit the model
    clf = svm.SVC(kernel=kernel, C=C)
    clf.fit(X, Y)

    # get the separating hyperplane
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.ylim(0, 7)
    plt.xlim(1,5)
    title = "kernel={},  C={}".format(kernel, C)
    plt.title(title)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()
    
#p1, p2 are points on max margin line, p3 = closest point
def calc_dist(p1, p2, p3):
    return np.abs((p2[1]-p1[1])*p3[0]-(p2[0]-p1[0])*p3[1] + p2[0]*p1[1] - p2[1]*p1[0])/(np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2))


# ### 5a

# The max margin separator appears to pass through $(x_1,y_1) =(2.5,0)$ and $(x_2,y_2) =(4,7)$. The closest point appears to be $(x_0,y_0) =(3,4)$. The distance between the line and point is given by 
# 
# $dist = \frac{|(y_2 - y_1)x_0 - (x_2 - x_1)y_0 + x_2y_1 - y_2x_1|}{\sqrt{(y_2 - y_1) + (x_2 - x_1)}}$
# 
# dist = 1.4142135623730949
# 
# 
# #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

# In[44]:

calc_dist((2,5,0), (4,7), (3,4))


# ### 5b

# In[45]:

X = np.asmatrix([[1.5,6], [2,1], [3,4], [3.5,2], [4,4], [4.5,4.5]])
Y = [0] * 3 + [1] * 3
kernel = 'linear'
C = 100
plt_max_margin(X, Y, kernel, C)


# In[46]:

#closest distance
p1 = (0,-10)
p2 = (5,10)
p3 = (3,4) #closest point
calc_dist(p1, p2, p3)


# ### 5c

# In[47]:

X = np.asmatrix([[1.5,6], [2,1], [3,4], [5,2], [4,4], [4.5,4.5]])
Y = [0] * 3 + [1] * 3
kernel = 'linear'
C = 100
plt_max_margin(X, Y, kernel, C)


# In[48]:

#closest distance
p1 = (3.5,6)
p2 = (3.5,1)
p3 = (3,4) #closest point
calc_dist(p1, p2, p3)


# ### 5d

# In[49]:


X = np.asmatrix([[1.5,6], [2,1], [3,4], [4.5,4.5], [3.5,2], [4,4], ])
Y = [0] * 4 + [1] * 2
kernel = 'linear'
C = 0.1
title = 'C = 0.1'
plt_max_margin(X, Y, kernel, C)

#For C = 0.1 #sorry, my code to auto-generate titles was not working


# In[50]:

C = 1
plt_max_margin(X, Y, kernel, C)
#for C = 1.0


# In[51]:

C = 10
plt_max_margin(X, Y, kernel, C)


# In[ ]:



