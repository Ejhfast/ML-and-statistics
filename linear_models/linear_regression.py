import theano.tensor as T
import theano
import numpy
from util import gradient_descent

iterations = 1000
x_dim = 5
y_dim = 2
n_data = 100

# theano data types
y = T.dmatrix('y')
X = T.dmatrix('X')
W = T.dmatrix('W')
b = T.dmatrix("b")

# We need b to broadcast across columns, e.g., each row of b corresponds to an individual class bias
b = T.addbroadcast(b,1)

# sum of squares cost function for linear regression
cost = T.sum((y-(T.dot(W,X)+b))**2)
# for L1 or L2 regularization, add T.sum(W**2) or T.sum(|W|)

# create test X data (add another 1s column for b parameter)
Xt = numpy.matrix(numpy.random.rand(x_dim,n_data))

# randomly choose parameters that we will later search for (x_dim+1 for b parameter)
W_goal = numpy.matrix(numpy.random.rand(y_dim,x_dim))
b_goal = numpy.matrix(numpy.random.rand(y_dim,1))

# generate y-data based on X and parameters
yt = W_goal*Xt+b_goal

print("searching for:")
print("weights:", W_goal)
print("b:",b_goal)

Wt, bt = gradient_descent(cost, X, y, W, b, Xt, yt, learning_rate=0.001)

print("After {} iterations found:".format(iterations))
print(Wt)
print(bt)
