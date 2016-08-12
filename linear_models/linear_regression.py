import theano.tensor as T
import theano
import numpy

iterations = 1000
x_dim = 5
y_dim = 2
n_data = 100

# theano data types
y = T.dmatrix('y')
X = T.dmatrix('X')
W = T.dmatrix('W')
b = T.dvector("b")

# sum of squares cost function for linear regression
cost = T.sum((y-(T.dot(X,W)+b))**2)
# for L1 or L2 regularization, add T.sum(W**2) or T.sum(|W|)

# compile executable cost
run_cost = theano.function([y,X,W,b],cost)

# get gradients in terms of W (parameter weights) and b
gwcost = T.grad(cost,W)
gbcost = T.grad(cost,b)

# executable derivative from the gradients
d_costw = theano.function([y,X,W,b],gwcost)
d_costb = theano.function([y,X,W,b],gbcost)

# create test X data (add another 1s column for b parameter)
Xt = numpy.matrix(numpy.random.rand(n_data,x_dim))

# randomly choose parameters that we will later search for (x_dim+1 for b parameter)
W_goal = numpy.matrix(numpy.random.rand(x_dim,y_dim))
b_goal = numpy.random.rand(y_dim)

# generate y-data based on X and parameters
yt = Xt*W_goal+b_goal
print(yt)

print("searching for:")
print("weights:", W_goal)
print("b:",b_goal)

# initialize search at zero for params
Wt = numpy.matrix(numpy.zeros((x_dim,y_dim)))
bt = numpy.zeros(y_dim)

learning_rate = 0.001

for i in range(0,iterations):
    w_update = d_costw(yt,Xt,Wt,bt)
    b_update = d_costb(yt,Xt,Wt,bt)
    Wt = Wt - (w_update*learning_rate)
    bt = bt - (b_update*learning_rate)
print("After {} iterations found:".format(iterations))
print(Wt)
print(bt)
