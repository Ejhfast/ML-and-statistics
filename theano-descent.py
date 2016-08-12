import theano.tensor as T
import theano
import numpy

iterations = 1000
x_dim = 5
y_dim = 1
n_data = 100

# theano data types
y = T.dmatrix('y')
X = T.dmatrix('X')
W = T.dmatrix('W')

# sum of squares cost function for linear regression
cost = T.sum((y-T.dot(X,W))**2)
# for L1 or L2 regularization, add T.sum(W**2) or T.sum(|W|)

# compile executable cost
run_cost = theano.function([y,X,W],cost)

# get gradients in terms of W (parameter weights)
gwcost = T.grad(cost,W)

# executable derivative from the gradients
d_cost = theano.function([y,X,W],gwcost)

# create test X data (add another 1s column for b parameter)
Xt = numpy.matrix(numpy.random.rand(n_data,x_dim+1))
Xt[:,x_dim] = 1 

# randomly choose parameters that we will later search for (x_dim+1 for b parameter)
W_goal = numpy.matrix(numpy.random.rand(x_dim+1,y_dim))

# generate y-data based on X and parameters
yt = Xt*W_goal

print("searching for:")
print("weights:", W_goal)

# initialize search at zero for params
Wt = numpy.matrix(numpy.zeros((x_dim+1,y_dim)))

learning_rate = 0.001

for i in range(0,iterations):
    w_update = d_cost(yt,Xt,Wt)
    Wt = Wt - (w_update*learning_rate)
print("After {} iterations found:".format(iterations))
print(Wt)
