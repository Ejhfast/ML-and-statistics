import theano.tensor as T
import theano
import numpy
from util import gradient_descent

iterations = 1000
x_dim = 5
y_dim = 3

# theano data types
y = T.dmatrix('y')
X = T.dmatrix('X')
W = T.dmatrix('W')
b = T.dmatrix('b')

# We need b to broadcast across columns, e.g., each row of b corresponds to an individual class bias
b = T.addbroadcast(b,1)

# cross entropy

# compute current class predictions
scores = W.dot(X)+b
expp = T.exp(scores)

nex = T.mul(-1 * T.log(expp / T.sum(expp,axis=1)),y)

ff = theano.function([X,W,b],T.sum(expp,axis=1))

cost = T.mean(T.sum(nex,axis=1))

# create test X data (add another 1s row for b parameter)
Xt = numpy.matrix([[0,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,1],
                   [0,0,0,1,0]])


yt = numpy.matrix([[1,0,1,0,1],
                   [0,1,0,0,0],
                   [0,0,0,1,0]])

Wt = numpy.matrix(numpy.zeros((yt.shape[0],Xt.shape[0])))
bt = numpy.matrix(numpy.zeros((yt.shape[0],1)))

print(theano.function([X,W,b],W.dot(X)+b)(Xt,Wt,bt))
print(theano.function([X,W,b],T.exp(W.dot(X)+b))(Xt,Wt,bt))
print(theano.function([X,W,b],T.sum(expp,axis=1))(Xt,Wt,bt))
print(theano.function([X,W,b],exp / T.sum(expp,axis=1))(Xt,Wt,bt))

# use generalizable gradient descent function
Wt, bt = gradient_descent(cost, X, y, W, b, Xt, yt, learning_rate=0.001)

print("After {} iterations predicted:".format(iterations))
print(numpy.argmax(Wt*Xt,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
