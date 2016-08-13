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
e_scores = T.exp(W.dot(X)+b)
cost = T.mean(T.sum(T.mul(-1 * T.log(e_scores / T.sum(e_scores,axis=0)),y),axis=1))+T.sum(W**2)

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

# use generalizable gradient descent function
Wt, bt = gradient_descent(cost, X, y, W, b, Xt, yt, learning_rate=0.001)

print("After {} iterations predicted:".format(iterations))
print(numpy.argmax(Wt*Xt,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
