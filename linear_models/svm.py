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
delta = T.dscalar('delta')

# We need b to broadcast across columns, e.g., each row of b corresponds to an individual class bias
b = T.addbroadcast(b,1)

# ONE-VS-ONE HINGE LOSS for SVM
# https://www.wikiwand.com/en/Hinge_loss#/Extensions

# set margin that best class must have over other classes
delta = 2
# compute current class predictions
scores = W.dot(X)+b
# get only the predictions for the correct classes
correct_scores = T.max(T.mul(y,scores),axis=0)
# compute margins between class scores and the correct class
margins = T.maximum(0, scores - correct_scores + delta)
# remove margin between correct class and itself
remove_correct = T.mul(margins, 1-y)
# take the mean of the sum of the error + the regularization parameter
cost = T.mean(T.sum(remove_correct))+T.sum(W**2)

# create test X data 
Xt = numpy.matrix([[0,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,1],
                   [0,0,0,1,0]])


yt = numpy.matrix([[1,0,1,0,1],
                   [0,1,0,0,0],
                   [0,0,0,1,0]])

# use generalizable gradient descent function
Wt, bt = gradient_descent(cost, X, y, W, b, Xt, yt, learning_rate=0.001)

print("After {} iterations predicted:".format(iterations))
print(numpy.argmax(Wt*Xt,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
