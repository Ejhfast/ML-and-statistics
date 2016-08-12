import theano.tensor as T
import theano
import numpy

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

# compile executable cost
run_cost = theano.function([y,X,W,b],cost)

# get gradients in terms of W (parameter weights)
gwcost, gbcost = T.grad(cost,[W,b])

# executable derivative from the gradients
d_costw = theano.function([y,X,W,b],gwcost)
d_costb = theano.function([y,X,W,b],gbcost)

# create test X data (add another 1s row for b parameter)
Xt = numpy.matrix([[0,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,1],
                   [0,0,0,1,0]])


yt = numpy.matrix([[1,0,1,0,1],
                   [0,1,0,0,0],
                   [0,0,0,1,0]])



# initialize search at zero for params
Wt = numpy.matrix(numpy.zeros((y_dim,x_dim)))
bt = numpy.matrix(numpy.zeros((y_dim,1)))

learning_rate = 0.001

for i in range(0,iterations):
    w_update = d_costw(yt,Xt,Wt,bt)
    b_update = d_costb(yt,Xt,Wt,bt)
    print(run_cost(yt,Xt,Wt,bt))
    Wt = Wt - (w_update*learning_rate)
    bt = bt - (b_update*learning_rate)

print("After {} iterations predicted:".format(iterations))
print(numpy.argmax(Wt*Xt,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
