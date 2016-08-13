import theano.tensor as T
import sys
import theano
import numpy
sys.path.append('..')
from linear_models.util import gradient_descent

# how big is the hidden layer?
d_hidden = 3

# output y
y = T.dmatrix('y')

# for mlp
X = T.dmatrix('X')
W = T.dmatrix('W')
b = T.dmatrix('b')
b = T.addbroadcast(b,1)

# for softmax on top
X2 = T.dmatrix('X2')
W2 = T.dmatrix('W2')
b2 = T.dmatrix('b2')
b2 = T.addbroadcast(b2,1)


# the hidden layer using tanh activation
mlp = T.tanh(W.dot(X)+b)
# now do softmax logistic regression on top:
# the rest of this cost function is normal logistic regression on top of mlp
scores = W2.dot(mlp)+b2
# take exponentials
exp_scores = T.exp(scores)
# normalize scores by other class predictions
adjusted_scores = -1 * T.log(exp_scores / T.sum(exp_scores, axis=0))
# zero out the every class except the correct one
adjusted_y = T.mul(adjusted_scores, y)
# mean of sum and regularization
cost = T.mean(T.sum(adjusted_y, axis=1)) + T.mean(T.sum(W**2))#+T.sum(W2**2)

# create test X data
Xt = numpy.matrix([[0,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,1],
                   [0,0,0,1,0]])


yt = numpy.matrix([[1,0,1,0,1],
                   [0,1,0,0,0],
                   [0,0,0,1,0]])

# this is SUPER important: weights need to be initialized randomly, will fail if initialized to 0s
# and the specific random interval can also be important: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
rng = numpy.sqrt(6.0/(d_hidden+Xt.shape[0]))

Wt = numpy.matrix(numpy.random.uniform(low=-1.0*rng, high=rng, size=(d_hidden, Xt.shape[0])))
bt = numpy.matrix(numpy.zeros((d_hidden,1)))

Wt2 = numpy.matrix(numpy.zeros((yt.shape[0],d_hidden)))
bt2 = numpy.matrix(numpy.zeros((yt.shape[0],1)))

# get cost gradients
gwcost, gbcost, gw2cost, gb2cost = T.grad(cost,[W,b,W2,b2])

# get executable derivatives
d_costw = theano.function([X,y,W,b,W2,b2],gwcost)
d_costb = theano.function([X,y,W,b,W2,b2],gbcost)
d_costw2 = theano.function([X,y,W,b,W2,b2],gw2cost)
d_costb2 = theano.function([X,y,W,b,W2,b2],gb2cost)
run_cost = theano.function([X,y,W,b,W2,b2], cost)

learning_rate = 0.005

for i in range(0,10000):
    w_update = d_costw(Xt,yt,Wt,bt,Wt2,bt2)
    b_update = d_costb(Xt,yt,Wt,bt,Wt2,bt2)
    w2_update = d_costw2(Xt,yt,Wt,bt,Wt2,bt2)
    b2_update = d_costb2(Xt,yt,Wt,bt,Wt2,bt2)
    Wt = Wt - (w_update*learning_rate)
    bt = bt - (b_update*learning_rate)
    Wt2 = Wt2 - (w2_update*learning_rate)
    bt2 = bt2 - (b2_update*learning_rate)
    print(run_cost(Xt,yt,Wt,bt,Wt2,bt2))

print("After {} iterations predicted:".format(10000))
print(numpy.argmax(Wt2*numpy.tanh(Wt*Xt+bt)+bt2,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
