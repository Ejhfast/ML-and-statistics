import theano.tensor as T
import sys
import theano
import numpy
sys.path.append('..')
from linear_models.util import gradient_descent

rng = numpy.random.RandomState(23)
random = T.shared_randomstreams.RandomStreams(rng.randint(999999))

# how big is the hidden layer?
d_hidden1 = 3
d_hidden2 = 3
reg_level = 0.0
dropout = 1 # higher = less dropout

# output y
y = T.dmatrix('y')

# for mlp1
X1 = T.dmatrix('X1')
W1 = T.dmatrix('W1')
b1 = T.dmatrix('b1')
b1 = T.addbroadcast(b1,1)

# for mlp2
W2 = T.dmatrix('W2')
b2 = T.dmatrix('b2')
b2 = T.addbroadcast(b2,1)

# for softmax on top
X3 = T.dmatrix('X3')
W3 = T.dmatrix('W3')
b3 = T.dmatrix('b3')
b3 = T.addbroadcast(b3,1)

# the hidden layers using tanh activation
mlp1 = T.mul(T.tanh(W1.dot(X1)+b1), random.binomial(n=1,p=dropout,size=(W1.shape[0],X1.shape[1]))) / dropout
mlp2 = T.mul(T.tanh(W2.dot(mlp1)+b2), random.binomial(n=1,p=dropout,size=(W2.shape[0],mlp1.shape[1]))) / dropout
# now do softmax logistic regression on top (can use "categorical_crossentropy", built into theano):
softmax = T.nnet.softmax(W3.dot(mlp2)+b3)
regularization = reg_level*(T.sum(W1**2)+T.sum(W2**2)+T.sum(T.abs_(W1))+T.sum(T.abs_(W2)))
cost = T.sum(T.mean(T.nnet.categorical_crossentropy(softmax,y))) + regularization

run_net = theano.function([X1,W1,b1,W2,b2,W3,b3], softmax)

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
rng = numpy.sqrt(6.0/(d_hidden1+Xt.shape[0]))

Wt1 = numpy.matrix(numpy.random.uniform(low=-1.0*rng, high=rng, size=(d_hidden1, Xt.shape[0])))
bt1 = numpy.matrix(numpy.zeros((d_hidden1,1)))

rng = numpy.sqrt(6.0/(d_hidden1+d_hidden2))

Wt2 = numpy.matrix(numpy.random.uniform(low=-1.0*rng, high=rng, size=(d_hidden2, d_hidden1)))
bt2 = numpy.matrix(numpy.zeros((d_hidden2,1)))

Wt3 = numpy.matrix(numpy.zeros((yt.shape[0],d_hidden2)))
bt3 = numpy.matrix(numpy.zeros((yt.shape[0],1)))

params = [X1,y,W1,b1,W2,b2,W3,b3]

# get cost gradients
gw1cost, gb1cost, gw2cost, gb2cost, gw3cost, gb3cost = T.grad(cost,params[2:])

# get executable derivatives
d_costw1 = theano.function(params,gw1cost)
d_costb1 = theano.function(params,gb1cost)
d_costw2 = theano.function(params,gw2cost)
d_costb2 = theano.function(params,gb2cost)
d_costw3 = theano.function(params,gw3cost)
d_costb3 = theano.function(params,gb3cost)

run_cost = theano.function(params, cost)

learning_rate = 0.005

params_t = [Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3]

for i in range(0,2000):
    w1_update = d_costw1(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    b1_update = d_costb1(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    w2_update = d_costw2(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    b2_update = d_costb2(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    w3_update = d_costw3(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    b3_update = d_costb3(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3)
    # print("w1",w1_update)
    # print("b1",b1_update)
    # print("w2",w2_update)
    # print("b2",b2_update)
    # print("w3",w3_update)
    # print("b3",b3_update)
    Wt1 = Wt1 - (w1_update*learning_rate)
    bt1 = bt1 - (b1_update*learning_rate)
    Wt2 = Wt2 - (w2_update*learning_rate)
    bt2 = bt2 - (b2_update*learning_rate)
    Wt3 = Wt3 - (w3_update*learning_rate)
    bt3 = bt3 - (b3_update*learning_rate)
    print(run_cost(Xt,yt,Wt1,bt1,Wt2,bt2,Wt3,bt3))

print("After {} iterations predicted:".format(10000))
print(numpy.argmax(Wt3.dot(numpy.tanh(Wt2.dot(numpy.tanh(Wt1.dot(Xt)+bt1))+bt2))+bt3,axis=0))
print("In comparison to desired predictions:")
print(numpy.argmax(yt,axis=0))
