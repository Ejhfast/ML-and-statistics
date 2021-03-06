import theano.tensor as T
import theano
import numpy

# bespoke gradient descent function for linear models

def gradient_descent(cost, X, y, W, b, Xt, yt, learning_rate=0.001, iterations=1000):
    # initialize zero weights:
    # assuming x data is [x-features * n-data]
    # and y data is [y-dimensions * n-data]
    Wt = numpy.matrix(numpy.zeros((yt.shape[0],Xt.shape[0])))
    bt = numpy.matrix(numpy.zeros((yt.shape[0],1)))
    # get cost gradients
    gwcost, gbcost = T.grad(cost, [W,b])
    # get executable derivatives
    d_costw = theano.function([X,y,W,b],gwcost)
    d_costb = theano.function([X,y,W,b],gbcost)

    run_cost = theano.function([X,y,W,b], cost)

    for i in range(0,iterations):
        w_update = d_costw(Xt,yt,Wt,bt)
        b_update = d_costb(Xt,yt,Wt,bt)
        Wt = Wt - (w_update*learning_rate)
        bt = bt - (b_update*learning_rate)
        print(run_cost(Xt,yt,Wt,bt))
    return (Wt, bt)
