import theano
import theano.tensor as T
import numpy

X = T.dmatrix('X')
W = T.dmatrix('W')
i = T.ivector('i')

# scheme 1

w1 = numpy.matrix([[1,2],
                   [3,3]])
x1 = numpy.matrix([[1],
                   [1]])

multiply1 = W.dot(X)

w2 = numpy.matrix([[1,3],
                   [2,3]])

x2 = numpy.matrix([1,1])

multiply2 = X.dot(W)

m1 = theano.function([X,W], multiply1)
m2 = theano.function([X,W], multiply2)

print(m1(x1,w1))
print(m2(x2,w2))

index_column = W[:,i].reshape((2,1))
index_row = W[i,:]

i1 = theano.function([i,W], index_column)
i2 = theano.function([i,W], index_row)

print(i1([1],w1))
print(i2([1],w1))
