import theano.tensor as T
import theano
import numpy

hidden_size = 4
learning_rate = 0.005

def make_char_embedding(str_list):
    all_char = sorted(set(''.join(str_list)))
    # 0 = null, 1 = start, 2 = end
    char2int = {c:(i+3) for i,c in enumerate(all_char)}
    int2char = {(i+3):c for c,i in char2int.items()}
    max_len = max([len(s) for s in str_list])+2
    char_dim = len(all_char)+3
    X = numpy.zeros((len(str_list), max_len))
    y = numpy.zeros((len(str_list), max_len))
    for i,s in enumerate(str_list):
        X[i, 0] = 1 # sentence start
        for c_i, c in enumerate(s):
            X[i, c_i+1] = char2int[c]
            y[i, c_i] = char2int[c]
        X[i, len(s)] = 2 # sentence end
        y[i, len(s)-1] = 2
    return char_dim, char2int, int2char, X, y

def initialize_weights(shape, n):
    rng = numpy.sqrt(6.0/n)
    return numpy.matrix(numpy.random.uniform(low=-1.0*rng, high=rng, size=shape))

char_dim, char2int, int2char, X, yd = make_char_embedding(["hello x", "goobye", "whatever dude"])

u_train = initialize_weights((hidden_size,char_dim),char_dim)
w_train = initialize_weights((hidden_size,hidden_size),hidden_size)
v_train = initialize_weights((char_dim,hidden_size),hidden_size)
init_s_dummy = numpy.matrix(numpy.zeros((hidden_size,1)))

# this x represents a single training example over time
x = T.vector('x',dtype='int64')
# rnn weights
U = T.dmatrix('U')
V = T.dmatrix('V')
W = T.dmatrix('W')
init_s = T.dmatrix('init_s') # memory needs initial state (0s)

# output y
y = T.vector('y',dtype='int64')

def forward_pass(curr_x, prev_s):
    # memory at time t
    s_t = T.tanh(U[:, curr_x].reshape((U.shape[0],1)) + W.dot(prev_s))
    # output at time t
    o_t = T.nnet.softmax(V.dot(s_t).reshape(V.shape[0],1))
    return o_t[0], s_t

# this function is obnoxious but powerful, see "scan_examples"
[output,s], updates = theano.scan(forward_pass, outputs_info=[None,{"initial":init_s}], sequences=x)

prediction = T.argmax(output)
cost = T.sum(T.nnet.categorical_crossentropy(output,y))

run_cost = theano.function([x,y,U,V,W,init_s], cost)

print("Computing gradients")
dU = T.grad(cost, U)
dV = T.grad(cost, V)
dW = T.grad(cost, W)

print("Compiling derivative functions")
exe_dU = theano.function([x,y,U,V,W,init_s], dU)
exe_dV = theano.function([x,y,U,V,W,init_s], dV)
exe_dW = theano.function([x,y,U,V,W,init_s], dW)

for i in range(0,1000):
    update_u, update_v, update_w = 0.0, 0.0, 0.0
    batch_cost = 0.0
    for i_x in range(0,X.shape[0]):
        x_e = X[i_x,:].astype(int)
        y_e = yd[i_x,:].astype(int)
        update_u += exe_dU(x_e,y_e,u_train,v_train,w_train,init_s_dummy)
        update_v += exe_dV(x_e,y_e,u_train,v_train,w_train,init_s_dummy)
        update_w += exe_dW(x_e,y_e,u_train,v_train,w_train,init_s_dummy)
        batch_cost += run_cost(x_e,y_e,u_train,v_train,w_train,init_s_dummy)
    u_train = u_train - (update_u*learning_rate)
    v_train = v_train - (update_v*learning_rate)
    w_train = w_train - (update_w*learning_rate)
    print(batch_cost / X.shape[0])
