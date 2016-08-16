import theano.tensor as T
import theano
import numpy

hidden_size = 4
learning_rate = 0.005

def make_char_embedding(str_list):
    all_char = sorted(set(''.join(str_list)))
    # 0 = null, 1 = start, 2 = end
    char2int = {c:(i+3) for i,c in enumerate(all_char)}
    int2char = {i:c for c,i in char2int.items()}
    int2char[0] = ''
    int2char[1] = '[' # symbol for sen start
    int2char[2] = ']'
    max_len = max([len(s) for s in str_list])+5
    char_dim = len(all_char)+3
    X = numpy.zeros((len(str_list), max_len, char_dim))
    y = numpy.zeros((len(str_list), max_len))
    for i,s in enumerate(str_list):
        X[i, 0, 1] = 1 # sentence start
        for c_i, c in enumerate(s):
            X[i, c_i+1, char2int[c]] = 1
            y[i, c_i] = char2int[c]
        X[i, len(s)+1, 2] = 1 # sentence end
        y[i, len(s)] = 2
    return char_dim, char2int, int2char, X, y

def initialize_weights(shape, n):
    rng = numpy.sqrt(6.0/n)
    return numpy.matrix(numpy.random.uniform(low=-1.0*rng, high=rng, size=shape))

char_dim, char2int, int2char, X, yd = make_char_embedding(["he said something pretty funny"])
print(int2char)

Ui_train = initialize_weights((hidden_size,hidden_size),char_dim)
Uf_train = initialize_weights((hidden_size,hidden_size),char_dim)
Uc_train = initialize_weights((hidden_size,hidden_size),char_dim)
Uo_train = initialize_weights((hidden_size,hidden_size),char_dim)
Wi_train = initialize_weights((char_dim,hidden_size),hidden_size)
Wf_train = initialize_weights((char_dim,hidden_size),hidden_size)
Wc_train = initialize_weights((char_dim,hidden_size),hidden_size)
Wo_train = initialize_weights((char_dim,hidden_size),hidden_size)

b_train = numpy.matrix(numpy.zeros((4,hidden_size)))
v_train = initialize_weights((hidden_size,char_dim),hidden_size)

init_s_dummy = numpy.zeros(hidden_size)#numpy.matrix(numpy.zeros((hidden_size,1)))

# W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o and V_o

# this x represents a single training example over time
x = T.dmatrix('x')

# rnn weights
U_i = T.dmatrix('U_i')
U_f = T.dmatrix('U_f')
U_c = T.dmatrix('U_c')
U_o = T.dmatrix('U_o')
W_i = T.dmatrix('W_i')
W_f = T.dmatrix('W_f')
W_c = T.dmatrix('W_c')
W_o = T.dmatrix('W_o')
V = T.dmatrix('V')

#bias matrix: b_i, b_f, b_c, b_o
b = T.dmatrix('b')

init_s = T.dvector('init_s') # memory needs initial state (0s)

# output y
y = T.vector('y',dtype='int64')

def forward_pass(curr_x, h_t_prev, c_t_prev):
    # forget gate: f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
    f_t = T.nnet.sigmoid(curr_x.dot(W_f) + h_t_prev.dot(U_f) + b[0,:])
    # input gate: i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
    i_t = T.nnet.sigmoid(curr_x.dot(W_i) + h_t_prev.dot(U_i) + b[1,:])
    # output gate: o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
    o_t = T.nnet.sigmoid(curr_x.dot(W_o) + h_t_prev.dot(U_o) + b[2,:])
    # c~ gate (candidate memory): \widetilde{C_t} = tanh(W_c x_t + U_c h_{t-1} + b_c)
    c_tilde = T.nnet.sigmoid(curr_x.dot(W_c) + h_t_prev.dot(U_c) + b[3,:])
    # memory state: C_t = i_t * \widetilde{C_t} + f_t * C_{t-1}
    c_t = T.mul(i_t, c_tilde) + T.mul(f_t, c_t_prev)
    # hidden output: h_t = o_t * tanh(C_t)
    h_t = T.mul(o_t, T.tanh(c_t))

    # out under classifier
    out = T.nnet.softmax(h_t.dot(V))[0]

    return out, h_t, c_t

# this function is obnoxious but powerful, see "scan_examples"
[output, h, c], updates = theano.scan(forward_pass, outputs_info=[None, {"initial":init_s},{"initial":init_s}], sequences=x)

cost = T.sum(T.nnet.categorical_crossentropy(output,y))

run_output = theano.function([x,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], output)
run_cost = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], cost)

# print(run_output(x_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy))


print("Computing gradients")
dU_i = T.grad(cost, U_i)
dU_f = T.grad(cost, U_f)
dU_o = T.grad(cost, U_o)
dU_c = T.grad(cost, U_c)
dW_i = T.grad(cost, W_i)
dW_f = T.grad(cost, W_f)
dW_o = T.grad(cost, W_o)
dW_c = T.grad(cost, W_c)
dV = T.grad(cost, V)
db = T.grad(cost, b)

print("Compiling derivative functions")
exe_dU_i = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dU_i)
exe_dU_f = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dU_f)
exe_dU_o = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dU_o)
exe_dU_c = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dU_c)
exe_dW_i = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dW_i)
exe_dW_f = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dW_f)
exe_dW_o = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dW_o)
exe_dW_c = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dW_c)
exe_dV = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], dV)
exe_db = theano.function([x,y,V,U_f,U_i,U_o,U_c,W_f,W_i,W_o,W_c,b,init_s], db)


# def sample(n):
#     start = [1]
#     for _ in range(n):
#         out = run_predict(start,u_train,v_train,w_train,init_s_dummy)
#         start.append(out[-1])
#     return ''.join([int2char[x] for x in start])

for i in range(0,10000):
    update_v, update_b = 0.0, 0.0
    update_u_i, update_w_i = 0.0, 0.0
    update_u_f, update_w_f = 0.0, 0.0
    update_u_o, update_w_o = 0.0, 0.0
    update_u_c, update_w_c = 0.0, 0.0
    batch_cost = 0.0
    for i_x in range(0,X.shape[0]):
        x_e = X[i_x,:,:]
        y_e = yd[i_x,:].astype(int)
        update_u_i += exe_dU_i(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_u_f += exe_dU_f(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_u_o += exe_dU_o(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_u_c += exe_dU_c(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_v += exe_dV(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_w_i += exe_dW_i(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_w_f += exe_dW_f(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_w_o += exe_dW_o(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_w_c += exe_dW_c(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        update_b += exe_db(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
        batch_cost += run_cost(x_e,y_e,v_train,Uf_train,Ui_train,Uo_train,Uc_train,Wf_train,Wi_train,Wo_train,Wc_train,b_train,init_s_dummy)
    Ui_train = Ui_train - (update_u_i*learning_rate)
    Uf_train = Uf_train - (update_u_f*learning_rate)
    Uo_train = Uo_train - (update_u_o*learning_rate)
    Uc_train = Uc_train - (update_u_c*learning_rate)
    v_train = v_train - (update_v*learning_rate)
    b_train = b_train - (update_b*learning_rate)
    Wi_train = Wi_train - (update_w_i*learning_rate)
    Wf_train = Wf_train - (update_w_f*learning_rate)
    Wo_train = Wo_train - (update_w_o*learning_rate)
    Wc_train = Wc_train - (update_w_c*learning_rate)
    print(batch_cost / X.shape[0])
    #print(sample(50))
