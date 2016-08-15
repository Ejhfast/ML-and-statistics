import theano
import theano.tensor as T
import numpy

test_seq = T.dvector('test_seq')
init_v = T.dscalar('init_v')
y = T.dvector('y')

# if calling scan with `sequences` then *current_value* will map on to each element of the sequence
# *previous_value* will map onto the value returned by the last iteration (by default, control this with outputs_info)
# the arguments in `non_sequences` will be passed to the function on each iteration
# but can also just use a global, e.g., "y" as shown here
def fancy(current_value, previous_output):
    return 2*current_value*previous_output*y[0]*(y[1]**2)

# use the "initial" key to set a base case (e.g., the previous value on the first step)
results, updates = theano.scan(fancy, outputs_info=[{"initial": init_v}], sequences=test_seq)

# generic formula for creating an executable function from a scan
sum_of_squares = theano.function(inputs=[test_seq, init_v, y], outputs=results, updates=updates)

print("test:", sum_of_squares([1.0,2.0,3.0],1.0,[2.0,2.0]))

# try computing the gradient

#unwrap return
ret_val = results[-1]

sum_of_squares_g = T.grad(ret_val, y)

d_sum_of_squares = theano.function([test_seq, init_v, y], sum_of_squares_g)

print(d_sum_of_squares([1.0,2.0,3.0],1.0,[2.0,2.0]))
