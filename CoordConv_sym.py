import mxnet as mx

# 2d CoordConv - 'An intriguing failing of convolutional neural networks and the CoordConv solution'
# MXNET symbolic implementation 
# author : wx. zhang 

def CoordConv2D(data, num_filter, kernel, stride, pad, no_bias, workspace, name):

    # Symbol.shape_array() not supported yet, so use Symbol.ones_like() for now

    prior=mx.sym.ones_like(data) # same shape matrix with '1's
    prior=mx.sym.max(prior,axis=(1)) # collapse RGB >> 1 to get the correct shape
    added_col=mx.sym.elemwise_add(mx.sym.argsort(prior,axis=2),prior) # +1, not sure whether necessary
    added_col=mx.sym.broadcast_div(added_col, mx.sym.max(added_col)) # normalize
    added_row=mx.sym.elemwise_add(mx.sym.argsort(prior,axis=1),prior)
    added_row=mx.sym.broadcast_div(added_row, mx.sym.max(added_row))
    added_layers=mx.sym.stack(added_col,added_row,axis=1)

    new_input = mx.sym.concat(data,added_layers,dim=1) # join everything as the new input for conv

    out = mx.sym.Convolution(data=new_input, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,no_bias=no_bias, workspace=workspace, name=name)

    return out