# mxnet-CoordConv

MXNET symbolic implementation of - 
'An intriguing failing of convolutional neural networks and the CoordConv solution'
(arxiv.org/abs/1807.03247)

usage:
1. download the .py file
2. from CoordConv_sym import CoordConv2D
3. in your model, replace mx.sym.Convolution() with CoordConv2D()
4. train model as usual

initially supports 2d-conv only, you may modify it for your own purposes
