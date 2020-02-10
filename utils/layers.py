#from keras import backend as K
from __future__ import division
from tensorflow.contrib.distributions import fill_triangular
#from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from pathlib import Path

libs_path = Path.home() / 'libs'

hungarian_module = tf.load_op_library(str(libs_path / 'munkres-tensorflow/hungarian.so'))
louvain_module = tf.load_op_library(str(libs_path / 'louvain-tensorflow/louvain.so'))

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.pooling import _GlobalPooling1D
float_type=tf.float32

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

def excl_range(size, excl):
    with ops.name_scope('excl_range', [size, excl]):
        rang = tf.expand_dims(tf.range(size), 0)
        excl = tf.reshape(excl, [1, -1])
        recurr = tf.constant([[0]])
        not_reduced_axes = tf.sets.set_union(tf.sets.set_difference(rang, excl), recurr)
        return not_reduced_axes.values

@ops.RegisterGradient("SparseReduceMax")
def _SpareReduceMaxGrad(op, out_grad):
    return _SparseReduceMinOrMaxGrad(op, out_grad)
    
@ops.RegisterGradient("SparseReduceMin")
def _SpareReduceMinGrad(op, out_grad):
    return _SparseReduceMinOrMaxGrad(op, out_grad)
    
def _SparseReduceMinOrMaxGrad(op, out_grad):
    sp_indices = op.inputs[0]
    sp_values = op.inputs[1]
    sp_shape = op.inputs[2]
    reduction_axes = op.inputs[3]
    output = op.outputs[0]
    
    # Handle keepdims
    output_shape_kept_dims = math_ops.reduced_shape(sp_shape, op.inputs[3])
    out_grad = array_ops.reshape(out_grad, output_shape_kept_dims)
    output = array_ops.reshape(output, output_shape_kept_dims)
    
    # Map input and output coefficients
    scale = sp_shape // math_ops.to_int64(output_shape_kept_dims)
    scaled_indices = sp_indices // scale
    
    # Map pooled values with corresponding max/min values
    sp_max_val = array_ops.gather_nd(output, scaled_indices)
    indicators = math_ops.cast(math_ops.equal(sp_values, sp_max_val), out_grad.dtype)
    grad_values = array_ops.gather_nd(out_grad, scaled_indices)
    
    # Compute the number of selected (maximum or minimum) elements in each
    # reduction dimension. If there are multiple minimum or maximum elements
    # then the gradient will be divided between them. 
    # (same as for MaxGrad)
    sp_indicators = sparse_tensor.SparseTensor(sp_indices, indicators, sp_shape)
    num_selected = array_ops.gather_nd(
        sparse_ops.sparse_reduce_sum(sp_indicators, axis=reduction_axes, keep_dims=True),
        scaled_indices
    )
    
    # (input_indices, input_values, input_shape, reduction_axes)
    return [None, math_ops.div(indicators, math_ops.maximum(num_selected, 1)) * grad_values, None, None]


# Optim
# 1. No edge weight and no global cost matrix
# 2. C op for masking
# 3. Sparse tensors
import sys, inspect
def custom_layers():
    return dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))

def flatten(l):
    return [x for y in l for x in y]

def setup(layer_class=None):
    if not layer_class:
        from keras.engine.topology import Layer
        layer_class = Layer
        
    # TODO: generalize wrt data_formats
    global GraphGlobalAveragePooling
    class GraphGlobalAveragePooling(layer_class):
        def __init__(self, data_format=None, **kwargs):
            self.data_format = data_format or K.image_data_format()
            super(GraphGlobalAveragePooling, self).__init__(**kwargs)
            
        def compute_output_shape(self, input_shapes):
            return (input_shapes[0][0], input_shapes[0][2])

        def call(self, inputs):
            feat, adj = inputs
            return tf.reduce_sum(feat,
                    axis=(1 if self.data_format == 'channels_last' else 2)
                    ) / tf.stop_gradient(tf.maximum(tf.reduce_sum(tf.matrix_diag_part(adj), keepdims=True, axis=1), 1.)) # gic or gci

        def get_config(self):
            config = {
                'data_format': self.data_format
            }
            base_config = super(GraphGlobalAveragePooling, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    
    global GraphMaxPooling
    def GraphMaxPooling(reduction_ratio):
        def f(inp):
            feat, adj = inp
            conversion_tensor = ClusteringAdjacency(reduction_ratio)([feat, adj])
            nfeat = GraphMaxPooling_()([feat, conversion_tensor])
            nadj = AdjacencyConversion()([adj, conversion_tensor])
            return nfeat, nadj
        return f
    
    global ClusteringAdjacency
    class ClusteringAdjacency(layer_class):
        def __init__(self, reduction_ratio, data_format=None, **kwargs):
            self.reduction_ratio = reduction_ratio
            self.data_format = data_format or K.image_data_format()
            super(ClusteringAdjacency, self).__init__(**kwargs)
        
        def build(self, input_shape):
            super(ClusteringAdjacency, self).build(input_shape)
        
        def compute_output_shape(self, input_shape):
            return input_shape[1]
        
        def call(self, batch):
            feat, adj = batch
            bool_adj = tf.cast(adj, tf.bool)
            
            if self.data_format == 'channels_last':
                weights = tf.einsum('gic,gjc->gij', feat, feat)
            else:
                weights = tf.einsum('gci,gcj->gij', feat, feat)
            mi = tf.reduce_min(feat, axis=1, keepdims=True)
            ma = tf.reduce_max(feat, axis=1, keepdims=True)
            scaled_weights = tf.cast((feat - mi)/(ma - mi)*1000, dtype=tf.int32)
            n_nodes = tf.reduce_sum(tf.matrix_diag_part(adj), axis=1)
            nclusters = tf.cast(tf.floor(n_nodes / self.reduction_ratio), dtype=tf.int32)
            return louvain_module.louvain(bool_adj, scaled_weights, nclusters)

        def get_config(self):
            config = {
                'reduction_ratio': self.reduction_ratio,
                'data_format': self.data_format
            }
            base_config = super(ClusteringAdjacency, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    global GraphMaxPooling_
    class GraphMaxPooling_(layer_class):
        def __init__(self, data_format=None, **kwargs):
            self.data_format = data_format or K.image_data_format()
            super(GraphMaxPooling_, self).__init__(**kwargs)
        
        def build(self, input_shape):
            super(GraphMaxPooling_, self).build(input_shape)
        
        def compute_output_shape(self, input_shape):
            return input_shape[0]
        
        def call(self, batch):
            feat, ct = batch # gic, gin -> gnc
            bool_ct = tf.cast(ct, tf.bool)
            tiling = tf.stack([1, 1, 1, tf.shape(feat)[-1]], axis=0)
            ext_ct = tf.tile(tf.expand_dims(ct, -1), tiling) # ginc
            w = tf.where(ext_ct)
            values = tf.gather_nd(feat, tf.gather(w, [0, 1, 3], axis=1))
            
            masked = tf.SparseTensor(
                        indices=w,
                        values=values,
                        dense_shape=tf.cast(tf.shape(ext_ct), tf.int64)
                        )
            
            nb = tf.SparseTensor(
                        indices=w,
                        values=tf.ones_like(values, dtype=tf.float32),
                        dense_shape=tf.cast(tf.shape(ext_ct), tf.int64)
                        )
            
            return tf.reshape(tf.sparse_reduce_max(masked, axis=1), tf.shape(feat))

        def get_config(self):
            config = {
                'data_format': self.data_format
            }
            base_config = super(GraphMaxPooling_, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    global AdjacencyConversion
    class AdjacencyConversion(layer_class):
        def __init__(self, data_format=None, **kwargs):
            self.data_format = data_format or K.image_data_format()
            super(AdjacencyConversion, self).__init__(**kwargs)
        
        def build(self, input_shape):
            super(AdjacencyConversion, self).build(input_shape)
        
        def compute_output_shape(self, input_shape):
            return input_shape[1]
        
        def call(self, batch):
            adj, conversion_tensor = batch # gij, gni
            x = tf.einsum('gij,gjn->gin', adj, conversion_tensor)
            x = tf.einsum('gin,gim->gnm', conversion_tensor, x)
            return tf.clip_by_value(x, 0., 1.) # gnm

        def get_config(self):
            config = {
                'data_format': self.data_format
            }
            base_config = super(AdjacencyConversion, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    
    def multiply_or_none(a, b):
        return a*b if a is not None and b is not None else None

    global ImageToGraphFeat
    class ImageToGraphFeat(layer_class):
        def __init__(self, data_format=None, **kwargs):
            self.data_format = data_format or K.image_data_format()
            super(ImageToGraphFeat, self).__init__(**kwargs)
        
        def build(self, input_shape):
            super(ImageToGraphFeat, self).build(input_shape)
        
        def compute_output_shape(self, input_shape):
            input_shape = list(input_shape)
            if self.data_format == 'channels_first':
                output_shape = input_shape[:2] + [multiply_or_none(input_shape[2], input_shape[3])]
            else:
                output_shape = [input_shape[0], multiply_or_none(input_shape[1], input_shape[2]), input_shape[3]]
            return tuple(output_shape)
        
        def call(self, batch):
            input_shape = tf.shape(batch)
            if self.data_format == 'channels_first':
                output_shape = tf.stack([input_shape[0], input_shape[1], input_shape[2]*input_shape[3]], axis=0)
            else:
                output_shape = tf.stack([input_shape[0], input_shape[1]*input_shape[2], input_shape[3]], axis=0)
            return tf.reshape(batch, output_shape)

        def get_config(self):
            config = {
                'data_format': self.data_format
            }
            base_config = super(ImageToGraphFeat, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    global GraphConvNoEdges
    class GraphConvNoEdges(layer_class):
        def __init__(self,
                     kernel_size,
                     filters,
                     nhops=1,
                     data_format=None,
                     activation=None,
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None,
                     **kwargs):
            self.data_format = data_format or K.image_data_format()
            self.kernel_size = kernel_size
            self.nhops = nhops
            self.filters = filters
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.channel_axis = -1 if 'channels_last' else 1
            super(GraphConvNoEdges, self).__init__(**kwargs)

        def build(self, input_shape):
            input_shape_feat = input_shape[0]
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel', 
                                          shape=(self.filters, input_shape_feat[self.channel_axis], self.kernel_size),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(name='bias', 
                                            shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            super(GraphConvNoEdges, self).build(input_shape)  # Be sure to call this somewhere!

        def call(self, batch):
            # feat then adj
            batch_feat, batch_adj = batch
            with ops.name_scope('graphconv', [batch_feat, batch_adj]):
                mat_w = self.kernel
                batch_size = tf.shape(batch_adj)[0]
                sadj = tf.shape(batch_adj)[1]
                batch_adj_orig = batch_adj
                for i in range(self.nhops-1):
                    batch_adj = tf.einsum('gij,gjk->gik', batch_adj, batch_adj_orig)
                batch_bool_adj = tf.cast(batch_adj, dtype=tf.bool)
                sfilt = self.kernel_size
                # Create cost tensor
                cost_tensors = self._build_cost_tensors(batch_feat, mat_w)
                solver_cost_tensor = cost_tensors
                assignment_tensor = self._bipartite_matching(solver_cost_tensor, batch_bool_adj)
                assignment_tensor_ids = tf.where(assignment_tensor)
                gather_id = [0, 1, 3, 4] if self.data_format == 'channels_first' else [0, 2, 3, 4]
                masked_cost_tensor_values = tf.gather_nd(solver_cost_tensor, tf.gather(assignment_tensor_ids, gather_id, axis=1))
                #masked_cost_tensor_values = tf.where(tf.is_inf(masked_cost_tensor_values), tf.zeros_like(masked_cost_tensor_values), masked_cost_tensor_values)
                s = tf.stack([self.filters, sadj, sadj, sfilt] if self.data_format == 'channels_first'
                        else [sadj, self.filters, sadj, sfilt],
                        axis=0)
                s = tf.concat([tf.expand_dims(batch_size, 0), s], axis=0)
                masked_cost_tensor = tf.SparseTensor(
                        indices=assignment_tensor_ids,
                        values=masked_cost_tensor_values,
                        dense_shape=tf.cast(s, tf.int64)
                        )
                # Get signals
                ## for nodes (reduce sum for every masked cost matrices)
                nodes_signals = tf.reshape(tf.sparse_reduce_sum(masked_cost_tensor, [3, 4]),
                        ([batch_size, self.filters, -1] if self.data_format == 'channels_first' else [batch_size, -1, self.filters])
                    )
                ## for edges (reduce mean for every edges)
                # Add biases if activated
                if self.use_bias:
                    nodes_signals = tf.nn.bias_add(nodes_signals, self.bias, data_format=('NHWC' if self.data_format == 'channels_last' else 'NCHW'))
                
                # Create output signal 
                if self.activation is not None:
                    return self.activation(nodes_signals)
                return nodes_signals
                
        def compute_output_shape(self, input_shape):
            input_shape_feat = list(input_shape[0])
            input_shape_feat[self.channel_axis] = self.filters
            return tuple(input_shape_feat)
        
        def get_config(self):
            config = {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'nhops': self.nhops,
                'data_format': self.data_format,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)
            }
            base_config = super(GraphConvNoEdges, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def _get_assignment_mask_ids(self, assignment_tensor_ids, name=None):
            #assignment_tensor ids[g, f, k, i, a]<->[n, 5]
            with ops.name_scope(name, "get_assignment_mask_ids", [assignment_tensor_ids]):
                le, wi = tf.shape(assignment_tensor_ids)[0], tf.shape(assignment_tensor_ids)[1]
                size_head = wi-2
                t1 = tf.reshape(tf.tile(tf.expand_dims(assignment_tensor_ids, -1), [le, 1, 1]), [le*le, wi])
                t2 = tf.reshape(tf.tile(tf.expand_dims(assignment_tensor_ids[:, size_head:], 0), [1, 1, le]), [le*le, wi-size_head])
                return tf.concat([t1, t2], axis=-1) # ids[k, f, i, a, j, b]

        def _bipartite_matching(self, solver_subcost, adj, name=None):#filters_subcost, name=None):
            with ops.name_scope(name, "bipartite_matching", [solver_subcost, adj]):#, filters_subcost]):
                # subcost[f, i_sub, a]
                final_cost_matrix = self._build_lsap_costmatrix(solver_subcost)# add epsilons and transform to minimization
                nodes_solution = hungarian_module.hungarian(final_cost_matrix, adj)
                #### Return one-hot vectors
                return nodes_solution
        
        def _build_lsap_costmatrix(self, costmat, name=None):
            with ops.name_scope(name, "build_lsap_costmatrix", [costmat]):
                shape = tf.shape(costmat) # costmat[channel,i*a,i,a] or costmat[channel, i, a]
                positive_costmat = tf.add(costmat, -tf.reduce_min(costmat))
                negative_minimization_costmat = -positive_costmat - 1.
                positive_whole_costmat = tf.add(negative_minimization_costmat, -tf.reduce_min(negative_minimization_costmat))
                return positive_whole_costmat

        def _build_cost_tensors(self, inp, filters, name=None):
            # inp[batch, channel, i], filters[filter, channel, a] -> [graph, filter, i, a]
            with ops.name_scope(name, "build_cost_tensors", [inp]):
                if self.data_format == 'channels_first':
                    return tf.einsum('gci,fca->gfia', inp, filters)
                else:
                    return tf.einsum('gic,fca->gfia', inp, filters)    
            
            
    global GraphConv
    class GraphConv(layer_class):
        def __init__(self, kernel_size, filters, use_biases=True, data_format=None, **kwargs):
            self.data_format = data_format or K.image_data_format()
            self.kernel_size = kernel_size
            self.filters = filters
            self.use_biases = use_biases
            self.channel_axis = 1
            super(GraphConv, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel', 
                                          shape=(self.filters, input_shape[self.channel_axis]-1, self._get_vector_size(self.kernel_size)),
                                          initializer='uniform',
                                          trainable=True)
            if self.use_biases:
                self.biases = self.add_weight(name='biases', 
                                              shape=(self.filters, 2),
                                              initializer='uniform',
                                              trainable=True)
            super(GraphConv, self).build(input_shape)  # Be sure to call this somewhere!

        def call(self, batch_x):
            with ops.name_scope('graphconv', [batch_x]):
                # batch_x[batch, n, channel] or [batch, channel, n]
                # Get input and weights as matrices
                mat_w = self._vector_to_symmat(self.kernel)
                # batch_x[batch, channel, n]
                batch_mat_x = self._vector_to_symmat(batch_x)
                batch_size = batch_x.shape.as_list()[0]
                batch_adj = batch_mat_x[:,0] # [graph, i, j]
                batch_bool_adj = tf.cast(batch_adj, dtype=tf.bool)
                batch_feat = batch_mat_x[:, 1:] # [graph, channel-1, i, j]

                sadj = batch_adj.shape.as_list()[1]
                sfilt = self.kernel_size
                # Create cost tensor
                ## Create global cost tensor (without symmetric part and illogical coefficients) [batch, filter, i, a, j, b]
                cost_tensors = self._build_cost_tensors(batch_feat, mat_w)
                mask = self._whole_mask(cost_tensors, batch_bool_adj)
                solver_cost_tensor = tf.where(mask, cost_tensors, tf.fill(cost_tensors.shape, tf.constant(-float('inf'), dtype=float_type))) # [batch, filters, i, a, j, b]
                #filter_cost_tensor = tf.where(mask, cost_tensors, tf.fill(cost_tensors.shape, 0.)) # [batch, filters, i, a, j, b]
                assignment_tensor = self._bipartite_matching(solver_cost_tensor, batch_bool_adj)
                assignment_tensor_ids = tf.where(assignment_tensor)
                assignment_mask_ids = self._get_assignment_mask_ids(assignment_tensor_ids)
                assignment_mask = tf.SparseTensor(
                         indices=assignment_mask_ids,
                         values=tf.ones(tf.shape(assignment_mask_ids)[0], dtype=float_type),
                         dense_shape=[batch_size, self.filters, sadj, sadj, sfilt, sadj, sfilt]
                         )
                #assignment_mask = self._get_assignment_mask(assignment_tensor)
                #masked_cost_tensor = tf.einsum("gfkiajb,gfiajb->gfkiajb", assignment_mask, filter_cost_tensor)
                masked_cost_tensor_values = tf.gather_nd(solver_cost_tensor, tf.gather(assignment_mask_ids, [0, 1, 3, 4, 5, 6], axis=1))
                #masked_cost_tensor_values = tf.concat([tf.gather_nd(solver_cost_tensor, ami) for ami in assignment_masks_ids], axis=0)
                masked_cost_tensor_values = tf.where(tf.is_inf(masked_cost_tensor_values), tf.zeros_like(masked_cost_tensor_values), masked_cost_tensor_values)
                masked_cost_tensor = tf.SparseTensor(
                        indices=assignment_mask_ids,
                        values=masked_cost_tensor_values,
                        dense_shape=[batch_size, self.filters, sadj, sadj, sfilt, sadj, sfilt]
                        )
                # Get signals
                ## for nodes (reduce sum for every masked cost matrices)
                #nodes_signals = tf.stack([tf.stack(nodes_res) for nodes_res in assignment_tensors], axis=2)
                #edges_signals = tf.einsum('gfi,gfj->gfij', nodes_signals, nodes_signals)
                nodes_signals = tf.reshape(
                            tf.sparse_reduce_sum(masked_cost_tensor, [3, 4, 5, 6]),
                        [batch_size, self.filters, sadj]
                    ) # [graph, f, k, i, a, j, b] -> [graph, f, k]
                #nodes_signals = tf.reduce_sum(masked_cost_tensor, [3, 4, 5, 6]) # [g,f,k,i,a,j,b] -> [g,f,k]
                ## for edges (reduce mean for every edges)
                edges_signals = tf.reshape(tf.sparse_reduce_sum(masked_cost_tensor, [2, 4, 6]) / tf.maximum(tf.sparse_reduce_sum(assignment_mask, [2, 4, 6]), tf.constant(1., dtype=float_type)), [batch_size, self.filters, sadj, sadj])# [graph, f, k, i, a, j, b] -> [graph, f, i, j]
                #edges_signals = tf.reduce_sum(masked_cost_tensor, [2, 4, 6]) / tf.maximum(tf.reduce_sum(assignment_mask, [2, 4, 6]), 1.)
                # Add biases if activated
                if self.use_biases:
                    nodes_signals = tf.add(nodes_signals, tf.tile(tf.reshape(self.biases[:, 0], [1, self.filters, 1]), [batch_size, 1, sadj]), name="add_biases1")
                    edges_signals = tf.add(edges_signals, tf.tile(tf.reshape(self.biases[:, 1], [1, self.filters, 1, 1]), [batch_size, 1, sadj, sadj]), name="add_biases2")

                # Create output signal 
                ## Lower triangle : nodes signal
                ## Diagonal : edges signal
                mat_outp = tf.matrix_set_diag(edges_signals, nodes_signals) # [batch, channel, i, j]
                outp = self._symmat_to_vector(mat_outp) # [batch, channel, el]
                
                adj_slic = batch_x[:, 0] # fetch adjacencies
                adj_slic = tf.expand_dims(adj_slic, self.channel_axis)
                return tf.concat([adj_slic, outp], axis=self.channel_axis)
        
        def _get_am_mct(self, assignment_tensors, solver_cost_tensor, shape, name=None):
            with ops.name_scope(name, "get_masked_cost_tensor", [assignment_tensors, solver_cost_tensor, shape]):
                #assignment_tensor = tf.stack([tf.stack(node_res) for node_res in assignment_tensors], axis=2) # ([f,i,a]*g)*k -> [g,f,i,a]*k->[g,f,k,i,a]
                assignment_masks_ids = [[self._get_assignment_mask_ids(at) for at in l] for l in assignment_tensors]# [[ids[f, i, a, j, b][batch]][k]]
                #assignment_tensor_ids = [self._sparse_ids_stack(l) for l in assignment_tensors] #[ids[batch, f, i, a, j, b][k]]
                assignment_masks_ids = [self._sparse_ids_stack(l) for l in assignment_masks_ids] #[ids[batch, f, i, a, j, b][k]]
                #assignment_tensor_ids = self._sparse_ids_stack(assignment_tensor_ids, axis=2) #ids[g,f,k,i,a,j,b]
                assignment_mask_ids = self._sparse_ids_stack(assignment_masks_ids, axis=2) #ids[g,f,k,i,a,j,b]
                #bool_assignment_mask = self._get_assignment_mask_v2(assignment_tensor)
                #assignment_tensor_ids = tf.where(assignment_tensor)
                #assignment_mask_ids = tf.where(bool_assignment_mask)
                #assignment_mask_ids = self._get_assignment_mask_ids(assignment_tensor_ids)
                assignment_mask = tf.SparseTensor(
                         indices=assignment_mask_ids,
                         values=tf.ones(tf.shape(assignment_mask_ids)[0], dtype=float_type),
                         dense_shape=shape
                         )
                #assignment_mask = self._get_assignment_mask(assignment_tensor)
                #masked_cost_tensor = tf.einsum("gfkiajb,gfiajb->gfkiajb", assignment_mask, filter_cost_tensor)
                #masked_cost_tensor_values = tf.gather_nd(solver_cost_tensor, tf.gather(assignment_mask_ids, [0, 1, 3, 4, 5, 6], axis=1))
                masked_cost_tensor_values = tf.concat([tf.gather_nd(solver_cost_tensor, ami) for ami in assignment_masks_ids], axis=0)
                masked_cost_tensor_values = tf.where(tf.is_inf(masked_cost_tensor_values), tf.zeros_like(masked_cost_tensor_values), masked_cost_tensor_values)
                masked_cost_tensor = tf.SparseTensor(
                        indices=assignment_mask_ids,
                        values=masked_cost_tensor_values,
                        dense_shape=shape
                        )
                return assignment_mask, masked_cost_tensor
            
        def compute_output_shape(self, input_shape):
            #if self.data_format == 'channels_last':
            #    return input_shape[:-1] + (self.filters+1,)
            #if self.data_format == 'channels_first':
            return (input_shape[0], self.filters+1) + input_shape[2:]

        def _get_assignment_mask_ids(self, assignment_tensor_ids, name=None):
            #assignment_tensor ids[g, f, k, i, a]<->[n, 5]
            with ops.name_scope(name, "get_assignment_mask_ids", [assignment_tensor_ids]):
                le, wi = tf.shape(assignment_tensor_ids)[0], tf.shape(assignment_tensor_ids)[1]
                size_head = wi-2
                t1 = tf.reshape(tf.tile(tf.expand_dims(assignment_tensor_ids, -1), [le, 1, 1]), [le*le, wi])
                t2 = tf.reshape(tf.tile(tf.expand_dims(assignment_tensor_ids[:, size_head:], 0), [1, 1, le]), [le*le, wi-size_head])
                return tf.concat([t1, t2], axis=-1) # ids[k, f, i, a, j, b]

        def _get_assignment_mask(self, assignment_tensor, name=None):
            with ops.name_scope(name, "get_assignment_mask", [assignment_tensor]):
                #assignment_tensor[g, f, k, i, a]
                return tf.einsum('gfkia,gfkjb->gfkiajb', assignment_tensor, assignment_tensor)

        def _get_assignment_mask_v2(self, assignment_tensor, name=None):
            with ops.name_scope(name, "get_assignment_mask_v2", [assignment_tensor]):
                #assignment_tensor[g, f, k, i, a]
                s = tf.shape(assignment_tensor)
                head = s[:-2]
                tail = s[-2:]
                ones_head = tf.ones_like(head, tf.int32)
                ones_tail = tf.ones_like(tail, tf.int32)
                ones_s = tf.ones_like(s, dtype=tf.int32)
                t1 = tf.tile(tf.reshape(assignment_tensor, tf.concat([s, ones_tail], 0)), tf.concat([ones_s, tail], 0))
                t2 = tf.tile(tf.reshape(assignment_tensor, tf.concat([head, ones_tail, tail], 0)), tf.concat([ones_head, tail, ones_tail], 0))
                return tf.logical_and(t1, t2)

        def _bipartite_matching(self, solver_subcost, adj, name=None):#filters_subcost, name=None):
            with ops.name_scope(name, "bipartite_matching", [solver_subcost, adj]):#, filters_subcost]):
                # subcost[f, i_sub, a]
                sc_shape = solver_subcost.shape.as_list()
                I_size, F_size = sc_shape[-2], sc_shape[-1]
                flattened_costmatrix = tf.reshape(solver_subcost, sc_shape[:-4] + [I_size*F_size, I_size*F_size])
                node_costmatrix = tf.matrix_diag_part(flattened_costmatrix)
                node_costmatrix = tf.reshape(node_costmatrix, sc_shape[:-4] + [I_size, F_size])
                final_cost_matrix = self._build_lsap_costmatrix(node_costmatrix)# add epsilons and transform to minimization
                nodes_solution = hungarian_module.hungarian(final_cost_matrix, adj)
                #### Return one-hot vectors
                #ohot = tf.one_hot(nodes_solution, self.kernel_size, on_value=True, off_value=False) # [f, i_sub, a]
                return nodes_solution#ohot[..., :I_size, :] # remove epsilons
        
        
        def _build_lsap_costmatrix(self, costmat, name=None):
            with ops.name_scope(name, "build_lsap_costmatrix", [costmat]):
                shape = tf.shape(costmat) # costmat[channel,i*a,i,a] or costmat[channel, i, a]
                positive_costmat = tf.add(costmat, -tf.reduce_min(costmat))
                negative_minimization_costmat = -positive_costmat - 1.
                #rows_add, cols_add = shape[-1], shape[-2]
                #blcorner_part = tf.matrix_set_diag(
                #    tf.fill(tf.concat([shape[:-2], [rows_add, rows_add]], axis=0), tf.constant(float("inf"), dtype=float_type)),
                #    tf.zeros(tf.concat([shape[:-2], [rows_add]], axis=0), dtype=float_type)
                #)
                #trcorner_part = tf.matrix_set_diag(
                #    tf.fill(tf.concat([shape[:-2], [cols_add, cols_add]], axis=0), tf.constant(float("inf"), dtype=float_type)),
                #    tf.zeros(tf.concat([shape[:-2], [cols_add]], axis=0), dtype=float_type)
                #)
                #brcorner_part = tf.zeros(
                #    tf.concat([shape[:-2], [rows_add, cols_add]], axis=0),
                #    dtype=float_type
                #)                
                #negative_whole_costmat = tf.concat(
                #    [
                #        tf.concat([negative_minimization_costmat, trcorner_part], axis=-1),
                #        tf.concat([blcorner_part, brcorner_part], axis=-1)
                #    ],
                #    axis=-2
                #)
                positive_whole_costmat = tf.add(negative_minimization_costmat, -tf.reduce_min(negative_minimization_costmat))
                return positive_whole_costmat

        def _build_lsap_costmatrix_fbp(self, costmat, name=None):
            with ops.name_scope(name, "build_lsap_costmatrix_fbp", [costmat]):
                shape = tf.shape(costmat) # costmat[channel,i*a,i,a] or costmat[channel, i, a]
                positive_costmat = tf.add(costmat, -tf.reduce_min(costmat))
                negative_minimization_costmat = -positive_costmat - 1.
                max_dim = tf.maximum(shape[-1], shape[-2])
                rows_add, cols_add = tf.maximum(shape[-1]-shape[-2], 0), tf.maximum(shape[-2]-shape[-1], 0)
                trcorner_part = tf.fill(tf.concat([shape[:-2], [shape[-2], cols_add]], axis=0), tf.constant(0.0, dtype=float_type))
                blcorner_part = tf.fill(tf.concat([shape[:-2], [rows_add, shape[-1]+cols_add]], axis=0), tf.constant(0.0, dtype=float_type))
                negative_whole_costmat = tf.concat(
                    [
                        tf.concat([negative_minimization_costmat, trcorner_part], axis=-1),
                        blcorner_part
                    ],
                    axis=-2
                )
                positive_whole_costmat = tf.add(negative_whole_costmat, -tf.reduce_min(negative_whole_costmat))
                return positive_whole_costmat

        def _get_global_assignment(self, subvect, ids_I, shape, name=None):
            with ops.name_scope(name, "get_global_assignment", [subvect, ids_I]):
                ids_assign = tf.where(subvect) # ids[f, i_sub_assign, a] <-> [n, 3]
                real_ids_assign = tf.gather(ids_I, ids_assign[:, 1])
                coord = tf.concat([tf.expand_dims(ids_assign[:, 0], axis=-1), tf.expand_dims(real_ids_assign, axis=-1), tf.expand_dims(ids_assign[:, 2], axis=-1)], axis=1, name="concat_final")

                return tf.sparse_to_dense(coord, shape, tf.constant(True), default_value=tf.constant(False))

        def _get_global_assignment_ids(self, subvect, ids_I, name=None):
            with ops.name_scope(name, "get_global_assignment_ids", [subvect, ids_I]):
                ids_assign = tf.where(subvect) # ids[f, i_sub_assign, a] <-> [n, 3]
                real_ids_assign = tf.gather(ids_I, ids_assign[:, 1])
                coord = tf.concat([tf.expand_dims(ids_assign[:, 0], axis=-1), tf.expand_dims(real_ids_assign, axis=-1), tf.expand_dims(ids_assign[:, 2], axis=-1)], axis=1, name="concat_final")

                return coord
        def _sparse_ids_stack(self, el_list, axis=0, name=None):
            with ops.name_scope(name, "sparse_ids_stack", [el_list]):
                return tf.concat(
                    [
                        tf.concat([
                            el[...,:axis],
                            tf.fill((tf.shape(el)[0], 1), tf.constant(i, dtype=tf.int64)),
                            el[...,axis:]
                        ], axis=-1)
                        for i, el in enumerate(el_list)
                    ],
                    axis=0
                )
                
        def _logical_mask(self, t, name=None):
            with ops.name_scope(name, "logical_mask", [t]):
                shape = t.shape.as_list()
                return tf.constant([[[
                    [
                        [
                            [
                                (i == j) == (a == b)
                                for b in range(shape[5])    
                            ]
                            for j in range(shape[4])
                        ]
                        for a in range(shape[3])
                    ]
                    for i in range(shape[2]) 
                ]]*shape[1]]*shape[0])

                #return tf.equal(op1, op2)

        def _adjacent_mask(self, t, adj, name=None):
            with ops.name_scope(name, "adjacent_mask", [t, adj]):
                #adj[batch, i, j]
                #t[batch, filters, i, a, j, b]
                shape = t.shape.as_list()
                ngraphs, nfilters, sI, sA, _, _ = shape
                #op = tf.einsum('ij,fab->fiajb', adj, tf.ones([s0, s1, s1]))
                bool_adj = adj#tf.cast(adj, dtype=tf.bool)
                madj = tf.reshape(bool_adj, [ngraphs, 1, sI, 1, sI, 1])
                return tf.tile(madj, [1, nfilters, 1, sA, 1, sA])

        def _whole_mask(self, t, adj, name=None):
            with ops.name_scope(name, "whole_mask", [t, adj]):
                #return tf.logical_and(self._logical_mask(t), tf.logical_and(self._no_conjugate_mask(t), self._adjacent_mask(t, adj)))
                return self._adjacent_mask(t, adj)#tf.logical_and(self._logical_mask(t), self._adjacent_mask(t, adj))

        def _mask_to_tensor(self, mask, val, name=None):
            with ops.name_scope(name, "mask_to_tensor", [mask, val]):
                ids = tf.where(mask)
                return tf.sparse_to_dense(ids, mask.shape, val)

        def _build_cost_tensors(self, inp, filters, name=None):
            # inp[batch, channel-1, i, j], filters[filter, channel-1, a, b] -> [graph, filter, i, a, j, b]
            #if self.data_format == 'channels_first':
            with ops.name_scope(name, "build_cost_tensors", [inp]):
                return tf.einsum('gcij,fcab->gfiajb', inp, filters)
            #else:
            #    return tf.einsum('ijf,abf->aibj', feat1, feat2)

        def _vector_to_symmat(self, v, name=None):
            with ops.name_scope(name, "vector_to_symmat", [v]):
                # v[batch, channel, el] -> [batch, channel, i, j]
                low = fill_triangular(v)
                transp = tf.transpose(low, perm=[0, 1, 3, 2])
                up = tf.matrix_set_diag(transp, tf.zeros(low.shape[:-1], dtype=float_type))
                return low + up # [batch, channel, i, j]

        def _symmat_to_vector(self, mat, name=None):
            with ops.name_scope(name, "symmat_to_vector", [mat]):
                # mat[batch, channel, i, j] -> [batch, channel, el]
                n = mat.shape.as_list()[-1]
                m = self._get_vector_size(n)
                indmat = fill_triangular(tf.range(1, m+1))
                ids = tf.where(tf.not_equal(indmat, 0))
                values = tf.gather_nd(indmat, ids, name="gather_indices")
                reordered = tf.gather(ids, tf.nn.top_k(values, k=m).indices)
                l1 = list()
                for s1 in tf.split(mat, mat.shape[0]):
                    s1 = tf.squeeze(s1, [0])
                    l2 = list()
                    for s2 in tf.split(s1, mat.shape[1]):
                        s2 = tf.squeeze(s2, [0])
                        vect = tf.reverse(tf.gather_nd(s2, reordered, name="gather_to_vect"),[0])
                        l2.append(vect)
                    l1.append(tf.stack(l2))
                return tf.stack(l1)

        def _get_vector_size(self, n):
                return n*(n+1)//2
