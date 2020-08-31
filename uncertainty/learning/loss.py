import os, sys

import tensorflow as tf

def loss_reduce(loss_vec, reduce):
    if reduce == 'mean':
        loss = tf.math.reduce_mean(loss_vec)
    elif reduce == 'sum':
        loss = tf.math.reduce_sum(loss_vec)
    elif reduce == 'none':
        loss = loss_vec
    else:
        raise NotImplementedError

    return loss


def loss_xe(x, y, model, model_iw=None, reduce='mean', tape=None):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    ##TODO: simplify
    if model_iw is not None:
        if tape is not None:
            with tape.stop_recording():
                iw = model_iw(x)
        else:
            iw = model_iw(x)
        iw = iw + 0.5
    else:
        iw = None

    logits = model(x)['logits']    
    loss_vec = scce(y, logits, sample_weight=iw) ## heuristic approach for applying importance weight.
    loss_red = loss_reduce(loss_vec, reduce)
    return {'loss': loss_red}


def loss_self(x, y, model, model_iw=None, reduce='mean', tape=None):
    """This function is currently not used. Will be the part of codebase after testing."""
    assert(len(x) == 2 and len(y) == 2)
    loss_L = loss_xe(x[0], y[0], model, model_iw, reduce=reduce, tape=tape)
    if x[1] is not None:
        loss_U = loss_xe(x[1], y[1], model, model_iw, reduce=reduce, tape=tape)
    else:
        loss_U = {'loss': tf.zeros((1))} ##TODO: clean-way?
        
    if reduce == 'mean':
        loss_ret = {'loss': 0.1*loss_L['loss'] + loss_U['loss'], 'loss_L': loss_L['loss'], 'loss_U': loss_U['loss']}
    elif reduce == 'none':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return loss_ret


def loss_bxe(x, y, model, reduce='mean'):
    bxe = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    probs = model(x)['probs']
    y = tf.one_hot(y, 2)
    loss_vec = bxe(y, tf.concat((1.0-probs, probs), 1))
    loss_red = loss_reduce(loss_vec, reduce)
    return {'loss': loss_red}


def loss_01(x, y, model, model_iw=None, reduce='mean'):
    logits = model(x)['logits']
    loss_vec = tf.cast(tf.argmax(logits, axis=1) != tf.cast(y, tf.int64), tf.float32)
    loss_red = loss_reduce(loss_vec, reduce)
    return {'loss': loss_red}


def loss_adv(x, y, model, reduce='mean'):
    return loss_bxe(x, y, lambda x: {'probs': model(x)['domain']}, reduce)


def loss_xe_adv(x, y, model, model_iw=None, reduce='mean', tape=None, reg_param_adv=0.0):
    x, xd = x
    y, yd = y
    l_xe = loss_xe(x, y, model, model_iw, reduce, tape)['loss']
    l_adv = loss_adv(xd, yd, model, reduce)['loss']
    l = l_xe + reg_param_adv*l_adv
    return {'loss': l, 'loss_xe': l_xe, 'loss_adv': l_adv}
    

def loss_entropy(x, model, reduce='mean'):
    logits = model(x)['logits']    
    probs = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss_vec = - tf.math.reduce_sum(probs * log_probs, axis=-1)
    loss_red = loss_reduce(loss_vec, reduce)
    return {'loss': loss_red}
    

def precision(x, y, model, model_conf, model_iw=None, reduce='mean'):

    logits = model(x)
    y_pred = tf.math.argmax(logits, axis=-1)
    conf = model_conf(x) == 1
    n_conf = tf.math.reduce_sum(tf.cast(conf, tf.int64))
    
    if model_iw is not None:
        # weighted precision
        w_pred = model_iw(x)
    else:
        w_pred = tf.cast(tf.ones_like(conf), tf.float32)
        
    if reduce == 'mean':
        if n_conf == 0:
            return tf.constant(1.0), n_conf
        else:
            corr = tf.cast(tf.cast(y, tf.int64) == tf.cast(y_pred, tf.int64), tf.float32)
            prec = corr[conf]*w_pred[conf]
            prec = tf.math.reduce_mean(prec)
            return prec, n_conf
    else:
        if n_conf == 0:
            return tf.zeros((0,)), n_conf
        else:
            corr = tf.cast(tf.cast(y, tf.int64) == tf.cast(y_pred, tf.int64), tf.float32)
            prec = corr[conf]*w_pred[conf]
            return prec, n_conf
 


def reg_l2(weights):
    l = tf.add_n([tf.nn.l2_loss(w) for w in weights])
    return l

