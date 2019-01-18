# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:30:46 2018

@author: greenteea
"""
import numpy as np

#sigmoid函数前向传播部分
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

#sigmoid函数反向传播部分
def sigmoid_backward(dA, cache):
    '''
    dA:反向传播前一层的梯度值
    cache:反向传播包含前一层变量的字典
    dZ:输出的梯度值
    '''
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert(dZ.shape == Z.shape)
    return dZ

#ReLU函数前向传播部分
def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape==Z.shape)
    cache = Z
    return A, cache
 
#ReLU函数反向传播部分
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

#生成初始权重值
def weight_variable(shape):
    mean, std = 0, 0.1
    initial = np.random.normal(mean,std,shape)
    return initial

#生成初始偏置值
def bias_variable(shape):
    initial = np.ones(shape) * 0.1
    return initial

#初始化所有神经元参数
def initialize_parameters_deep(layer_dims):
    '''
    layer_dims:包含神经网络中神经元的层数及个数的列表  
    parameters:包含神经网络参数的字典
    WL: L层的权重值(layer_dims[l],layer_dims[l-1]）
    bL: l层的偏置向量(layer_dims[l],1)
    '''
    np.random.rand()
    parameters = {}
    L = len(layer_dims) 

    for l in range(1,L):
        parameters['W' + str(l)] = weight_variable((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = bias_variable((layer_dims[l],1))    # assert (y_training.shape[0]==1)
        assert(parameters['W' + str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l],1))

    return parameters

#线性函数的前向传播部分
def linear_forward(A,W,b):
    '''
    A:上一层的输出或来自输入层(size of previous layer, number of examples)
    W:权重向量: 数组，格式(size of current layer, size of previous layer)
    b:偏置向量, 格式(size of current layer,1)
    Z:激活函数的输入
    cache:字典，包含 'A','W','b'
    '''
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A,W,b)
    
    return Z,cache

#激活函数的前向传播部分
def linear_activation_forward(A_prev, W, b, activation):
    '''
    A_prev: 上一层的激活函数输出:(size of previous layer, number of examples)
    W: 权重向量 (size of current layer, size of previous layer)
    b: 偏置向量(size of current layer, 1)
    activation: 该层使用的激活函数，字符串:'sigmoid' or 'relu'
    A: 该层激活函数的输出
    cache: 记录中间层输出的字典
    '''
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache

#模型前向传播 , 加入了残差项
def L_model_forward(X, parameters):
    '''
    X：神经网络的输入
    parameters:存有每一层参数的字典，格式(权重向量、偏置向量)
    caches: 存有每一层的输出的记录字典，格式(线性输出，激活函数输出)
    AL: 神经网络的预测值
    '''
    caches = []
    A = X
    #X_identity表示残差网络中的identity项
    X_identity = X
    #对X_identity进行降采样，使得X_res的维度与输出层一致
    identity_index = np.random.choice(a=X_identity.shape[0], size=1)
    X_identity = X_identity[identity_index , :]
    
    L = len(parameters) // 2 

    #使用relu作为激活函数的神经元，在caches末尾添加了上一层的输出cache
    for l in range(1,L):
        #如果是前L层，使用全连接 + ReLU层
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    #最后一层，将全连接 和 ReLU分开，方便加入identity项
    A_prev = A
    A, cache_last_linear = linear_forward(A_prev, parameters['W'+str(L)], parameters['b'+str(L)]) 
    
    #加入残差项
    A = A + X_identity
    
    #对最后一层进行ReLU操作
    A, cache_last_relu = relu(A)
    cache = (cache_last_linear , cache_last_relu)
    caches.append(cache)

    # 传播至最后一层sigmoid激活函数
    AL , cache = sigmoid(A)
    caches.append(cache)
#    print(AL.shape)
    assert(AL.shape == (1,X.shape[1]))
    
    return AL, caches

#计算代价函数
def compute_cost(AL, Y):
    '''
    AL: 神经网络的预测值, 格式(1, number of examples)
    Y: 实际值,格式(1,number of examples)
    cost: 代价函数
    '''
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)
    assert(cost.shape==())
    
    return cost

#反向传播的线性部分
def linear_backward(dZ, cache):
    '''
    dZ:当前层的代价梯度
    cache:元组，存有前一层神经元的输入，该层神经元的权重，偏置
    dA_prev:前一层激活函数输出的代价梯度
    dW:该层权重下的代价梯度
    db:该层偏置下的代价梯度
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]
    # assert (y_training.shape[0]==1)
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

#经过激活函数后反向传播的数值
def linear_activation_backward(dA, cache, activation):
    '''
    dA:当前层的前向传播激活函数的梯度
    cache:激活函数的输入（A,W,b)
    activation:激活函数的类型
    dA_prev:前一层激活函数输出的代价梯度
    dW:该层权重下的代价梯度
    db:该层偏置下的代价梯度
    '''
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

#模型反向传播
def L_model_backward(AL, Y, caches):
    '''
    AL:神经网络的预测值
    Y:真实值
    caches:存有中间层的参数
    grads:梯度字典
    '''
    grads = {}
    L = len(caches) # 神经元的层数
    Y = Y.reshape(AL.shape) # 使预测值和真实值维度一致
    # 输出层反向传播梯度计算
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    #最后一层中sigmoid激活函数到线性函数的梯度计算
    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = sigmoid_backward(dAL, current_cache) , 1 , 1

    for l in reversed(range(L-1)):
        # 输出层之前的反向传播
        # 输入: "grads['dA'+str(l+2)],caches" 输出: "grads['dA'+str(l+1)],grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

#更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters

#神经网络的主函数
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iteration = 3000, print_cost = False, SGD=False):
    '''
    X: 输入值
    Y: 输入值对应的真实标签
    layers_dims: 神经网络神经元的参数，格式[size1 , size2 , size3 ,...]
    learning_rate: 学习率
    num_iteration: 迭代次数
    SGD: 随机梯度下降标识符
    parameters: 存有每一层参数的字典，格式(权重向量、偏置向量)
    '''
    np.random.seed(1)
    costs = []  # 代价
    # 参数初始化
    parameters = initialize_parameters_deep(layers_dims)

    #循环，计算参数
    for i in range(1, num_iteration + 1):
        #随机梯度下降
        if SGD == 'True':
            #通过随机从输入参数中不放回取500个参数作为神经元的输入，减小计算量
            index = np.random.choice(a=X.shape[0], size=500, replace=False, p=None)
            X = X[index,:]
            Y = Y[index,:]

        # 前向传播
        AL, caches = L_model_forward(X, parameters)
        # 代价计算
        cost = compute_cost(AL, Y)
        # 反向传播
        grads = L_model_backward(AL, Y, caches)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每100次，进行代价输出
        if i % 100 == 0:
            print('轮数: {}/{} '.format(i, num_iteration),
                  '训练误差: {}'.format(cost))
            costs.append(cost)
            
    return parameters

#生成随机的训练集和测试集
def gener_dataset(training_shape,test_shape):
    """
    training_shape: 训练集的shape，shape[0] = 样本数 , shape[1] = 特征数
    test_shape: 测试集的shape
    """
    
    #生成训练集
    np.random.rand()
    x_training = np.random.rand(training_shape[0], training_shape[1])
    y_training = np.hstack((np.ones((1, training_shape[0]//2)), np.zeros((1, training_shape[0]-training_shape[0]//2))))
    assert (y_training.shape[0]==1)
    np.random.shuffle(y_training)
    
    #生成测试集
    x_test = np.random.rand(test_shape[0], test_shape[1])
    y_test = np.hstack((np.ones((1, test_shape[0]//2)), np.zeros((1, test_shape[0]-test_shape[0]//2))))
    
    #随机排列y_test
    np.random.shuffle(y_test)
    
    return x_training, y_training, x_test, y_test

#预测函数
def predict(x, y, parameters):

    m = x.shape[1]
    p = np.zeros((1, m))

    # 前向传播
    probas, caches = L_model_forward(x, parameters)

    # 预测输出值
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Test Accuracy: {}".format(str(np.sum((p == y) / float(m)))))

    return p

if __name__ == '__main__':
    '''
    x_training.shape() = (10000 , 50)
    y_training.shape() = (1 , 10000)
    x_test.shape() = (300 , 50)
    y_test.shape() = (1 , 300)
    其中x参数输入后会进行一次转置操作,即x = x.T
    '''
    x_training, y_training, x_test, y_test = gener_dataset([10000,50],[300,50])
    x_training = x_training.T
    x_test = x_test.T
    
    #神经网络层数 = len(layers_dims) - 1 
    layers_dims = [50,20,1]   
    parameters = L_layer_model(x_training, y_training, layers_dims, num_iteration=2500, print_cost=True, SGD=True)
    pred_test = predict(x_test, y_test, parameters)