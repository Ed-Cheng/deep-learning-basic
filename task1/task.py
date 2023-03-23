import tensorflow as tf
import time 

def polynomial_fun(w, x):
    """
    A simple polynomial function

    Args
        w: M+1 vector, weight vector
        x: scalar, variable
    Return
        y: scalar, result of a polynomial function
    """
    y = 0
    for count, m in enumerate(w):
        y += m * x**count
    
    return y

def fit_polynomial_ls(x, t, M):
    """
    Least square solver that solves the fitting polynomial function
    minimizes the least square error between data

    Args
        x: N tesorflow data, the x value
        t: N tesorflow data, the corresponding y values of x value
        M: scalar, the polynomial degree
    Return
        w: n tensor numpy, the optimal weights 
    """
    M += 1

    store_x1 = []
    for i in range(M):
      for j in range(M):
        power_x = x.numpy()**(i+j)
        sum_x = sum(power_x)
        store_x1.append(sum_x)

    x1 = tf.reshape(store_x1, [M, M])

    store_x2 = []
    for dim in range(M):
        power_x = x.numpy()**dim * t.numpy()
        sum_x = sum(power_x)
        store_x2.append(sum_x)

    x2 = tf.reshape(store_x2, [M, 1])
    w = tf.linalg.matmul(tf.linalg.inv(x1), x2)
    w = w.numpy()[:,0]

    return w

def get_y(w, x, noise):
    """
    Returns the y values of polynomial_fun of x, noise can be added if wanted.

    Args
        w: (M+1) tensorflow data, the weight 
        x: n tesorflow data, the x value
        noise: boolin, if true add noise

    Return
        y: tensor, the y values of the polynomial function in polynomial_fun 
    """
    nData = x.shape[0]
    x = x.numpy()
    y = []
    for i in range(nData):
        y.append(polynomial_fun(w, x[i]))

    if noise == True:
        y = y + tf.random.normal([nData], 0.0, 2.0)
    else:
        y = y + tf.zeros(nData)

    return y

def cal_error(eval_y, x, w):
    """
    calculate the error mean and the std of a set of data

    Args
        eval_y: tensor, the y values waiting to be commpared with the true-y values of x (without noise)  
        x: n tesorflow data, the x value 
        w: (M+1) tensorflow data, the weight 
    Return
        mean: the mean of the difference between true_y and eval_y
        std: the std of the difference between true_y and eval_y
    """
    true_poly = get_y(w, x, False)

    mean = tf.reduce_mean(tf.abs(eval_y - true_poly))
    std = tf.math.reduce_std(tf.abs(eval_y - true_poly))

    return mean, std


def fit_polynomial_sgd(x, t, M, lr_rate, minibatch):
    """
    SGD solver that solves the fitting polynomial function

    Args
        x: n*1 tensorflow data, containing x value
        t: n*1 tensorflow data, containing correspondong y value of x
        M: scalar, the (polynomial-1) degree
        lr_rate: scalar, the learning rate
        minibatch: scalar, the size of minibatch. Will be optimized if needed.
    Return
        best_SDG_w: 1*n tensorflow data, the optimal weight calculated by sgd.
    """
    x_shape = x.shape[0]

    # check if the minibatch can fully divide the total dataset, tune it if need
    increment = 0
    while x_shape % minibatch != 0:
        increment += 1
        if increment % 2:
            minibatch += increment
        else:
            minibatch -= increment
    print(f"\nminibatch changed to {minibatch} to suite the size of data\n")

    # extend x dimension in polynomial degree for easier computing later
    extend_x = tf.zeros([x_shape , 1])
    for i in range(M+1, 0, -1):
        extend_x = tf.concat([extend_x, x**i], 1)
    extend_x = extend_x[:,1:(M+1)+1]

    batch_num = x_shape/minibatch
    split_x = tf.split(extend_x , minibatch)
    split_y = tf.split(t , minibatch)

    batch_num = tf.cast(batch_num, dtype=tf.int32)
    split_x = tf.cast(split_x, dtype=tf.float64)
    split_y = tf.cast(split_y, dtype=tf.float64)    

    # Initialize random weights 
    SDG_weight = tf.Variable([[int(tf.random.uniform([1], -2.0, 5.0)) for k in range(M+1)]], dtype=tf.float64)


    opt = tf.keras.optimizers.Adam(lr_rate)

    print("start epoch")
    MSE_loss = 600
    storeMSE = 10000
    epoch = 1
    bestep = 0
    while MSE_loss > 500 and epoch < 20000:
        for i in range(batch_num):
            with tf.GradientTape() as tape:
                predict_y = tf.matmul(split_x[i], SDG_weight, transpose_b=True)
                MSE_loss = tf.math.reduce_mean((split_y[i] - predict_y)**2)    
            gradients = tape.gradient(MSE_loss, SDG_weight)
            opt.apply_gradients(zip([gradients], [SDG_weight]))


        if epoch % 200 == 0:
            print(f"epoch: {epoch}, MSE_loss: {tf.round(MSE_loss*100)/100}")
        if MSE_loss < storeMSE:
            storeMSE = MSE_loss
            best_SDG_w = SDG_weight
            bestep = epoch

        epoch += 1

    print(f"In epoch: {bestep} found MSE_loss: {tf.round(storeMSE*100)/100}")

    return best_SDG_w

if __name__ == '__main__':

    trainX, testX = tf.random.uniform([100], -20.0, 20.0), tf.random.uniform([50], -20.0, 20.0)

    w = tf.transpose(tf.constant([1,2,3,4]))

    # get_y is using polynomial_fun
    trainy, testy = get_y(w, trainX, True), get_y(w, testX, True)

    LS_start_time = time.time()
    optimal_w = fit_polynomial_ls(trainX, trainy, M = 4)
    LS_end_time = time.time()
    LS_time = LS_end_time - LS_start_time


    pred_trainy, pred_testy = get_y(optimal_w, trainX, False), get_y(optimal_w, testX, False)

    # difference a) between the observed training data and the underlying “true” polynomial curve
    trainy_err_mean, trainy_err_std = cal_error(trainy, trainX, w)

    # difference b) between the “LS-predicted” values and the underlying “true” polynomial curve.
    LSpredy_err_mean, LSpredy_err_std = cal_error(pred_trainy, trainX, w)

    print(f"Difference between 'observed training data' and the 'true polynomial': mean {trainy_err_mean} std {trainy_err_std}")
    print(f"Difference between 'LS-predicted data' and the 'true polynomial'     : mean {LSpredy_err_mean} std {LSpredy_err_std}")


    # learning rate between 0.001 and 0.009 were tested 
    # found 0.004, 0.005, 0.006 were good
    # third test 0.0045 performed the best between [0.004, 0.0045, 0.005, 0.0055, 0.006] 
    reshape_trainX, reshape_trainy = tf.reshape(trainX, (trainX.shape[0], 1)), tf.reshape(trainy, (trainy.shape[0], 1))
    SGD_start_time = time.time()
    SGD_w = fit_polynomial_sgd(reshape_trainX, reshape_trainy, 4, 0.0045, 21)
    SGD_end_time = time.time()
    SGD_time = SGD_end_time - SGD_start_time

    # predict target y for train and test set using SGD 
    SGD_trainy, SGD_testy = get_y(tf.transpose(SGD_w), trainX, False), get_y(tf.transpose(SGD_w), testX, False)

    # difference between the SGD-predicted values and the underlying true polynomial curve
    SGDpredy_err_mean, SGDpredy_err_std = cal_error(SGD_testy, testX, w)
    print(f"Difference between 'SGD-predicted' and the 'true polynomial': mean {SGDpredy_err_mean} std {SGDpredy_err_std}")


    # Compare the two methods' RMSEs
    RMSE_loss_LS = tf.math.sqrt(tf.math.reduce_mean((pred_testy - testy)**2))
    RMSE_loss_SGD = tf.math.sqrt(tf.math.reduce_mean((SGD_testy - testy)**2))
    print(f"RMSE of LS method: {RMSE_loss_LS}\n",
    f"RMSE of SGD method: {RMSE_loss_SGD}\n")

    # comparing the speed
    print(f"Time spent for LS (to get minimum MSE):  {LS_time} seconds\n", 
    f"Time spent for SGD (to get train_MSE<500 or reached max epoch 20k):  {SGD_time} seconds")
