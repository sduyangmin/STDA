from sklearn.metrics import mean_squared_error
import numpy as np

def get_MSE(pred, real):
    return mean_squared_error(real.flatten(), pred.flatten())

def print_metrics(pred, real):
    mse = get_MSE(pred, real)

    print('Test: MSE={:.6f}'.format(mse))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    

def get_MAPE(pred, real):
    mapes = []
    for i in range(len(pred)):
        gt_sum = np.sum(np.abs(real[i]))
        er_sum = np.sum(np.abs(real[i] - pred[i]))
        mapes.append(er_sum / gt_sum)
    return np.mean(mapes)