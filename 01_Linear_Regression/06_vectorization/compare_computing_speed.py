import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import copy
import time

def get_cwd():
    cwd = os.getcwd()
    if not '06_vectorization' in cwd:
        cwd += '\\01_Linear_Regression\\06_vectorization'
    return cwd
def compute_cost(x_, y_, w1_, w2_, w3_, w4_, w5_, b_):
    x1 = x_[:,0]
    x2 = x_[:,1]
    x3 = x_[:,2]
    x4 = x_[:,3]
    x5 = x_[:,4]

    m = x1.shape[0]

    cost_sum = 0
    for i in range(m):
        f_x = w1_*x1[i] + w2_*x2[i] + w3_*x3[i] + w4_*x4[i] + w5_*x5[i] + b_
        cost = (f_x - y_[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2.0 * m)) * cost_sum
    return total_cost[0]

def compute_cost_vect(x_, y_, w_, b_):
    m = x_.shape[0]

    f_x = np.dot(x_, w_) + b_
    cost = (f_x - y_) ** 2
    cost_sum = np.sum(cost)
    total_cost = (1 / (2.0 * m)) * cost_sum
    return total_cost

def compute_grad(x_, y_, w1_, w2_, w3_, w4_, w5_, b_):

    x1 = x_[:,0]
    x2 = x_[:,1]
    x3 = x_[:,2]
    x4 = x_[:,3]
    x5 = x_[:,4]

    m = x1.shape[0]

    dJ_dw_1_sum = 0
    dJ_dw_2_sum = 0
    dJ_dw_3_sum = 0
    dJ_dw_4_sum = 0
    dJ_dw_5_sum = 0
    dJ_db_sum = 0

    for i in range(m):
        f_x = w1_*x1[i] + w2_*x2[i] + w3_*x3[i] + w4_*x4[i] + w5_*x5[i] + b_
        dJ_dw_1_i = (f_x - y_[i]) * x1[i]
        dJ_dw_2_i = (f_x - y_[i]) * x2[i]
        dJ_dw_3_i = (f_x - y_[i]) * x3[i]
        dJ_dw_4_i = (f_x - y_[i]) * x4[i]
        dJ_dw_5_i = (f_x - y_[i]) * x5[i]
        dJ_db_i = (f_x - y_[i])

        dJ_dw_1_sum += dJ_dw_1_i
        dJ_dw_2_sum += dJ_dw_2_i
        dJ_dw_3_sum += dJ_dw_3_i
        dJ_dw_4_sum += dJ_dw_4_i
        dJ_dw_5_sum += dJ_dw_5_i
        dJ_db_sum += dJ_db_i
    dJ_dw_1_sum = dJ_dw_1_sum / m
    dJ_dw_2_sum = dJ_dw_2_sum / m
    dJ_dw_3_sum = dJ_dw_3_sum / m
    dJ_dw_4_sum = dJ_dw_4_sum / m
    dJ_dw_5_sum = dJ_dw_5_sum / m
    dJ_db_sum = dJ_db_sum / m

    return (dJ_dw_1_sum[0], dJ_dw_2_sum[0], dJ_dw_3_sum[0], dJ_dw_4_sum[0], dJ_dw_5_sum[0], dJ_db_sum[0])

def compute_grad_vect(x_, y_, w_, b_):
    m = x_.shape[0]

    f_wb = np.dot(x_, w_) + b_
    cost = f_wb - y_
    dJ_dw = np.dot(x_.T, cost) * (1 / m)
    dJ_db = np.sum(cost) * (1 / m)

    return (dJ_dw, dJ_db)

def train_non_vect_model(x_, y_, w1_init_, w2_init_, w3_init_, w4_init_, w5_init_, b_init_, iterations_, alpha_):

    w1 = copy.deepcopy(w1_init_)
    w2 = copy.deepcopy(w2_init_)
    w3 = copy.deepcopy(w3_init_)
    w4 = copy.deepcopy(w4_init_)
    w5 = copy.deepcopy(w5_init_)
    b = copy.deepcopy(b_init_)

    cost_history = []
    model_history = []

    process_start_time = time.time()
    for i in range(iterations_):
        dJ_dw1, dJ_dw2, dJ_dw3, dJ_dw4, dJ_dw5, dJ_db = compute_grad(x_, y_, w1, w2, w3, w4, w5, b)
        w1 = w1 - dJ_dw1 * alpha_
        w2 = w2 - dJ_dw2 * alpha_
        w3 = w3 - dJ_dw3 * alpha_
        w4 = w4 - dJ_dw4 * alpha_
        w5 = w5 - dJ_dw5 * alpha_
        b = b - dJ_db * alpha_
        cost_history.append(compute_cost(x_, y_, w1, w2, w3, w4, w5, b))
        model_history.append([w1, w2, w3, w4, w5, b])
    process_end_time = time.time()

    print(f"Non-Vectorized Proces takes: {(process_end_time - process_start_time):.2f} seconds. Lowest cost: {cost_history[-1]:.4f}")
    print("Non-Vectorized Best model equation: f = {:.4f}.x1 + {:.4f}.x2 + {:.4f}.x3 + {:.4f}.x4 + {:.4f}.x5 + {:.4f}".format(
        model_history[-1][0],
              model_history[-1][1],
              model_history[-1][2],
              model_history[-1][3],
              model_history[-1][4],
              model_history[-1][5]))

def train_vect_model(x_, y_, w1_init_, w2_init_, w3_init_, w4_init_, w5_init_, b_init_, iterations_, alpha_):
    # Vectorized training:
    w_init = np.array([[w1_init_], [w2_init_], [w3_init_], [w4_init_], [w5_init_]])

    w = copy.deepcopy(w_init)
    b = copy.deepcopy(b_init_)


    cost_history = []
    model_history = []

    process_start_time = time.time()
    for i in range(iterations_):
        dJ_dw, dJ_db = compute_grad_vect(x_, y_, w, b)
        w = w - np.dot(dJ_dw, alpha_)
        b = b - np.dot(dJ_db, alpha_)
        cost_history.append(compute_cost_vect(x_, y_, w, b))
        model_history.append([w, b])

    process_end_time = time.time()
    print(f"Vectorized Proces takes: {(process_end_time - process_start_time):.2f} seconds. Lowest cost: {cost_history[-1]:.4f}")
    print("Vectorized Best model equation: f = {:.4f}.x1 + {:.4f}.x2 + {:.4f}.x3 + {:.4f}.x4 + {:.4f}.x5 + {:.4f}".format(
        model_history[-1][0][0][0],
              model_history[-1][0][1][0],
              model_history[-1][0][2][0],
              model_history[-1][0][3][0],
              model_history[-1][0][4][0],
              model_history[-1][1]))


if __name__ == "__main__":

    cwd = get_cwd()
    data = pd.read_csv(f'{cwd}\\datasets\\Housing.csv')
    not_number_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    data.drop(columns=not_number_columns, axis=1, inplace=True)

    x_df = data.drop(['price'],axis=1)
    y_df = data['price']

    # Normalisation = Scaling:
    scaler = StandardScaler()
    scaler.fit(x_df)
    x = scaler.transform(x_df) # numpy ndarray scaled
    y = np.array(y_df).reshape(-1, 1)

    w1 = 81
    w2 = 10
    w3 = 3
    w4 = 4
    w5 = 2
    b = 30
    iterations = 1000
    alpha = 3.0e-3

    train_non_vect_model(x, y, w1, w2, w3, w4, w5, b, iterations, alpha)
    train_vect_model(x, y, w1, w2, w3, w4, w5, b, iterations, alpha)


