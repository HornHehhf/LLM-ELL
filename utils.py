import torch
import numpy as np
import random
import pickle
import copy
import math
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.switch_backend('agg')


def set_random_seed(seed):
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_data_with_pickle(data, data_path):
    pickle_data = open(data_path, "wb")
    pickle.dump(data, pickle_data)
    pickle_data.close()


def load_data_from_pickle(data_path):
    pickle_data = open(data_path, 'rb')
    data = pickle.load(pickle_data)
    pickle_data.close()
    return data


def linear_regression(X, Y):
    X = copy.deepcopy(X).reshape(-1, 1)
    Y = copy.deepcopy(Y)
    Y = np.log10(Y)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    rsquare = r2_score(Y, y_pred)
    print('Coefficients: \n', regr.coef_, regr.intercept_)
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % rsquare)
    return X, Y, y_pred, rsquare


def round_to_first_nonzero(num):
    # Convert to string to identify the first non-zero digit
    num_str = '{:.10g}'.format(num)

    # Find the first non-zero digit
    for i, char in enumerate(num_str):
        if char not in ['0', '.', '-']:
            first_nonzero_index = i
            break

    # Count significant digits to keep
    significant_digits = first_nonzero_index + 1

    # Round the number to the calculated significant digits
    rounded_num = round(num, significant_digits - len(num_str.split('.')[0]) - 1)

    return rounded_num


def analyze_terminal_features(loss_list, figure_path, ylim_range=None, xlim_range=None):
    loss_list = np.array(loss_list)
    X = np.arange(len(loss_list))
    Y = loss_list
    print('overall ratio:', Y[-1]/Y[0])
    if ylim_range is None:
        y_min, y_max = min(Y), max(Y)
    else:
        y_min, y_max = ylim_range
    print(y_min, y_max)
    X, Y, y_pred, _ = linear_regression(X, Y)
    print('pearson correlation', stats.pearsonr(X.reshape(-1), Y))
    plt.rcParams.update({'font.size': 26})
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))
    if xlim_range is not None:
        max_X = xlim_range[1]
    else:
        max_X = max(X)
    if max_X >= 20:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
    else:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.plot(X, y_pred, color='#1f77b4', linewidth=4)
    plt.scatter(X, Y, color='k', s=80)

    y_ticks = np.arange(0, 1.01, 0.01)
    y_max_tick = round(y_ticks[y_ticks > y_max][0], 2)
    y_min_tick = round(y_ticks[y_ticks < y_min][-1], 2)
    if y_min_tick == 0:
        y_min_tick = round_to_first_nonzero(y_min)
    y_middle_tick = np.power(10, (np.log10(y_min_tick) + np.log10(y_max_tick)) / 2)
    if xlim_range is not None:
        plt.xlim(xlim_range[0], xlim_range[1])
    plt.yticks([np.log10(y_min_tick), np.log10(y_middle_tick), np.log10(y_max_tick)], [f'{y_min_tick: .2f}', f'{y_middle_tick:.2f}', f'{y_max_tick: .2f}'])
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches='tight', format='eps')
    plt.close()

def analyze_terminal_features_separation_fuzziness(rate_reduction_list, terminal_layer_path):
    rate_reduction_list = np.array(rate_reduction_list)
    X = np.arange(len(rate_reduction_list)) + 1
    Y = rate_reduction_list
    X, Y, y_pred, rsquare = linear_regression(X, Y)
    print('pearson correlation', stats.pearsonr(X.reshape(-1), Y))
    plt.rcParams.update({'font.size': 26})
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if max(X) >= 20:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
        # major_ticks = [1] + list(np.arange(10, max(X) + 10, 10))
        # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    else:
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
        # major_ticks = [1] + list(np.arange(5, max(X) + 5, 5))
        # plt.gca().xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    plt.xlim(0, max(X) + 1)
    plt.plot(X, y_pred, color='#1f77b4', linewidth=4)
    plt.scatter(X, Y, color='k', s=80)
    min_value = math.floor(min(np.log10(rate_reduction_list)))
    max_value = math.ceil(max(np.log10(rate_reduction_list)))
    y_ticks = []
    for i in list(range(min_value, max_value + 1)):
        y_ticks.append(r'$10^{}$'.format(i))
    plt.yticks(np.arange(min_value, max_value + 1), y_ticks, fontsize=26)
    plt.tight_layout()
    plt.savefig(terminal_layer_path, bbox_inches='tight', format='eps')
    plt.close()


def feature_standardization(inputs):
    mean = np.mean(inputs, axis=-1, keepdims=True)
    std = np.std(inputs, axis=-1, keepdims=True)
    normalized_inputs = (inputs - mean) / std
    return normalized_inputs
