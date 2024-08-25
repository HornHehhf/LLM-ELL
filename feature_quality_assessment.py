import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import copy


from utils import analyze_terminal_features, analyze_terminal_features_separation_fuzziness


def get_class_features(features, labels):
    class_features = {}
    for i in range(len(labels)):
        if labels[i] in class_features:
            class_features[labels[i]].append(features[i])
        else:
            class_features[labels[i]] = [features[i]]
    return class_features


def get_label_distribution(labels):
    label_distribution = {}
    for label in labels:
        if label in label_distribution:
            label_distribution[label] += 1
        else:
            label_distribution[label] = 1
    label_distribution = dict(sorted(label_distribution.items(), key=lambda item: item[0]))
    return label_distribution


def get_separation_fuzziness(features, labels):
    features = copy.deepcopy(features)
    labels = copy.deepcopy(labels)
    avg_feature = np.mean(features, axis=0)
    features -= avg_feature
    label_distribution = get_label_distribution(labels)
    class_features = get_class_features(features, labels)
    feature_dim = features.shape[-1]
    between_class_covariance = np.zeros((feature_dim, feature_dim))
    within_class_covariance = np.zeros((feature_dim, feature_dim))
    classes = np.unique(labels)
    for i in range(len(classes)):
        cur_class_features = np.array(class_features[classes[i]])
        cur_class_avg_feature = np.mean(cur_class_features, axis=0)
        between_class_covariance += np.matmul(cur_class_avg_feature.reshape(-1, 1),
                                              cur_class_avg_feature.reshape(1, -1)) * label_distribution[classes[i]]
        cur_class_centralized_features = cur_class_features - cur_class_avg_feature
        cur_class_covariance = np.matmul(np.transpose(cur_class_centralized_features), cur_class_centralized_features)
        within_class_covariance += cur_class_covariance
    between_class_covariance /= len(labels)
    within_class_covariance /= len(labels)
    between_class_inverse_covariance = scipy.linalg.pinv(between_class_covariance)
    print('check pseudo inverse:', np.allclose(between_class_covariance, np.dot(between_class_covariance, np.dot(between_class_inverse_covariance, between_class_covariance))))
    within_variation = np.trace(np.matmul(within_class_covariance, between_class_inverse_covariance))
    return within_variation


def Regression_measure(inputs, labels):
    print('labels shape:', labels.shape)
    reg = LinearRegression().fit(inputs, labels)
    predictions = reg.predict(inputs)
    loss = mean_squared_error(labels, predictions) / np.var(labels)
    return loss, predictions


def get_layer_feature_quality(figure_path, token_features, token_outputs, measure_option):
    if measure_option == 'shuffled-vocabulary':
        print('shuffled vocabulary')
        unique_labels = np.unique(token_outputs)
        random_mapping = np.random.permutation(unique_labels)
        label_mapping = dict(zip(unique_labels, random_mapping))
        token_outputs = np.array([label_mapping[label] for label in token_outputs])
    layer_num = token_features.shape[1]
    loss_list = []
    prediction_list = []
    for i in range(layer_num):
        if measure_option == 'separation-fuzziness':
            cur_loss = get_separation_fuzziness(token_features[:, i, :], token_outputs)
        else:
            cur_loss, cur_predictions = Regression_measure(token_features[:, i, :], token_outputs)
            prediction_list.append(cur_predictions)
        print('Layer', i, cur_loss)
        loss_list.append(cur_loss)
    print("loss without embedding layer scaling law")
    if measure_option == 'separation-fuzziness':
        analyze_terminal_features_separation_fuzziness(loss_list[1:], figure_path)
    else:
        analyze_terminal_features(loss_list[1:], figure_path)

