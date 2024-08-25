import time
import sys
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from utils import set_random_seed, load_data_from_pickle
from feature_learning import convert_sentences_to_examples, load_model, normalize_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_pca_visualization(figure_path, token_features, token_outputs, selected_tokens):
    print(len(selected_tokens))
    colors = ['red', 'blue', 'green']
    colors = colors[:len(selected_tokens)]
    label_to_color = {label: color for label, color in zip(selected_tokens, colors)}

    layer_num = token_features.shape[1]
    for i in range(layer_num):
        layer_features = token_features[:, i, :]
        pca = PCA(n_components=2, svd_solver='full')
        reduced_layer_features = pca.fit_transform(layer_features)
        print('Layer', i)
        for label in selected_tokens:
            plt.scatter(reduced_layer_features[token_outputs == label][:, 0], reduced_layer_features[token_outputs == label][:, 1],
                        color=label_to_color[label], label=label, s=100)
        plt.rcParams.update({'font.size': 20})
        if i == 0:
            plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(figure_path + 'pca_visualization_layer={}.eps'.format(i), dpi=300, bbox_inches='tight')
        plt.close()


def select_tokens(token_features, token_outputs, token_vocab, token_set):
    if token_set == 'they-them':
        token_list = ["they</w>", "them</w>"]
    elif token_set == 'have-had':
        token_list = ["have</w>", "had</w>"]
    elif token_set == 'are-is':
        token_list = ["are</w>", "is</w>"]
    elif token_set == 'medicine':
        token_list = ['patients</w>', 'cells</w>', 'disorder</w>']
    elif token_set == 'law':
        token_list = ['law</w>', 'policy</w>']
    elif token_set == 'politics':
        token_list = ['president</w>', 'country</w>']
    selected_tokens = []
    for token in token_list:
        if token in token_vocab:
            print(token, token_vocab[token])
            selected_tokens.append(token)
    print(selected_tokens)
    selected_token_features = []
    selected_token_outputs = []
    for index in range(len(token_outputs)):
        if token_outputs[index] in selected_tokens:
            selected_token_features.append(token_features[index])
            selected_token_outputs.append(token_outputs[index])
    selected_token_features = np.array(selected_token_features)
    selected_token_outputs = np.array(selected_token_outputs)
    return selected_token_features, selected_token_outputs, selected_tokens


if __name__ == '__main__':
    time_start = time.time()
    model_option = sys.argv[1].split('=')[1]
    model_phase_option = sys.argv[2].split('=')[1]
    data_option = sys.argv[3].split('=')[1]
    sent_num = int(sys.argv[4].split('=')[1])
    batch_size = int(sys.argv[5].split('=')[1])
    feature_option = sys.argv[6].split('=')[1]
    measure_option = sys.argv[7].split('=')[1]
    normalize_option = sys.argv[8].split('=')[1]
    token_set = sys.argv[9].split('=')[1]

    config = {'seed': 666, 'model_option': model_option, 'model_phase_option': model_phase_option,
              'data_option': data_option, 'sent_num': sent_num, 'batch_size': batch_size,
              'dir_path': '/path/to/working/dir/'}

    if config['data_option'] == 'bookcorpus':
        feature_data_path = config['dir_path'] + 'data/{0}_{2}_{1}_features_size={3}_seed={4}.pickle'.format(
            config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'],
            config['seed'])
    else:
        feature_data_path = config['dir_path'] + 'data/{0}_{2}_{1}_features_size={3}_seed={4}_truncate=512.pickle'.format(
            config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'],
            config['seed'])
    feature_visualization_figure_path = config['dir_path'] + 'figures/{0}_{2}_{1}_features_size={3}_seed={4}_{5}_'.format(
        config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'], config['seed'],
        token_set)

    set_random_seed(config['seed'])
    print('load data')
    sentences, tokens, feature_data = load_data_from_pickle(feature_data_path)
    print('load model')
    tokenizer, model = load_model(config)
    print('normalize features')
    feature_data = normalize_features(feature_data, model, config, normalize_option=normalize_option)
    print('convert sentences to examples')
    token_features, token_outputs, token_vocab = convert_sentences_to_examples(tokens, feature_data, tokenizer, config)
    # for token in token_vocab:
    #     print(token, token_vocab[token])
    token_outputs = tokenizer.convert_ids_to_tokens(token_outputs)
    selected_token_features, selected_token_outputs, selected_tokens = select_tokens(token_features, token_outputs, token_vocab, token_set=token_set)
    print('get feature pca visualization')
    feature_pca_visualization(feature_visualization_figure_path, selected_token_features, selected_token_outputs, selected_tokens)
    time_end = time.time()
    print('time:', time_end - time_start)


