import time
import sys
import torch


from utils import set_random_seed, save_data_with_pickle, load_data_from_pickle
from feature_quality_assessment import get_layer_feature_quality
from feature_learning import sample_sentences, convert_sentences_to_examples, get_features_only, load_model, \
    normalize_features, load_model_from_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    config = {'seed': 666, 'model_option': model_option, 'model_phase_option': model_phase_option,
              'data_option': data_option, 'sent_num': sent_num, 'batch_size': batch_size,
              'measure_option': measure_option, 'normalize_option': normalize_option,
              'dir_path': '/path/to/working/dir/', 'model_dir': '/path/to/model/dir/'}

    if config['data_option'] == 'bookcorpus':
        feature_data_path = config['dir_path'] + 'data/{0}_{2}_{1}_features_size={3}_seed={4}.pickle'.format(
            config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'], config['seed'])
    else:
        feature_data_path = config['dir_path'] + 'data/{0}_{2}_{1}_features_size={3}_seed={4}_truncate=512.pickle'.format(
            config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'], config['seed'])
    feature_figure_path = config['dir_path'] + 'figures/{0}_{2}_{1}_features_size={3}_seed={4}_{5}_{6}.eps'.format(
        config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'], config['seed'],
        config['measure_option'], config['normalize_option'])

    if len(sys.argv) > 9:
        token_option = sys.argv[9].split('=')[1]
        config['token_option'] = token_option
        feature_figure_path = config['dir_path'] + 'figures/{0}_{2}_{1}_features_size={3}_seed={4}_{5}_{6}_{7}.eps'.format(
            config['model_option'], config['model_phase_option'], config['data_option'], config['sent_num'],
            config['seed'], config['measure_option'], config['normalize_option'], config['token_option'])

    if feature_option == 'save':
        set_random_seed(config['seed'])
        print('sample sentences')
        sentences = sample_sentences(config)

        set_random_seed(config['seed'])
        print('load model')
        if 'global_step' in config['model_option']:
            tokenizer, model = load_model_from_path(config)
        else:
            tokenizer, model = load_model(config)
        set_random_seed(config['seed'])
        print('get features')
        tokens, feature_data = get_features_only(sentences, tokenizer, model, config)
        print('save data')
        save_data_with_pickle((sentences, tokens, feature_data), feature_data_path)
    else:
        set_random_seed(config['seed'])
        print('load data')
        sentences, tokens, feature_data = load_data_from_pickle(feature_data_path)
        print('load model')
        if 'global_step' in config['model_option']:
            tokenizer, model = load_model_from_path(config)
        else:
            tokenizer, model = load_model(config)
        print('normalize features')
        if config['normalize_option'] == 'no-normalization':
            print('no normalization')
        else:
            feature_data = normalize_features(feature_data, model, config, normalize_option=normalize_option)
        print('convert sentences to examples')
        token_features, token_outputs, token_vocab = convert_sentences_to_examples(tokens, feature_data, tokenizer, config)
        print('get layer-wise feature quality')
        get_layer_feature_quality(feature_figure_path, token_features, token_outputs, measure_option=measure_option)
    time_end = time.time()
    print('time:', time_end - time_start)


