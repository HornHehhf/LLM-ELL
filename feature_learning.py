import numpy as np
import torch
from torch import nn
from datasets import load_dataset
import random

from transformers import AutoModel, AutoTokenizer, AutoConfig, T5EncoderModel, T5ForConditionalGeneration

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm
from transformers.models.mamba.modeling_mamba import MambaRMSNorm
from transformers.models.t5.modeling_t5 import T5LayerNorm

from utils import feature_standardization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_sentences(config):
    if config['data_option'] == 'c4':
        dataset = load_dataset("stas/c4-en-10k")
    elif config['data_option'] == 'openwebtext':
        dataset = load_dataset("stas/openwebtext-10k")
    elif config['data_option'] == 'wiki':
        dataset = load_dataset("NeelNanda/wiki-10k")
    elif config['data_option'] == 'pes2o':
        dataset = load_dataset("nampdn-ai/mini-peS2o")
    elif config['data_option'] == 'pile':
        dataset = load_dataset("NeelNanda/pile-10k")
    elif config['data_option'] == 'redpajama':
        dataset = load_dataset("michelangelo-engs/RedPajama-Data-1T-1024Sample")
    elif config['data_option'] == 'oscar':
        dataset = load_dataset("stas/oscar-en-10k")
    elif config['data_option'] == 'MedRAG-textbooks':
        dataset = load_dataset('MedRAG/textbooks')
    elif config['data_option'] == 'legalbench':
        dataset = load_dataset("nguha/legalbench", name='unfair_tos')
    elif config['data_option'] == 'us-congressional-speeches':
        dataset = load_dataset('Eugleo/us-congressional-speeches')
    else:
        dataset = load_dataset(config['data_option'])
    if config['data_option'] == 'legalbench':
        sent_id_list = np.arange(len(dataset['test']))
    else:
        sent_id_list = np.arange(len(dataset['train']))
    np.random.shuffle(sent_id_list)
    chosen_sentences = []
    for i in range(config['sent_num']):
        text_option = 'text'
        if config['data_option'] == 'MedRAG-textbooks':
            text_option = 'content'
        if config['data_option'] == 'legalbench':
            chosen_sentences.append(dataset['test'][int(sent_id_list[i])][text_option])
        else:
            chosen_sentences.append(dataset['train'][int(sent_id_list[i])][text_option])
    return chosen_sentences


def load_model(config):
    tokenizer, model = None, None

    if config['model_option'] == 'openai-gpt':
        tokenizer = AutoTokenizer.from_pretrained(config['model_option'])
        model = AutoModel.from_pretrained(config['model_option'])
        model.to(device)
    elif config['model_option'][:4] == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(config['model_option'])
        model = AutoModel.from_pretrained(config['model_option'])
        model.to(device)
    elif config['model_option'][:5] == "llama":
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/" + config['model_option'])
        model = AutoModel.from_pretrained("huggyllama/" + config['model_option'], device_map="auto")
    elif config['model_option'][:7] == "Llama-2":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/" + config['model_option'])
        model = AutoModel.from_pretrained("meta-llama/" + config['model_option'], device_map="auto")
    elif config['model_option'][:12] == 'Meta-Llama-3':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/" + config['model_option'])
        model = AutoModel.from_pretrained("meta-llama/" + config['model_option'], device_map="auto")
    elif config['model_option'][:7] == 'Mistral':
        if config['model_option'] == 'Mistral-7B-v0.2':
            tokenizer = AutoTokenizer.from_pretrained("mistral-community/" + config['model_option'])
            model = AutoModel.from_pretrained("mistral-community/" + config['model_option'])
        else:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/" + config['model_option'])
            model = AutoModel.from_pretrained("mistralai/" + config['model_option'])
        model.to(device)
    elif config['model_option'][:3].lower() == 'phi':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/" + config['model_option'])
        model = AutoModel.from_pretrained("microsoft/" + config['model_option'], device_map="auto")
    elif config['model_option'][:6] == 'pythia':
        model_option = config['model_option'].split('_')[0]
        step_option = config['model_option'].split('_')[1]
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/" + model_option, revision=step_option)
        model = AutoModel.from_pretrained("EleutherAI/" + model_option, revision=step_option, device_map="auto")
    elif config['model_option'][:4] == 'rwkv':
        tokenizer = AutoTokenizer.from_pretrained("RWKV/" + config['model_option'])
        model = AutoModel.from_pretrained("RWKV/" + config['model_option'], device_map="auto")
    elif config['model_option'][:5] == 'mamba':
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/" + config['model_option'])
        model = AutoModel.from_pretrained("state-spaces/" + config['model_option'], device_map="auto")
    elif config['model_option'][:4] == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("google-bert/" + config['model_option'])
        model = AutoModel.from_pretrained("google-bert/" + config['model_option'])
        model.to(device)
    elif config['model_option'][:7] == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/" + config['model_option'])
        model = AutoModel.from_pretrained("FacebookAI/" + config['model_option'])
        model.to(device)
    elif config['model_option'][:2] == 't5':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/" + config['model_option'])
        # to run next token prediction with T5, we need to use T5EncoderModel
        model = T5EncoderModel.from_pretrained("google-t5/" + config['model_option'], device_map="auto")
        if 'task_option' in config:
            if config['task_option'] == 'sc':
                # to run span corruption, we need to use T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained("google-t5/" + config['model_option'],
                                                                   device_map="auto")

    # Without this part, the model cannot deal with batches
    tokenizer.add_special_tokens({'eos_token': '<eos>'})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def load_model_from_path(config):
    model_path = config['model_dir'] + config['model_option']
    if config['model_option'][:4] == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        tokenizer = None
    model_config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=model_config)

    # Without this part, the model cannot deal with batches
    tokenizer.add_special_tokens({'eos_token': '<eos>'})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    return tokenizer, model


def get_features_only(sentences, tokenizer, model, config):
    all_tokens = []
    all_features = []
    for i in range(0, len(sentences), config['batch_size']):
        print('sentence', i, end='\r')
        batch_sentences = sentences[i:i + config['batch_size']]
        if config['data_option'] == 'bookcorpus':
            inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        else:
            inputs = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_masks = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
            features = outputs.hidden_states
        for j in range(features[0].shape[0]):
            mask = attention_masks[j]
            non_padded_indices = mask.nonzero().squeeze(1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[j][non_padded_indices])
            all_tokens.append(tokens)
            sentence_features = [layer[j][non_padded_indices].to('cpu').detach().numpy() for layer in features]
            all_features.append(np.array(sentence_features))
            assert len(tokens) == np.array(sentence_features).shape[1]

    return all_tokens, all_features


def mask_tokens(inputs, tokenizer):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    labels = inputs.clone()

    # We sample a few tokens in each sequence for MLM training (with probability 0.15)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # Save original tokens at masked positions
    original_tokens = inputs.clone()

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels, original_tokens


def get_features_only_mlm(sentences, tokenizer, model, config):
    all_original_tokens = []
    all_masked_features = []

    for i in range(0, len(sentences), config['batch_size']):
        print('sentence', i, end='\r')
        batch_sentences = sentences[i:i + config['batch_size']]
        if config['data_option'] == 'bookcorpus':
            inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        else:
            inputs = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Mask some tokens in the input_ids
        input_ids, _, original_tokens = mask_tokens(inputs['input_ids'], tokenizer)
        inputs['input_ids'] = input_ids

        attention_masks = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
            features = outputs.hidden_states

        for j in range(features[0].shape[0]):
            mask = (input_ids[j] == tokenizer.mask_token_id)
            masked_indices = mask.nonzero().squeeze(1)

            original_tokens_batch = tokenizer.convert_ids_to_tokens(original_tokens[j][masked_indices].tolist())
            all_original_tokens.append(original_tokens_batch)

            masked_features = [layer[j][masked_indices].cpu().detach().numpy() for layer in features]
            all_masked_features.append(np.array(masked_features))

    return all_original_tokens, all_masked_features


def corrupt_text(text, tokenizer, corruption_rate=0.15):
    tokens = tokenizer.tokenize(text)
    corrupted_tokens = []
    targets = []
    current_span = []

    for i, token in enumerate(tokens):
        if random.random() < corruption_rate:
            current_span.append(token)
        else:
            if current_span:
                corrupted_tokens.append(f"<extra_id_{len(targets)}>")
                targets.append("".join(current_span))
                current_span = []
            corrupted_tokens.append(token)

    if current_span:
        corrupted_tokens.append(f"<extra_id_{len(targets)}>")
        targets.append("".join(current_span))

    corrupted_text = "".join(corrupted_tokens)
    target_text = "".join([f"<extra_id_{i}>{target}" for i, target in enumerate(targets)])

    return corrupted_text, target_text


def get_features_only_sc(sentences, tokenizer, model, config):
    all_features = []
    all_targets = []

    total_loss = 0
    for i in range(0, len(sentences), config['batch_size']):
        print('sentence', i, end='\r')
        batch_sentences = sentences[i:i + config['batch_size']]

        # Corrupt and tokenize sentences
        corrupted_texts, target_texts = zip(*[corrupt_text(sentence, tokenizer) for sentence in batch_sentences])

        if i == 0:
            print('corrupted texts:', corrupted_texts)
            print('target texts:', target_texts)
        if config['data_option'] == 'bookcorpus':
            inputs = tokenizer(corrupted_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        else:
            inputs = tokenizer(corrupted_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                device)
            targets = tokenizer(target_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
                device)

        input_ids = inputs['input_ids']
        attention_masks = inputs["attention_mask"]
        target_ids = targets['input_ids']
        target_attention_masks = targets["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, decoder_input_ids=target_ids,
                            decoder_attention_mask=target_attention_masks, output_hidden_states=True, return_dict=True)
            features = outputs.decoder_hidden_states

            total_loss += model(input_ids=input_ids, attention_mask=attention_masks, labels=target_ids,
                                decoder_attention_mask=target_attention_masks).loss.item()

        # Decode tokens and extract features
        for j in range(features[0].shape[0]):
            target_mask = target_attention_masks[j]
            non_padded_indices = target_mask.nonzero(as_tuple=True)[0]
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids[j][non_padded_indices])
            all_targets.append(target_tokens)
            sentence_features = [layer[j][non_padded_indices].cpu().detach().numpy() for layer in features]
            all_features.append(np.array(sentence_features))
            assert len(target_tokens) == np.array(sentence_features).shape[1]
            if i == 0 and j == 0:
                print('target tokens:', target_tokens)
                print('feature shape:', np.array(sentence_features).shape)
    print('total loss:', total_loss)

    return all_targets, all_features


def normalize_features(all_features, model, config, normalize_option):
    if config['model_option'][:4] == 'gpt2':
        ln_f = nn.LayerNorm(model.config.hidden_size, eps=model.config.layer_norm_epsilon).to(device)
    elif config['model_option'][:5] == 'llama' or config['model_option'][:7] == "Llama-2" \
            or config['model_option'][:12] == 'Meta-Llama-3':
        ln_f = LlamaRMSNorm(model.config.hidden_size, eps=model.config.rms_norm_eps).to(device)
    elif config['model_option'][:7] == 'Mistral':
        ln_f = MistralRMSNorm(model.config.hidden_size, eps=model.config.rms_norm_eps).to(device)
    elif config['model_option'][:3] == 'phi':
        ln_f = nn.LayerNorm(model.config.hidden_size, eps=model.config.layer_norm_eps).to(device)
    elif config['model_option'][:5] == 'Phi-3':
        ln_f = Phi3RMSNorm(model.config.hidden_size, eps=model.config.rms_norm_eps).to(device)
    elif config['model_option'][:6] == 'pythia':
        ln_f = nn.LayerNorm(model.config.hidden_size, eps=model.config.layer_norm_eps).to(device)
    elif config['model_option'][:4] == 'rwkv':
        ln_f = nn.LayerNorm(model.config.hidden_size, eps=model.config.layer_norm_epsilon).to(device)
    elif config['model_option'][:5] == 'mamba':
        ln_f = MambaRMSNorm(model.config.hidden_size, eps=model.config.layer_norm_epsilon).to(device)
    elif config['model_option'][:2] == 't5':
        ln_f = T5LayerNorm(model.config.d_model, eps=model.config.layer_norm_epsilon).to(device)
    else:
        print('no normalization {}'.format(config['model_option']))
        return all_features
    for param in ln_f.parameters():
        param.requires_grad = False

    normalized_all_features = []
    for i, sentence_features in enumerate(all_features):
        normalized_features = []
        if normalize_option == 'initialized-LN':
            with torch.no_grad():
                normalized_features = [ln_f(torch.tensor(state).to(device)).to('cpu').detach().numpy() for state in sentence_features[:-1]]
                normalized_features.append(sentence_features[-1])
        elif normalize_option == 'standardization':
            normalized_features = feature_standardization(np.array(sentence_features[:-1])).tolist()
            normalized_features.append(sentence_features[-1])
        normalized_all_features.append(np.array(normalized_features))
    return normalized_all_features


def convert_sentences_to_examples(tokens, feature_data, tokenizer, config):
    token_features = []
    token_outputs = []
    token_vocab = {}
    num_sents = len(tokens)
    for i in range(num_sents):
        cur_tokens = tokens[i]
        cur_features = feature_data[i]
        if 'token_option' not in config:
            token_option = 'next'
        else:
            token_option = config['token_option']
        if token_option in ['current']:
            for j in range(len(cur_tokens)):
                cur_token_features = cur_features[:, j, :]
                cur_output = cur_tokens[j]
                token_features.append(cur_token_features)
                token_outputs.append(cur_output)
                if cur_output in token_vocab:
                    token_vocab[cur_output] += 1
                else:
                    token_vocab[cur_output] = 1
        elif token_option == 'next-next':
            for j in range(len(cur_tokens))[:-2]:
                cur_token_features = cur_features[:, j, :]
                cur_output = cur_tokens[j + 2]
                token_features.append(cur_token_features)
                token_outputs.append(cur_output)
                if cur_output in token_vocab:
                    token_vocab[cur_output] += 1
                else:
                    token_vocab[cur_output] = 1
        elif token_option == 'previous':
            for j in range(len(cur_tokens))[1:]:
                cur_token_features = cur_features[:, j, :]
                cur_output = cur_tokens[j - 1]
                token_features.append(cur_token_features)
                token_outputs.append(cur_output)
                if cur_output in token_vocab:
                    token_vocab[cur_output] += 1
                else:
                    token_vocab[cur_output] = 1
        else:
            for j in range(len(cur_tokens))[:-1]:
                # ignore the last token since we don't know the special eos token; but can be added later
                cur_token_features = cur_features[:, j, :]
                cur_output = cur_tokens[j + 1]
                token_features.append(cur_token_features)
                token_outputs.append(cur_output)
                if cur_output in token_vocab:
                    token_vocab[cur_output] += 1
                else:
                    token_vocab[cur_output] = 1
    token_features = np.array(token_features)
    token_vocab = {k: v for k, v in sorted(token_vocab.items(), key=lambda item: item[1], reverse=True)}
    token_outputs = np.array(tokenizer.convert_tokens_to_ids(token_outputs))
    print("token features shape:", token_features.shape)
    print("token vocab size:", len(token_vocab))
    print("token outputs shape:", token_outputs.shape)
    return token_features, token_outputs, token_vocab


def convert_sentences_to_examples_mlm(tokens, feature_data, tokenizer, config=None):
    token_features = []
    token_outputs = []
    token_vocab = {}
    num_sents = len(tokens)
    for i in range(num_sents):
        cur_tokens = tokens[i]
        cur_features = feature_data[i]
        for j in range(len(cur_tokens))[:-1]:
            cur_token_features = cur_features[:, j, :]
            cur_output = cur_tokens[j]
            token_features.append(cur_token_features)
            token_outputs.append(cur_output)
            if cur_output in token_vocab:
                token_vocab[cur_output] += 1
            else:
                token_vocab[cur_output] = 1
    token_features = np.array(token_features)
    token_vocab = {k: v for k, v in sorted(token_vocab.items(), key=lambda item: item[1], reverse=True)}
    token_outputs = np.array(tokenizer.convert_tokens_to_ids(token_outputs))
    print("token features shape:", token_features.shape)
    print("token vocab size:", len(token_vocab))
    print("token outputs shape:", token_outputs.shape)
    return token_features, token_outputs, token_vocab
