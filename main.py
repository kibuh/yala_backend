from __future__ import division, print_function
import numpy as np

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify




# Model saved with Keras model.save()
num_encoder_tokens = 40
num_decoder_tokens = 67
max_encoder_seq_length = 36
max_decoder_seq_length = 47

LSTM_PATH = 'models/lstm_model.h5'
ENCODER_PATH = 'models/encoder_model.h5'
DECODER_PATH = 'models/decoder_model.h5'

# Load your trained model
model = load_model(LSTM_PATH)
encoder_model = load_model(ENCODER_PATH)
decoder_model = load_model(DECODER_PATH)
model.compile()
decoder_model.compile()
encoder_model.compile()


# Hyperparameters
batch_size = 64
latent_dim = 256
num_samples = 10000

# Vectorize the data.
input_texts = []
target_texts = []
input_chars = set()
target_chars = set()

with open('files/eng_yala.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_chars:
            input_chars.add(char)
    for char in target_text:
        if char not in target_chars:
            target_chars.add(char)

input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))
num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Define data for encoder and decoder
input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_in_data[i, t, input_token_id[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_in_data[i, t, target_token_id[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_id[char]] = 1.

reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())


# Define Decode Sequence
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Get the first character of target sequence with the start character.
    target_seq[0, 0, target_token_id['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def translate_english_to_yala(english_sentence):
    # Use a breakpoint in the code line below to debug your script.
    input_seq = np.zeros((len(english_sentence), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoded_sentence = decode_sequence(input_seq)
    return decoded_sentence

# Define a flask app
app = Flask(__name__)

@app.route('/api/v1/translate', methods=['POST'])
def handle_translation():
    body = request.json
    yala_sentence = translate_english_to_yala(body['input'])

    return jsonify({'english': body['input'][0], 'yala': yala_sentence})


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0', use_debugger=True, port=6007)
    # print_hi('PyCharm')

