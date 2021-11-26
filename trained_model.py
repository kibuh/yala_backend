import numpy as np

# Keras
from keras.models import load_model


class TrainedModel:

    LSTM_PATH = 'models/lstm_model.h5'
    ENCODER_PATH = 'models/encoder_model.h5'
    DECODER_PATH = 'models/decoder_model.h5'

    def __init__(self, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, batch_size, latent_dim, num_samples, input_texts, target_texts):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_chars = set()
        self.target_chars = set()
        self.encoder_model = load_model('models/encoder_model.h5')
        self.decoder_model = load_model('models/decoder_model.h5')
        self.decoder_model.compile()
        self.encoder_model.compile()
        self.input_token_id = dict([(char, i) for i, char in enumerate(self.input_chars)])
        self.target_token_id = dict([(char, i) for i, char in enumerate(self.target_chars)])


    def params_init(self):
        with open('files/eng_yala.txt', 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_chars:
                    self.input_chars.add(char)
            for char in target_text:
                if char not in self.target_chars:
                    self.target_chars.add(char)

        input_chars = sorted(list(self.input_chars))
        target_chars = sorted(list(self.target_chars))
        num_encoder_tokens = len(input_chars)
        num_decoder_tokens = len(target_chars)
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        # Define data for encoder and decoder
        input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
        target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

        encoder_in_data = np.zeros((len(self.input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

        decoder_in_data = np.zeros((len(self.input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        decoder_target_data = np.zeros((len(self.input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                encoder_in_data[i, t, input_token_id[char]] = 1.
            for t, char in enumerate(target_text):
                decoder_in_data[i, t, target_token_id[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_id[char]] = 1.

        reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())

    # Define Decode Sequence
    def decode_sequence(self, input_seq):

        self.params_init()

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Get the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_id['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def translate_english_to_yala(self,english_sentence):
        # Use a breakpoint in the code line below to debug your script.
        input_seq = np.zeros((len(english_sentence), self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        decoded_sentence = self.decode_sequence(input_seq)
        return decoded_sentence
