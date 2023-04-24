import json
import pickle
import re


import numpy as np
from flask import Flask, request, jsonify
from keras.models import model_from_json
from keras.utils import pad_sequences
from nltk.corpus import stopwords
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model


max_len_text = 100
max_len_headlines = 15

with open('x_tokenizer.pkl', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

with open('y_tokenizer.pkl', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

# with open('encoder_model.json', 'r') as f:
#     emodel = json.load(f)
# encoder_model = model_from_json(json.dumps(emodel))

encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

# with open('decoder_model.json', 'r') as f:
#     dmodel = json.load(f)
# decoder_model = model_from_json(json.dumps(dmodel))

target_word_index = y_tokenizer.word_index
reverse_target_word_index = y_tokenizer.index_word

app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.json.get('text')
    if not input_text.strip():
        return jsonify({'error': 'Please enter some text.'})
    # input_seq = []
    # for word in input_text.split():
    #     if word in
    input_text = clean_text(input_text)
    x_tokenizer.fit_on_texts([input_text])
    input_seq = x_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len_text, padding='post')
    summary_text = decode_sequence(np.array(input_seq).reshape(1,100))
    print(summary_text)
    return jsonify({'summary': summary_text})


def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = re.sub('"', "'", text)
    text = re.sub(r'https?://.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    return text


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_len_headlines - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
    print("test")