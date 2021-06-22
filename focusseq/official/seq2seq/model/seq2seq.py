# -*- coding: utf-8 -*-

from __future__ import print_function
import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.ops import array_ops
import sys
sys.path.append("official/seq2seq/model/")
import beamsearch_decoder_helper
import attention_wrapper_helper as attention_wrapper

UNK = "UNK"  # 2
PAD = "<pad>"  # 0
EOS = "<EOS>"  # 1
UNK_ID = 2
PAD_ID = 0
EOS_ID = 1


NEG_INF = -1e9


class Seq2SeqModel(object):
    """
    An attention-based seq2seq model
    """

    def __init__(self, params, is_train, is_eval):

        self.params = vars(params)
        self.is_train = is_train
        self.is_eval = is_eval

    def __call__(self, inputs, targets=None):

        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")

        with tf.variable_scope("Seq2Seq", initializer=initializer, reuse=tf.AUTO_REUSE):
            # Calculate attention bias for encoder self-attention and decoder

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.

            self.encoder_inputs_length = tf.reduce_sum(tf.to_int32(tf.not_equal(inputs, PAD_ID)), axis=1)
            
            self.batch_size = tf.shape(inputs)[0]

            self.encoder_embeddings = tf.get_variable(
                name='encoder_embedding',
                shape=[self.params["source_vocab_size"],
                       self.params["hidden_size"]],
                dtype=tf.float32
            )

            self.decoder_embeddings = tf.get_variable(
                name='decoder_embedding',
                shape=[self.params["target_vocab_size"],
                       self.params["hidden_size"]],
                dtype=tf.float32)


            encoder_last_state, encoder_outputs = self.encode(inputs)

            self.output_layer = Dense(self.params["target_vocab_size"], name='output_projection')

            p_mu, p_sigma = self.prior(encoder_outputs) 

            if targets is None:
                latent_z = self.sample(p_mu, p_sigma)
            else:
                q_mu, q_sigma = self.posterior(targets, encoder_outputs)
                latent_z = self.sample(q_mu, q_sigma)

            focus_vector = self.get_focus_vector(latent_z, encoder_outputs, inputs)            

            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell(encoder_last_state, encoder_outputs, focus_vector)

            # Generate output sequence if targets is None, or return logits if target
            if targets is None:
                return self.predict()
            else:
                logits, coverage = self.decode(targets)

                bow_logits = self.get_bow(encoder_outputs, latent_z)
                kl_loss = self.kl_loss(p_mu, p_sigma, q_mu, q_sigma)
                focus_loss = self.focus_loss(focus_vector, coverage)
                return logits, kl_loss, bow_logits, focus_loss

    def get_focus_vector(self, latent_z, encoder_outputs, inputs):
        mask = tf.to_float(tf.equal(inputs, PAD_ID)) * NEG_INF +  tf.to_float(tf.equal(inputs, EOS_ID)) * NEG_INF
        latent_z = tf.expand_dims(latent_z, 1)
        focus_key_layer = Dense(self.params["hidden_size"], name="focus_key_layer")
        encoder_outputs = focus_key_layer(encoder_outputs)
        vf = tf.get_variable("f_attention", [self.params["hidden_size"]])
        focus_vector = tf.reduce_sum(vf * tf.nn.tanh(encoder_outputs + latent_z), [2])
        #FOR EXPLICITLY DECODING
        ## focus_vector: [batch_size, encoder_sequence_length]
        focus_vector = tf.nn.softmax(focus_vector + mask)
        return focus_vector
        

    def focus_loss(self, focus_vector, coverage):
        loss = tf.reduce_sum((focus_vector - coverage)**2, axis=1) 
        #loss = tf.norm(focus_vector-coverage, axis=1) 
        return tf.reduce_mean(loss)


    def get_bow(self, encoder_outputs, latent_z):
        encoder_output = tf.reduce_mean(encoder_outputs, axis=1)
        mlp = Dense(self.params["target_vocab_size"], name="bow_mlp", use_bias=True)
        logits = mlp(tf.concat([encoder_output, latent_z], axis=-1))
        return logits 



    def kl_loss(self, p_mu, p_sigma, q_mu, q_sigma):
        kl_losses = 0.5 * tf.reduce_sum(
            tf.exp(q_sigma - p_sigma) +
            (q_mu - p_mu) ** 2 / tf.exp(p_sigma) - 1. - q_sigma + p_sigma, axis=1)
        return tf.reduce_mean(kl_losses)


    def posterior(self, targets, encoder_outputs):
        """
        generate posterior mu and sigma
        """
        encoder_output = tf.reduce_mean(encoder_outputs, 1)
        _, decoder_outputs = self.encode(targets, is_input=False)
        decoder_output = tf.reduce_mean(decoder_outputs, 1)
        encoder_decoder_output = tf.concat([encoder_output, decoder_output], axis=-1) 

        dense_p_mu    = Dense(self.params["hidden_size"], use_bias=True)
        dense_p_sigma = Dense(self.params["hidden_size"], use_bias=True, activation=tf.nn.relu)

        mu = dense_p_mu(encoder_decoder_output)
        sigma = dense_p_sigma(encoder_decoder_output)

        return mu, sigma

   
    def prior(self, encoder_outputs):                                                                                                                        
        """
        generate prior mu and sigma
        """
        encoder_output = tf.reduce_mean(encoder_outputs, axis=1)

        dense_p_mu    = Dense(self.params["hidden_size"], use_bias=True)
        dense_p_sigma = Dense(self.params["hidden_size"], use_bias=True, activation=tf.nn.relu)

        mu = dense_p_mu(encoder_output)
        sigma = dense_p_sigma(encoder_output)

        return mu, sigma


    def sample(self, mu, sigma):
        normal = tf.random_normal(shape=tf.shape(mu))
        z_sample = mu + tf.exp(sigma / 2.) * normal
        return z_sample
    


    def encode(self, inputs, is_input=True):
        print("building encoder..")

        # Building encoder_cell
        encoder_inputs_length = tf.reduce_sum(tf.to_int32(tf.not_equal(inputs, PAD_ID)), axis=1)

        # Embedded_inputs: [batch_size, time_step, embedding_size]
        if is_input: 
            encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=inputs)
        else:
            encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.decoder_embeddings, ids=inputs)

        # Input projection layer to feed embedded inputs to the cell
        # ** Essential when use_residual=True to match input/output dims
        # input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

        # Embedded inputs having gone through input projection layer
        # self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)
        # Encode input sequences into context vectors:
        # encoder_outputs: [batch_size, max_time_step, cell_output_size]
        # encoder_state: [batch_size, cell_output_size]


        if self.params["bidirectional"]:
            encoder_inputs_word_embedded = encoder_inputs_embedded
            for layer_id in range(self.params["depth"]):
                encoder_cell_fw, encoder_cell_bw = self.build_encoder_cell() 
                encoder_outputs, encoder_last_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_cell_fw,
                    cell_bw=encoder_cell_bw,
                    inputs=encoder_inputs_embedded,
                    sequence_length=encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=False
                )
                concat_layer = Dense(self.params["hidden_size"], name='bidirection_concat_%d'%layer_id)
                encoder_outputs = concat_layer(tf.concat([encoder_outputs[0], encoder_outputs[1]], axis=-1))
                
                encoder_inputs_embedded = encoder_outputs
        else:
            raise Exception("No support uni-directional")
        
        encoder_outputs += encoder_inputs_word_embedded


        
        encoder_last_state = [tf.reduce_mean(encoder_outputs, 1) for _ in range(self.params["depth"])]

        return encoder_last_state, encoder_outputs

    def decode(self, targets):
        print("building decoder and attention..")
        
        self.decoder_inputs_length_train = tf.reduce_sum(tf.to_int32(tf.not_equal(targets, PAD_ID)), axis=1)

    
        # Input projection layer to feed embedded inputs to the cell
        # ** Essential when use_residual=True to match input/output dims
        # input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')

        # Output projection layer to convert cell_outputs to logits
        

        decoder_inputs_embedded = tf.nn.embedding_lookup(
            params=self.decoder_embeddings, ids=targets)


        # Shift the targets to the right, and remove the last element
        decoder_inputs_embedded = tf.pad(decoder_inputs_embedded, 
            [[0, 0], [1, 0], [0, 0]])[:, :-1, :]


        # Embedded inputs having gone through input projection layer

        # Helper to feed inputs for training: read inputs from dense ground truth vectors
        training_helper = seq2seq.TrainingHelper(
            inputs=decoder_inputs_embedded,
            sequence_length=self.decoder_inputs_length_train,
            time_major=False,
            name='training_helper')

        training_decoder = seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=training_helper,
            initial_state=self.decoder_initial_state,
            output_layer=self.output_layer)

        # Maximum decoder time_steps in current batch
        max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train) 

        # decoder_outputs_train: BasicDecoderOutput
        #                        namedtuple(rnn_outputs, sample_id)
        # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
        #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
        # decoder_outputs_train.sample_id: [batch_size], tf.int32
        decoder_outputs_train, decoder_last_state_train,\
        decoder_outputs_length_train = (seq2seq.dynamic_decode(
            decoder=training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_decoder_length))

        # More efficient to do the projection on the batch-time-concatenated tensor
        # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
        # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)

        decoder_logits_train = decoder_outputs_train.rnn_output

                
        #alignment_history : [decoder_sequence_length, batch_size, encoder_sequence_length]
        alignment_history = decoder_last_state_train[-1].alignment_history.stack()

        masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
            maxlen=max_decoder_length, dtype=tf.float32, name='masks')
        
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(alignment_history)[2], 1])
        masks = tf.transpose(masks, [2, 0, 1])

        # coverage: [batch_size, encoder_sequence_length]
        coverage = tf.reduce_sum(alignment_history * masks, axis=0)
        coverage = coverage / tf.to_float(tf.expand_dims(self.decoder_inputs_length_train, 1))
        

        return decoder_logits_train, coverage

    def predict(self):

        # Start_tokens: [batch_size,] `int32` vector
        start_tokens = tf.zeros([self.batch_size, ], tf.int32) # start_token

        def embed_and_input_proj(inputs):
            return tf.nn.embedding_lookup(self.decoder_embeddings, inputs)

        
        inference_decoder = beamsearch_decoder_helper.BeamSearchIgnoreUnkDecoder(
            cell=self.decoder_cell,
            embedding=embed_and_input_proj,
            start_tokens=start_tokens,
            end_token=EOS_ID,
            initial_state=self.decoder_initial_state,
            beam_width=self.params["beam_width"],
            output_layer=self.output_layer, )



        self.decoder_outputs_decode, self.decoder_last_state_decode, \
            self.decoder_outputs_length_decode = seq2seq.dynamic_decode(
                decoder=inference_decoder,
                output_time_major=False,
                # impute_finished=True,	# error occurs
                maximum_iterations=self.params["max_decode_step"])

        decoder_pred_decode = self.decoder_outputs_decode.predicted_ids


        return decoder_pred_decode

    def build_single_cell(self):
        cell_type = LSTMCell
        if self.params["cell_type"].lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.params["hidden_size"])

        if self.is_train:
            cell = DropoutWrapper(cell, 
                dtype=tf.float32,
                input_keep_prob=1.0-self.params["dropout"],
                output_keep_prob=1.0-self.params["dropout"],
                state_keep_prob=1.0-self.params["dropout"],
                variational_recurrent=False,
                input_size=tf.TensorShape([self.params["hidden_size"]]))

        if self.params["residual"]:
            cell = ResidualWrapper(cell)

        return cell

    # Building encoder cell
    def build_encoder_cell(self):
        if self.params["bidirectional"]:
            multi_cell_fw = self.build_single_cell() #MultiRNNCell([self.build_single_cell() for _ in range(self.params["depth"])])
            multi_cell_bw = self.build_single_cell() #MultiRNNCell([self.build_single_cell() for _ in range(self.params["depth"])])
            return multi_cell_fw, multi_cell_bw
        else:
            return MultiRNNCell([self.build_single_cell() for _ in range(self.params["depth"])]), None

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self, encoder_last_state, encoder_outputs, focus_vector):
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if not self.is_train and not self.is_eval:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.params["beam_width"])
            focus_vector = seq2seq.tile_batch(
                focus_vector, multiplier=self.params["beam_width"])
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.params["beam_width"]), encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.params["beam_width"])

        ## 'Luong' style attention: https://arxiv.org/abs/1508.04025
        #if self.params["attention"].lower() == 'luong':
        #    attention_mechanism = attention_wrapper.LuongAttention(
        #        num_units=self.params["hidden_size"], memory=encoder_outputs,
        #        memory_sequence_length=encoder_inputs_length, )
        #else:
        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.params["hidden_size"], memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length, focus_vector=focus_vector)
         
        # Building decoder_cell
        decoder_cell_list = [
            self.build_single_cell() for _ in range(self.params["depth"])]

        #decoder_initial_state = encoder_last_state
        def attn_decoder_input_fn(inputs, attention):
            if not self.params["attn_input_feeding"]:
                # no feed the input of the cell with last attention information
                return inputs
            # Essential when use_residual=True
            _input_layer = Dense(self.params["hidden_size"], dtype=tf.float32,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer


        using_alignment_history = True #self.is_train or self.is_eval

        decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=decoder_cell_list[-1],
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.params["hidden_size"],
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=using_alignment_history,
            name='Attention_Wrapper',
            encoder_sequence_length=tf.shape(encoder_outputs)[1])
        

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        if not self.is_train and not self.is_eval:
            batch_size = self.batch_size * self.params["beam_width"]
        else:
            batch_size = self.batch_size 

        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_last_state[-1])
        #initial_state[-1] = decoder_cell_list[-1].get_initial_state(
        #    batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_last_state[-1])



        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(decoder_cell_list), decoder_initial_state

