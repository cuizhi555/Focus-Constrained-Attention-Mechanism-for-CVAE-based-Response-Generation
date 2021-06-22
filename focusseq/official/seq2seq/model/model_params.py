"""Defines model parameters."""


class Seq2SeqBaseParams(object):
    """Parameters for the base model."""
    # Input params
    batch_size = 1024 # Maximum number of tokens per batch of examples.
    max_length = 56  # Maximum number of tokens per example.

    # Model params
    initializer_gain = 1.0  # Used in trainable variable initialization.
    vocab_size = 30000  # Number of tokens defined in the vocabulary file.
    hidden_size = 720  # Model dimension in the hidden layers.
    depth = 2  # Number of layers in the encoder and decoder stacks.

    # !!!
    source_vocab_size = vocab_size
    target_vocab_size = vocab_size

    shared_embedding_softmax_weights = False
    # done

    # Cell Type
    cell_type = "gru"
    bidirectional = True

    # Dropout values (only used when training)
    dropout = 0.2 
    attention = "bahdanau"
    attn_input_feeding = False
    residual = True

    # Training params
    label_smoothing = 0.1
    learning_rate = 2.0
    learning_rate_decay_rate = 1.0
    learning_rate_warmup_steps = 8000

    # Optimizer params
    optimizer_adam_beta1 = 0.9
    optimizer_adam_beta2 = 0.997
    optimizer_adam_epsilon = 1e-09

    # Default prediction params
    ignore_unk = True
    beamsearch = True #True
    max_decode_step = 56
    extra_decode_length = 56 
    beam_width = 1
    alpha = 0.6  # used to calculate length normalization in beam search

