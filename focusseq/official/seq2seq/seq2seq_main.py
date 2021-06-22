"""
Train and evaluate the Seq2Seq model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.seq2seq.model import model_params
from official.seq2seq.model import seq2seq
from official.seq2seq.utils import dataset
from official.seq2seq.utils import metrics
from official.seq2seq.utils import tokenizer
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

PARAMS_MAP = {
    "base": model_params.Seq2SeqBaseParams,
}

DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = int(1e9)

# Dictionary containing tensors that are logged by the logging hooks. Each item
# maps a string to the tensor name.
TENSORS_TO_LOG = {
    "learning_rate": "model/get_train_op/learning_rate/learning_rate",
    "cross_entropy_loss": "model/cross_entropy",
    "kl_loss": "model/get_train_op/kl_loss", 
    "kl_weight": "model/get_train_op/kl_weight", 
    "bow_loss": "model/get_train_op/bow_loss", 
    "focus_loss": "model/get_train_op/focus_loss"}


def training_model_fn(features, labels, mode, params):
    return model_fn(features, labels, mode, params, None)


def eval_model_fn(features, labels, mode, params):
    return model_fn(features, labels, mode, params, None)


def model_fn(features, labels, mode, params, t_mode=None):
    """Defines how to train, evaluate and predict from the transformer model."""
    with tf.variable_scope("model"):
        inputs, targets = features, labels

        # Create model and get output logits.
        model = seq2seq.Seq2SeqModel(params, mode == tf.estimator.ModeKeys.TRAIN, mode == tf.estimator.ModeKeys.EVAL)
       
        if targets is None:
            output = model(inputs, targets)
        else:
            logits, kl_loss, bow_logits, focus_loss = model(inputs, targets)


        # When in prediction mode, the labels/targets is None. The model output
        # is the prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions={"result": output})


        # Calculate model loss.
        # xentropy contains the cross entropy loss of every nonpadding token in the
        # targets.
        tf.logging.info(logits)
        tf.logging.info(targets)

        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, params.label_smoothing, params.target_vocab_size)
        # Compute the weighted mean of the cross entropy losses
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        bow_xentropy, bow_weights = metrics.word_padded_cross_entropy_loss(
            bow_logits, targets, params.label_smoothing, params.target_vocab_size)
        # Compute the weighted mean of the cross entropy losses
        bow_loss = tf.reduce_mean(tf.reduce_sum(bow_xentropy * bow_weights, axis=1))
       

        

        # Save loss as named tensor that will be logged with the logging hook.
        tf.identity(loss, "cross_entropy")

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss)
        else:
            train_op, loss = get_train_op(loss, kl_loss, bow_loss, focus_loss, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")
        # Save learning rate value to TensorBoard summary.
        tf.summary.scalar("learning_rate", learning_rate)

        return learning_rate


def get_learning_rate_new(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        global_step = tf.to_float(tf.train.get_or_create_global_step())
        lr = learning_rate
        decay_rate = get_learning_rate_linear_warmup_rsqrt_decay(hidden_size, global_step, warmup_steps)
        lr *= decay_rate
        tf.identity(lr, "learning_rate")
        tf.summary.scalar("learning_rate", lr)
        return lr


def get_learning_rate_linear_warmup_rsqrt_decay(hidden_size, global_step, warmup_steps):
    return hidden_size**-0.5 * tf.minimum(
        (global_step + 1) * warmup_steps**-1.5, (global_step + 1)**-0.5)


def get_variable_initializer(hparams):
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")

def get_kl_weight(global_step, total_step, ratio):
    ratio = tf.constant(ratio)
    progress = tf.cast(global_step, tf.float32) / tf.cast(tf.constant(total_step), tf.float32)
    return (tf.nn.tanh(6.0 * progress / ratio - 5.0) + 1.0)/8.0


def get_train_op(loss, kl_loss, bow_loss, focus_loss, params):
    """Generate training operation that updates variables based on loss."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            params.learning_rate, params.hidden_size,
            params.learning_rate_warmup_steps)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params.optimizer_adam_beta1,
            beta2=params.optimizer_adam_beta2,
            epsilon=params.optimizer_adam_epsilon)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        ce_loss = loss
        kl_weight = get_kl_weight(global_step, params.train_steps, 0.75)
        loss = 0.8 * loss + 0.2 * ( kl_loss * kl_weight + bow_loss + focus_loss)

        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")

        # Save gradient norm to Tensorboard
        tf.summary.scalar("global_norm/gradient_norm",
                          tf.global_norm(list(zip(*gradients))[0]))

        tf.identity(ce_loss, "ce_loss")
        tf.identity(kl_loss, "kl_loss")
        tf.identity(bow_loss, "bow_loss")
        tf.identity(focus_loss, "focus_loss")
        tf.identity(kl_weight, "kl_weight")
        tf.summary.scalar("ce_loss", ce_loss)
        tf.summary.scalar("kl_loss", kl_loss) 
        tf.summary.scalar("bow_loss", bow_loss)
        tf.summary.scalar("focus_loss", focus_loss)
        tf.summary.scalar("kl_weight", kl_weight)

        return train_op, loss



def get_global_step(estimator):
    """Return estimator's last checkpoint."""
    return int(estimator.latest_checkpoint().split("-")[-1])


def train_schedule(
        estimator, train_eval_iterations, single_iteration_train_steps=None,
        single_iteration_train_epochs=None, train_hooks=None, benchmark_logger=None,
        bleu_source=None, bleu_ref=None, bleu_threshold=None, vocab_file_path_source=None, vocab_file_path_target=None,
        decode_to_path=None):
    """Train and evaluate model, and optionally compute model's BLEU score.
  
    **Step vs. Epoch vs. Iteration**
  
    Steps and epochs are canonical terms used in TensorFlow and general machine
    learning. They are used to describe running a single process (train/eval):
      - Step refers to running the process through a single or batch of examples.
      - Epoch refers to running the process through an entire dataset.
  
    E.g. training a dataset with 100 examples. The dataset is
    divided into 20 batches with 5 examples per batch. A single training step
    trains the model on one batch. After 20 training steps, the model will have
    trained on every batch in the dataset, or, in other words, one epoch.
  
    Meanwhile, iteration is used in this implementation to describe running
    multiple processes (training and eval).
      - A single iteration:
        1. trains the model for a specific number of steps or epochs.
        2. evaluates the model.
        3. (if source and ref files are provided) compute BLEU score.
  
    This function runs through multiple train+eval+bleu iterations.
  
    Args:
      estimator: tf.Estimator containing model to train.
      train_eval_iterations: Number of times to repeat the train+eval iteration.
      single_iteration_train_steps: Number of steps to train in one iteration.
      single_iteration_train_epochs: Number of epochs to train in one iteration.
      train_hooks: List of hooks to pass to the estimator during training.
      benchmark_logger: a BenchmarkLogger object that logs evaluation data
      bleu_source: File containing text to be translated for BLEU calculation.
      bleu_ref: File containing reference translations for BLEU calculation.
      bleu_threshold: minimum BLEU score before training is stopped.
      vocab_file_path: Path to vocabulary file used to subtokenize bleu_source.
  
    Raises:
      ValueError: if both or none of single_iteration_train_steps and
        single_iteration_train_epochs were defined.
    """
    # Ensure that exactly one of single_iteration_train_steps and
    # single_iteration_train_epochs is defined.
    if single_iteration_train_steps is None:
        if single_iteration_train_epochs is None:
            raise ValueError(
                "Exactly one of single_iteration_train_steps or "
                "single_iteration_train_epochs must be defined. Both were none.")
    else:
        if single_iteration_train_epochs is not None:
            raise ValueError(
                "Exactly one of single_iteration_train_steps or "
                "single_iteration_train_epochs must be defined. Both were defined.")

    evaluate_bleu = bleu_source is not None and bleu_ref is not None

    # Print details of training schedule.
    tf.logging.info("Training schedule:")
    if single_iteration_train_epochs is not None:
        tf.logging.info("\t1. Train for %d epochs." % single_iteration_train_epochs)
    else:
        tf.logging.info("\t1. Train for %d steps." % single_iteration_train_steps)
    tf.logging.info("\t2. Evaluate model.")
    if evaluate_bleu:
        tf.logging.info("\t3. Compute BLEU score.")
        if bleu_threshold is not None:
            tf.logging.info("Repeat above steps until the BLEU score reaches %f" %
                            bleu_threshold)
    if not evaluate_bleu or bleu_threshold is None:
        tf.logging.info("Repeat above steps %d times." % train_eval_iterations)

    if evaluate_bleu:
        # Create summary writer to log bleu score (values can be displayed in
        # Tensorboard).
        bleu_writer = tf.summary.FileWriter(
            os.path.join(estimator.model_dir, BLEU_DIR))
        if bleu_threshold is not None:
            # Change loop stopping condition if bleu_threshold is defined.
            train_eval_iterations = INF

    # Loop training/evaluation/bleu cycles
    for i in xrange(train_eval_iterations):
        tf.logging.info("Starting iteration %d" % (i + 1))
        tf.logging.info("Each_single_iteration_steps %d" %single_iteration_train_steps)
        # Train the model for single_iteration_train_steps or until the input fn
        # runs out of examples (if single_iteration_train_steps is None).
        estimator.train(
            dataset.train_input_fn, steps=single_iteration_train_steps,
            hooks=train_hooks)

        #eval_results = estimator.evaluate(dataset.eval_input_fn)
        #tf.logging.info("Evaluation results (iter %d/%d):" %
        #               (i + 1, train_eval_iterations))
        #tf.logging.info(eval_results)

        #if i+1 >= 26:
        #    break

        #benchmark_logger.log_evaluation_result(eval_results)

        ## The results from estimator.evaluate() are measured on an approximate
        ## translation, which utilize the target golden values provided. The actual
        ## bleu score must be computed using the estimator.predict() path, which
        ## outputs translations that are not based on golden values. The translations
        ## are compared to reference file to get the actual bleu score.
        #if evaluate_bleu:
        #    uncased_score, cased_score = evaluate_and_log_bleu(
        #        estimator, bleu_source, bleu_ref, vocab_file_path_source, vocab_file_path_target, decode_to_path)

        #    # Write actual bleu scores using summary writer and benchmark logger
        #    global_step = get_global_step(estimator)
        #    summary = tf.Summary(value=[
        #        tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
        #        tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
        #    ])
        #    bleu_writer.add_summary(summary, global_step)
        #    bleu_writer.flush()
        #    benchmark_logger.log_metric(
        #        "bleu_uncased", uncased_score, global_step=global_step)
        #    benchmark_logger.log_metric(
        #        "bleu_cased", cased_score, global_step=global_step)

        #    # Stop training if bleu stopping threshold is met.
        #    if model_helpers.past_stop_threshold(bleu_threshold, uncased_score):
        #        bleu_writer.close()
        #        break


def define_transformer_flags():
    """Add flags and flag validators for running transformer_main."""
    # Add common flags (data_dir, model_dir, train_epochs, etc.).
    flags_core.define_base(multi_gpu=True, export_dir=False)
    flags_core.define_performance(
        num_parallel_calls=True,
        inter_op=False,
        intra_op=False,
        synthetic_data=False,
        max_train_steps=False,
        dtype=False
    )
    flags_core.define_benchmark()

    # Set flags from the flags_core module as "key flags" so they're listed when
    # the '-h' flag is used. Without this line, the flags defined above are
    # only shown in the full `--helpful` help text.
    flags.adopt_module_key_flags(flags_core)

    # Add transformer-specific flags
    flags.DEFINE_string(
        name="param_set", short_name="mp", default="big",
       # enum_values=["base", "big", "beam_mid_1shard_4", "beam_mid_1shard_4_d0", "beam_mid_1shard_test4"],
        help=flags_core.help_wrap(
            "Parameter set to use when creating and training the model. The "
            "parameters define the input shape (batch size and max length), "
            "model configuration (size of embedding, # of hidden layers, etc.), "
            "and various other settings. The big parameter set increases the "
            "default batch size, embedding/hidden size, and filter size. For a "
            "complete list of parameters, please see model/model_params.py."))

    # Flags for training with steps (may be used for debugging)
    flags.DEFINE_integer(
        name="train_steps", short_name="ts", default=None,
        help=flags_core.help_wrap("The number of steps used to train."))
    flags.DEFINE_integer(
        name="steps_between_evals", short_name="sbe", default=1000,
        help=flags_core.help_wrap(
            "The Number of training steps to run between evaluations. This is "
            "used if --train_steps is defined."))
    # Visible GPU
    flags.DEFINE_string(
       name="gpus", default=None,
       help=flags_core.help_wrap(
            "Visible gpus for training"))

    # BLEU score computation
    flags.DEFINE_string(
        name="bleu_source", short_name="bls", default=None,
        help=flags_core.help_wrap(
            "Path to source file containing text translate when calculating the "
            "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
            "must be set. Use the flag --stop_threshold to stop the script based "
            "on the uncased BLEU score."))
    flags.DEFINE_string(
        name="bleu_ref", short_name="blr", default=None,
        help=flags_core.help_wrap(
            "Path to source file containing text translate when calculating the "
            "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
            "must be set. Use the flag --stop_threshold to stop the script based "
            "on the uncased BLEU score."))
    flags.DEFINE_string(
        name="problem", default=None,
        help=flags_core.help_wrap("problem."))
    flags.DEFINE_string(
        name="fro", default="zh",
        help=flags_core.help_wrap("problem."))
    flags.DEFINE_string(
        name="to", default="en",
        help=flags_core.help_wrap("problem."))

    flags.DEFINE_string(
        name="decode_path", default="decode.txt",
        help=flags_core.help_wrap("path to save decode result."))
    flags.DEFINE_integer(
        name="keep_checkpoint_max", default=10,
        help=flags_core.help_wrap("."))
    flags.DEFINE_integer(
        name="save_checkpoints_secs", default=100,
        help=flags_core.help_wrap(""))

    flags.DEFINE_integer(
        name="vocabulary", default=30000,
        help=flags_core.help_wrap(
            "Name of vocabulary file containing subtokens for subtokenizing the "
            "bleu_source file. This file is expected to be in the directory "
            "defined by --data_dir."))
    
    flags_core.set_defaults(data_dir="/tmp/translate_ende",
                            model_dir="/tmp/transformer_model",
                            batch_size=None,
                            train_epochs=None)

    @flags.multi_flags_validator(
        ["train_epochs", "train_steps"],
        message="Both --train_steps and --train_epochs were set. Only one may be "
                "defined.")
    def _check_train_limits(flag_dict):
        return flag_dict["train_epochs"] is None or flag_dict["train_steps"] is None

    # @flags.multi_flags_validator(
    #     ["data_dir", "bleu_source", "bleu_ref", "vocabulary",'fro','to'],
    #    message="--bleu_source, --bleu_ref, and/or --vocab_file don't exist. "
    #           "Please ensure that the file paths are correct.")

    def _check_bleu_files(flags_dict):
        """Validate files when bleu_source and bleu_ref are defined."""
        if flags_dict["bleu_source"] is None or flags_dict["bleu_ref"] is None:
            return True
        # Ensure that bleu_source, bleu_ref, and vocab files exist.
        vocab_file_path_source = os.path.join(
            flags_dict["data_dir"], "vocab_" + str(FLAGS.vocabulary) + "." + FLAGS.fro)
        vocab_file_path_target = os.path.join(
            flags_dict["data_dir"], "vocab_" + str(flags_dict["vocabulary"]) + "." + flags_dict['to'])

        return all([
            tf.gfile.Exists(flags_dict["bleu_source"]),
            tf.gfile.Exists(flags_dict["bleu_ref"]),
            tf.gfile.Exists(vocab_file_path_source),
            tf.gfile.Exists(vocab_file_path_target)])


def run_seq2seq(flags_obj):
    """Create tf.Estimator to train and evaluate transformer model.
  
    Args:
      flags_obj: Object containing parsed flag values.
    """
    # Determine training schedule based on flags.
    
    if flags_obj.train_steps is not None:
        if tf.train.latest_checkpoint(flags_obj.model_dir):
            latest_checkpoint = int(tf.train.latest_checkpoint(flags_obj.model_dir).split("-")[-1]) 
            flags_obj.train_steps = flags_obj.train_steps - latest_checkpoint
        if flags_obj.train_steps % flags_obj.steps_between_evals:
            flags_obj.steps_between_evals = flags_obj.train_steps
        train_eval_iterations = (flags_obj.train_steps // flags_obj.steps_between_evals)
        single_iteration_train_steps = flags_obj.steps_between_evals
        single_iteration_train_epochs = None
    else:
        train_epochs = flags_obj.train_epochs or DEFAULT_TRAIN_EPOCHS
        train_eval_iterations = train_epochs // flags_obj.epochs_between_evals
        single_iteration_train_steps = None
        single_iteration_train_epochs = flags_obj.epochs_between_evals

    tf.logging.info("Total train_eval_iterations %d" %train_eval_iterations)
    # Add flag-defined parameters to params object
    params = PARAMS_MAP[flags_obj.param_set]
    params.data_dir = flags_obj.data_dir
    params.num_parallel_calls = flags_obj.num_parallel_calls
    tf.logging.info(params.num_parallel_calls)
    params.epochs_between_evals = flags_obj.epochs_between_evals
    params.repeat_dataset = single_iteration_train_epochs
    params.batch_size = flags_obj.batch_size or params.batch_size
    params.train_steps = flags_obj.train_steps
    #params.steps_between_evals = single_iteration_train_steps 

    fp = open(os.path.join(flags_obj.data_dir, 'vocab.' + str(flags_obj.vocabulary) + "." + flags_obj.fro), 'r')
    lines = fp.readlines()
    params.source_vocab_size = len(lines)
    fp = open(os.path.join(flags_obj.data_dir, 'vocab.' + str(flags_obj.vocabulary) + "." + flags_obj.to), 'r')
    lines = fp.readlines()
    params.target_vocab_size = len(lines)

    # Create hooks that log information about the training and metric values
    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        tensors_to_log=TENSORS_TO_LOG,  # used for logging hooks
        batch_size=params.batch_size  # for ExamplesPerSecondHook
    )
    benchmark_logger = logger.config_benchmark_logger(flags_obj)
    benchmark_logger.log_run_info(
        model_name="seq2seq",
        dataset_name=flags_obj.problem,
        run_params=params.__dict__)

    model_idr = None
    config = None

    model_dir = flags_obj.model_dir
    if flags_obj.gpus is None:
        raise Exception("Please make sure to assign gpus for the training job")
    num_gpus = len(flags_obj.gpus.split(","))
    params.batch_size = params.batch_size // num_gpus
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    config = tf.estimator.RunConfig(keep_checkpoint_max=flags_obj.keep_checkpoint_max,
                                    save_checkpoints_secs=flags_obj.save_checkpoints_secs,
                                    train_distribute=distribution_strategy)

    # Train and evaluate

    estimator = tf.estimator.Estimator(
        model_fn=training_model_fn, model_dir=model_dir, params=params, config=config)

    train_schedule(
        estimator=estimator,
        # Training arguments
        train_eval_iterations=train_eval_iterations,
        single_iteration_train_steps=single_iteration_train_steps,
        single_iteration_train_epochs=single_iteration_train_epochs,
        train_hooks=train_hooks,
        benchmark_logger=benchmark_logger,
        # BLEU calculation arguments
        bleu_source=flags_obj.bleu_source,
        bleu_ref=flags_obj.bleu_ref,
        bleu_threshold=flags_obj.stop_threshold,
        vocab_file_path_source=os.path.join(flags_obj.data_dir,
                                            'vocab.' + str(flags_obj.vocabulary) + "." + flags_obj.fro),
        vocab_file_path_target=os.path.join(flags_obj.data_dir,
                                            'vocab.' + str(flags_obj.vocabulary) + "." + flags_obj.to),)
        #decode_to_path=os.path.join(flags_obj.model_dir, flags_obj.decode_path))


def main(_):
    run_seq2seq(flags.FLAGS)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_transformer_flags()
    absl_app.run(main)
