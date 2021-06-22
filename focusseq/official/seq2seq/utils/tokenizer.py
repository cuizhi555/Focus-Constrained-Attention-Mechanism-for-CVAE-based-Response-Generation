# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines Subtokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import sys
import unicodedata

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
UNK = "UNK"
UNK_ID = 2
RESERVED_TOKENS = [PAD, EOS, UNK]

# min_count is the minimum number of times a subtoken must appear in the data
# before before it is added to the vocabulary. The value is found using binary
# search to obtain the target vocabulary size.
_MIN_MIN_COUNT = 1  # min value to use when binary searching for min_count
_MAX_MIN_COUNT = 1000  # max value to use when binary searching for min_count


class Subtokenizer(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self, vocab_file, reserved_tokens=None):
        """Initializes class, creating a vocab file if data_files is provided."""
        tf.logging.info("Initializing Subtokenizer from file %s." % vocab_file)

        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS

        self.subtoken_list = _load_vocab_file(vocab_file, reserved_tokens)
        self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)

    #   self.max_subtoken_length = 0
    #    for subtoken in self.subtoken_list:
    #      self.max_subtoken_length = max(self.max_subtoken_length, len(subtoken)
    @staticmethod
    def init_from_files(
            vocab_file, target_vocab_size, reserved_tokens=None):
        """Create subtoken vocabulary based on files, and save vocab to file.

        Args:
          vocab_file: String name of vocab file to store subtoken vocabulary.
          target_vocab_size: target vocabulary size to generate.
          reserved_tokens: List of string tokens that are guaranteed to be at the
            beginning of the subtoken vocabulary list.

        Returns:
          Subtokenizer object
        """
        if reserved_tokens is None:
            reserved_tokens = RESERVED_TOKENS
        if tf.gfile.Exists(vocab_file):
            tf.logging.info("Vocab file already exists (%s)" % vocab_file)
            with open(vocab_file, 'r')as fp:
                subtoken_list = fp.readlines()
            subtoken_list = [line.strip() for line in subtoken_list]
            subtoken_list = [word  for word in subtoken_list if word not in RESERVED_TOKENS]
            subtoken_list = RESERVED_TOKENS + subtoken_list
            subtoken_list = [_native_to_unicode(word) for word in subtoken_list]
            _save_vocab_file(vocab_file, subtoken_list)
        else:
            raise OSError("Vocabulary does not exist.")
        #     tf.logging.info("No vocabulary and nedd add vocabulary to data_dir.")
        return Subtokenizer(vocab_file)

    def encode(self, raw_string, add_eos=False):
        """Encodes a string into a list of int subtoken ids."""
        ret = []
        tokens = raw_string.strip().split()
        for token in tokens:
            ret.extend(self._token_to_ids(_native_to_unicode(token)))
        if add_eos:
            ret.append(EOS_ID)
        return ret

    def _token_to_ids(self, token):
        """Encode a single token into a list of subtoken ids."""
        ret = _split_token_to_subtokens(token, self.subtoken_to_id_dict)
        ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]
        return ret

    def decode(self, subtokens):
        """Converts list of int subtokens ids into a string."""
        if isinstance(subtokens, np.ndarray):
            # Note that list(subtokens) converts subtokens to a python list, but the
            # items remain as np.int32. This converts both the array and its items.
            subtokens = subtokens.tolist()
        if not subtokens:
            return ""

        assert isinstance(subtokens, list) and isinstance(subtokens[0], int), (
            "Subtokens argument passed into decode() must be a list of integers.")
        return _unicode_to_native(
            _join_tokens_to_string(self._ids_to_tokens(subtokens)))

    def _ids_to_tokens(self, subtokens):
        """Convert list of int subtoken ids to a list of string tokens."""
        ret = []
        for s in subtokens:
            ret.append(self.subtoken_list[s])
        return ret


def _save_vocab_file(vocab_file, subtoken_list):
    """Save subtokens to file."""
    with tf.gfile.Open(vocab_file, mode="w") as f:
        for subtoken in subtoken_list:
            f.write("%s\n" % subtoken)


def _load_vocab_file(vocab_file, reserved_tokens=None):
    """Load vocabulary while ensuring reserved tokens are at the top."""
    if reserved_tokens is None:
        reserved_tokens = RESERVED_TOKENS
    with tf.gfile.Open(vocab_file, mode="r") as f:
        subtoken_list = f.readlines()
    reserved_tokens = [_native_to_unicode(word) for word in reserved_tokens]
    subtoken_list = [_native_to_unicode(word.strip()) for word in subtoken_list ]
    subtoken_list =[word for word in subtoken_list  if word not in reserved_tokens]
    tf.logging.info(len(subtoken_list))
    subtoken_list = reserved_tokens +subtoken_list
    tf.logging.info(len(subtoken_list))

    return subtoken_list


def _native_to_unicode(s):
    """Convert string to unicode (required in Python 2)."""
    if six.PY2:
        return s if isinstance(s, unicode) else s.decode("utf-8")
    else:
        return s


def _unicode_to_native(s):
    """Convert string from unicode to native format (required in Python 2)."""
    if six.PY2:
        return s.encode("utf-8") if isinstance(s, unicode) else s
    else:
        return s


def _split_string_to_tokens(text):
    """Splits text to a list of string tokens."""
    if not text:
        return []
    ret = []
    ret = text.strip().split(" ")
    return ret


def _join_tokens_to_string(tokens):
    """Join a list of string tokens into a single string."""
    ret = tokens
    return " ".join(ret)


def _list_to_index_dict(lst):
    """Create dictionary mapping list items to their indices in the list."""
    return {item: n for n, item in enumerate(lst)}


def _split_token_to_subtokens(token, subtoken_dict):
    """Splits a token into subtokens defined in the subtoken dict."""
    ret = []
    if token in subtoken_dict:
        ret.append(token)
    else:
        ret.append(_native_to_unicode(UNK))
    return ret
