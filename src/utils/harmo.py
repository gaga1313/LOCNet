import tensorflow as tf
from harmonization.common.clickme_dataset import FEATURE_DESCRIPTION, GAUSSIAN_KERNEL, AUTO, gaussian_blur, \
    CLICKME_BASE_URL, NB_VAL_SHARDS
import json


PATH_TO_JSON = '../../data/human_identifable_category_info.json'

# Load the JSON file
with open(PATH_TO_JSON, 'r') as file:
    class_info = json.load(file)

# Create a dictionary to map original indices to new class labels
idx_to_new_idx = {}
for new_idx, info in enumerate(class_info.values()):
    for idx in info['IN_indices']:
        idx_to_new_idx[int(idx)] = new_idx


def parse_clickme_prototype(prototype):
    data = tf.io.parse_single_example(prototype, FEATURE_DESCRIPTION)

    image = tf.io.decode_jpeg(data['image'])
    image = tf.reshape(image, (256, 256, 3))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224), method='bilinear')

    heatmap = tf.io.decode_jpeg(data['heatmap'])
    heatmap = tf.reshape(heatmap, (256, 256, 1))
    heatmap = tf.cast(heatmap, tf.float32)
    heatmap = tf.image.resize(heatmap, (64, 64), method="bilinear")
    heatmap = gaussian_blur(heatmap, GAUSSIAN_KERNEL)
    heatmap = tf.image.resize(heatmap, (224, 224), method="bilinear")

    original_label = tf.cast(data['label'], tf.int32)

    # Check if the label is in the new class indices and map it
    new_label = tf.py_function(lambda x: idx_to_new_idx.get(x.numpy(), -1), [original_label], Tout=tf.int32)

    # Filter out unwanted labels
    is_valid_label = tf.not_equal(new_label, -1)

    # Convert the label to one-hot encoding
    new_label = tf.one_hot(new_label, len(class_info))

    return image, heatmap, new_label, is_valid_label


def custom_load_clickme(shards_paths, batch_size):
    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(shards_paths, num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order)

    dataset = dataset.map(parse_clickme_prototype, num_parallel_calls=AUTO)

    # Filter samples based on valid labels
    dataset = dataset.filter(lambda image, heatmap, label, is_valid: is_valid)

    # Drop the is_valid flag from the dataset
    dataset = dataset.map(lambda image, heatmap, label, _: (image, heatmap, label))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset


def custom_load_clickme_val(batch_size=64):
    """
    Loads the click-me validation set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me validation dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """

    shards_paths = [
        tf.keras.utils.get_file(f"clickme_val_{i}",
                                f"{CLICKME_BASE_URL}/val/val-{i}.tfrecords",
                                cache_subdir="datasets/click-me") for i in range(NB_VAL_SHARDS)
    ]

    return custom_load_clickme(shards_paths, batch_size)


if __name__ == '__main__':
    dataset = custom_load_clickme_val(batch_size=128)
    # from harmonization.common import load_clickme_val
    # dataset_orig = load_clickme_val(batch_size=128)
    # print(1)
