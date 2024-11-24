# -*- coding: utf-8 -*-
import tensorflow_datasets as tfds
import tensorflow as tf


def load_data(data_name, pretraining_config, finetune_config):
    print('---' * 30)
    print('PREPARING DATA...')

    if data_name == "imagenet32x32":
        ds, info = tfds.load("imagenet_resized/32x32", as_supervised=True, with_info=True)
        dataset = (
            ds["train"]
            .cache()
            .shuffle(5 * pretraining_config.BATCH_SIZE_NNCLR)
            .batch(pretraining_config.BATCH_SIZE_NNCLR)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        dataset_test = (
            ds["validation"]
            .cache()
            .batch(pretraining_config.BATCH_SIZE_NNCLR)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        pretraining_config.CLASSIFICATION_AUGMENTER["input_shape"] = info.features['image'].shape
        pretraining_config.CONTRASTIVE_AUGMENTER["input_shape"] = info.features['image'].shape
        finetune_config.CLASSIFICATION_AUGMENTER["input_shape"] = info.features['image'].shape
        pretraining_config.PSEUDO_N_CLUSTERS = info.features['label'].num_classes
    elif data_name == "mnist":
        ds, info = tfds.load("mnist", as_supervised=True, with_info=True)
        dataset = (
            ds["train"]
            .cache()
            .shuffle(5 * pretraining_config.BATCH_SIZE_NNCLR)
            .batch(pretraining_config.BATCH_SIZE_NNCLR)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        dataset_test = (
            ds["validation"]
            .cache()
            .batch(pretraining_config.BATCH_SIZE_NNCLR)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        pretraining_config.CLASSIFICATION_AUGMENTER["input_shape"] = info.features['image'].shape
        pretraining_config.CONTRASTIVE_AUGMENTER["input_shape"] = info.features['image'].shape
        finetune_config.CLASSIFICATION_AUGMENTER["input_shape"] = info.features['image'].shape
        pretraining_config.PSEUDO_N_CLUSTERS = info.features['label'].num_classes

    return dataset, dataset_test
