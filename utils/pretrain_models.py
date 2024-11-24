import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from os import path
from utils.evaluation import *
from wandb.keras import WandbMetricsLogger


def check_filepath_naming(filepath):
    """
    Check and modify the file path to avoid overwriting existing files.

    If the specified file path exists, a number is appended to the file name to
    create a unique file name.

    Arguments:
        filepath: The file path to check.

    Returns:
        A modified file path if the original exists, otherwise the original file path.
    """
    if path.exists(filepath):
        numb = 1
        while True:
            new_path = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(new_path):
                numb += 1
            else:
                return new_path
    return filepath

class RandomResizedCrop(tf.keras.layers.Layer):
    def __init__(self, scale, ratio):
        super().__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
        random_ratios = tf.exp(
            tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
        )

        new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
        height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
        width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images, bounding_boxes, tf.range(batch_size), (height, width)
        )
        return images

class RandomBrightness(tf.keras.layers.Layer):
    def __init__(self, brightness):
        super().__init__()
        self.brightness = brightness

    def blend(self, images_1, images_2, ratios):
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def call(self, images):
        images = self.random_brightness(images)
        return images

def augmenter(brightness, name, scale, input_shape):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Rescaling(1 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=scale, ratio=(3 / 4, 4 / 3)),
            RandomBrightness(brightness=brightness),
        ],
        name=name,
    )
class NNCLR(tf.keras.Model):
    """
    NNCLR (Neural Network Contrastive Learning Representation) is a custom Keras Model for semi-supervised learning.

    This class implements a NNCLR model that uses contrastive learning for feature extraction and a linear probe
    for classification. It is designed for scenarios where labeled data is scarce, and the model needs to learn
    useful representations from unlabeled data. The model consists of an encoder, a projection head for contrastive
    learning, and a linear probe for classification.

    Attributes:
    model_config: Configuration object containing model-specific parameters.
    pretraining_config: Configuration object containing pretraining-specific parameters.
    encoder: The neural network used to encode the input data.
    contrastive_augmenter: A function or layer that applies augmentation for contrastive learning.
    classification_augmenter: A function or layer that applies augmentation for classification.
    probe_accuracy, correlation_accuracy, contrastive_accuracy: Keras metrics for tracking training performance.
    probe_loss: Loss function for the probe classifier.
    projection_head: A Keras Sequential model used in the projection of encoded features.
    linear_probe: A Keras Sequential model for classification.
    temperature: The temperature parameter used in contrastive loss calculation.
    feature_queue: A queue of features used in contrastive learning for negative sampling.

    Methods:
    compile(contrastive_optimizer, probe_optimizer, **kwargs): Compiles the NNCLR model with given optimizers.
    nearest_neighbour(projections): Computes nearest neighbours in the feature space for contrastive learning.
    update_contrastive_accuracy(features_1, features_2): Updates the contrastive accuracy metric.
    update_correlation_accuracy(features_1, features_2): Updates the correlation accuracy metric.
    contrastive_loss(projections_1, projections_2): Computes the contrastive loss between two sets of projections.
    train_step(data): Defines a single step of training.
    test_step(data): Defines a single step of testing.

    The NNCLR model uses contrastive learning to learn representations by maximizing agreement between differently
    augmented views of the same data instance and uses a linear classifier (probe) to perform classification based on
    these representations.
    """
    def __init__(self, model_config, pretraining_config, encoder, contrastive_augmenter, classification_augmenter):
        super().__init__()
        self.probe_optimizer = None
        self.contrastive_optimizer = None
        self.model_config = model_config
        self.pretraining_config = pretraining_config
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.projection_width = self.pretraining_config.PROJECTION_WIDTH
        self.contrastive_augmenter = augmenter(**contrastive_augmenter)
        self.classification_augmenter = augmenter(**classification_augmenter)
        self.encoder = encoder
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.model_config.LATENT_LEN,)),
                tf.keras.layers.Dense(self.projection_width, activation="relu"),
                tf.keras.layers.Dense(self.projection_width),
            ],
            name="projection_head",
        )
        self.temperature = self.pretraining_config.TEMPERATURE

        feature_dimensions = self.model_config.LATENT_LEN
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(self.pretraining_config.QUEUE_SIZE, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, images):
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        res = {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
        }

        return res

    def test_step(self, images):
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        features_1 = self.encoder(augmented_images_1)
        features_2 = self.encoder(augmented_images_2)
        projections_1 = self.projection_head(features_1)
        projections_2 = self.projection_head(features_2)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)


        return {"c_loss_test": contrastive_loss}


def pretrain_model(model, model_config, pretraining_config, dataset, dataset_test, save_weights,
                   save_dir):
    """
    Pretrains a model using a specified method and configuration on given datasets.

    This function supports different pretraining methods, particularly 'reconstruction' and 'NNCLR' (Neural Network
    Contrastive Learning Representation). It handles the entire pretraining process, including setting up the optimizer,
    loss functions, and learning rate schedules. It also provides support for early stopping and logs training metrics
    using wandb.

    Parameters:
    model (tf.keras.Model): The neural network model to be pretrained.
    model_config: Configuration object containing model-specific parameters.
    pretraining_config: Configuration object containing pretraining-specific parameters.
    pretrain_method (str): The method used for pretraining ('reconstruction', 'NNCLR', etc.).
    dataset (tf.data.Dataset): The training dataset.
    dataset_test (tf.data.Dataset): The test dataset.
    save_weights (bool): Whether to save the pretrained weights.
    save_dir (str): Directory where the pretrained weights should be saved.

    The function first checks the pretraining method. For 'reconstruction', it uses a Mean Squared Error loss to train
    the model to reconstruct its inputs. For 'NNCLR', it sets up a NNCLR model and trains it using contrastive learning.
    The function supports early stopping based on validation loss and learning rate scheduling. After training, it saves
    the model weights if 'save_weights' is True.

    Returns:
    str: The path where the pretrained model's weights are saved.
    """


    nnclr = NNCLR(model_config=model_config,
                  pretraining_config=pretraining_config,
                  encoder=model,
                  contrastive_augmenter=pretraining_config.CONTRASTIVE_AUGMENTER,
                  classification_augmenter=pretraining_config.CLASSIFICATION_AUGMENTER)
    nnclr.compile(contrastive_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.
                                                                 LEARNING_RATE_NNCLR),
                  probe_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.LEARNING_RATE_NNCLR))

    nnclr.fit(dataset,
              epochs=pretraining_config.NUM_EPOCHS_NNCLR,
              validation_data=dataset_test,
              verbose=1,
              callbacks=[WandbMetricsLogger()])

    if save_weights:  # add numbering system if file already exists

        if "Pretrain_" in save_dir:
            # Split the path at 'Pretrain_' and take the second part
            pseudo = tf.keras.Sequential([nnclr.encoder, nnclr.projection_head])
            for step, batch in enumerate(dataset):
                batch_t = batch[0][0]
                break
            pseudo.predict(batch_t)


            save_pseudo = check_filepath_naming(save_dir)
            pseudo.save_weights(save_pseudo)
            wandb.log({"Final save name": save_pseudo})

    return save_pseudo
