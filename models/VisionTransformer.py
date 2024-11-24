# -*- coding: utf-8 -*-
"""
Definition of the VisionTransformer
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as KL

class MHAttention(tf.keras.layers.Layer):

    def __init__(self,

                 d_model=65,

                 dff=128,

                 num_heads=8,

                 dropout_rate=0.1):

        super(MHAttention, self).__init__()

        self.d_model = d_model

        self.dff = dff

        self.num_heads = num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {self.d_model} should be divisible by number of heads = {self.num_heads}"
            )

        self.dropout_rate = dropout_rate

        self.attn = KL.MultiHeadAttention(self.num_heads, self.d_model // self.num_heads)

        self.attn_sum1 = KL.Add()

        self.attn_norm1 = KL.LayerNormalization(axis=-1)

        self.attn_norm2 = KL.LayerNormalization(axis=-1)

        self.Dropout1 = KL.Dropout(self.dropout_rate)

        self.Dropout2 = KL.Dropout(self.dropout_rate)

        self.attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),

                                    KL.Dropout(self.dropout_rate),

                                    KL.Dense(self.d_model),

                                    KL.Dropout(self.dropout_rate)])

    def call(self, inputs, training):

        normed_inputs = self.attn_norm1(inputs)

        attn_out = self.attn(normed_inputs, normed_inputs, return_attention_scores=False)

        attn_out = self.Dropout1(attn_out, training=training)

        out = attn_out + inputs

        mlp_out = self.attn_mlp(out)

        mlp_out = self.Dropout1(mlp_out, training=training)

        return out + mlp_out


class VisionTransformer(tf.keras.Model):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 depth=4,
                 #num_classes=10,
                 d_model=64,
                 num_heads=4,
                 dff=128,
                 latent_dim=64,
                 dropout_rate=0.1,
                 channels=3):
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.channels = channels
        self.num_patches = (image_size // self.patch_size) ** 2
        self.patch_dim = channels * self.patch_size ** 2


        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, self.num_patches + 1, self.d_model)
        )
        #self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model))
        self.patch_proj = KL.Dense(self.d_model)
        self.enc_layers = [
            MHAttention(self.d_model, self.num_heads, self.dff, self.dropout_rate)
            for _ in range(self.depth)
        ]

        self.mlp = Sequential([KL.LayerNormalization(axis=-1),
                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),
                    KL.Dropout(self.dropout_rate),
                    KL.Dense(self.latent_dim)])

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        #batch_size = tf.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        #class_emb = tf.broadcast_to(
        #    self.class_emb, [batch_size, 1, self.d_model]
        #)
        #x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)
        x = self.mlp_head(x)

        return x



