import numpy as np
import tensorflow as tf

import functions
from functions import make_keys_grid

class ImagesHelper:

    @staticmethod
    def get_sai_and_epi(query_scene, scene_shape):
        get_prediction = lambda keys_ds: tf.concat([query_scene(k) for k in keys_ds], 0)  # noqa: E731

        def get_keys_ds(keys):
            return tf.data.Dataset.from_tensor_slices(keys).batch(2**12).prefetch(tf.data.experimental.AUTOTUNE)
        h, _, v, u, _ = scene_shape.numpy() // 2
        keys_ranges = scene_shape[:-1].numpy()
        keys_dim = tf.size(keys_ranges)
        grid = make_keys_grid(keys_ranges)
        sai_keys = tf.reshape(grid[:, :, v, u], [-1, keys_dim])
        sai_keys_ds = get_keys_ds(sai_keys)
        sai_shape = tf.gather(scene_shape, [0, 1, 4])
        sai_pred = tf.reshape(get_prediction(sai_keys_ds), sai_shape)
        epi_keys = tf.reshape(grid[h, :, v, :], [-1, keys_dim])
        epi_keys_ds = get_keys_ds(epi_keys)
        epi_shape = tf.gather(scene_shape, [1, 3, 4])
        epi_pred = tf.reshape(get_prediction(epi_keys_ds), epi_shape)
        epi_pred = tf.transpose(epi_pred, [1, 0, 2])
        return sai_pred, epi_pred

class MultipleScenesImagesCallbackFactory:

    def __init__(self, tensorboard_path, model, ds_factory, max_scenes_to_log=16):
        self.tensorboard_path = tensorboard_path
        self.model = model
        self.ds_factory = ds_factory
        self.max_scenes_to_log = max_scenes_to_log

    def make(self, is_adadrop=False):
        def save_sai_and_epi(epoch, _):
            for j in self.scene_indices:
                query_scene = (lambda k: self.model([j, k])[0]) if is_adadrop else (lambda k: self.model([j, k]))
                sai, epi = ImagesHelper.get_sai_and_epi(query_scene, self.ds_factory.scene_shapes[j])
                scene_name = self.ds_factory.scene_names[j]
                h, _, v, u, _ = self.ds_factory.scene_shapes[j].numpy() // 2
                with file_writer_images.as_default():
                    tf.summary.image(f"({scene_name}) SAI [(v, u) = ({v}, {u})]", sai[tf.newaxis], step=epoch)
                    tf.summary.image(f"({scene_name}) EPI [(h, v) = ({h}, {v})]", epi[tf.newaxis], step=epoch)
        file_writer_images = tf.summary.create_file_writer(self.tensorboard_path + "/img")
        return tf.keras.callbacks.LambdaCallback(on_epoch_end=save_sai_and_epi)

    @property
    def scene_indices(self):
        if self.ds_factory.num_scenes <= self.max_scenes_to_log:
            return range(self.ds_factory.num_scenes)
        np.random.seed(0)
        indices = np.random.permutation(self.ds_factory.num_scenes)[:self.max_scenes_to_log]
        return [int(j) for j in indices]

class SingleSceneImagesCallbackFactory:

    def __init__(self, tensorboard_path, model, ds_factory):
        self.tensorboard_path = tensorboard_path
        self.model = model
        self.ds_factory = ds_factory

    def make(self):
        def save_sai_and_epi(epoch, _):
            sai, epi = ImagesHelper.get_sai_and_epi(self.model, self.ds_factory.data_shape)
            h, _, v, u, _ = self.ds_factory.data_shape.numpy() // 2
            with file_writer_images.as_default():
                tf.summary.image(f"SAI [(v, u) = ({v}, {u})]", sai[tf.newaxis], step=epoch)
                tf.summary.image(f"(EPI [(h, v) = ({h}, {v})]", epi[tf.newaxis], step=epoch)
        file_writer_images = tf.summary.create_file_writer(self.tensorboard_path + "/img")
        return tf.keras.callbacks.LambdaCallback(on_epoch_end=save_sai_and_epi)
