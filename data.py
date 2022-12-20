import glob
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

import functions
from functions import make_keys_grid

class SingleSceneFactory:
    def __init__(self, path):
        self.path = path

    def make(self):
        paths = glob.glob(str(Path(self.path) / "*"))
        paths = sorted(paths)
        paths = [p for p in paths if Path(p).suffix == ".png"]
        ang_res = int(math.sqrt(len(paths)))
        sais = [SingleSceneFactory._get_image(path) for path in paths]
        data = tf.concat(sais, axis=2)
        shape = tf.concat([tf.shape(data)[:2], [ang_res, ang_res, 3]], -1)
        return tf.reshape(data, shape)

    @staticmethod
    @tf.function
    def _get_image(path):
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=3)
        return tf.image.convert_image_dtype(image, tf.float32)

class SingleSceneDatasetFactory:

    def __init__(self, data):
        self.data = data

    @property
    def data_shape(self):
        return tf.shape(self.data)

    @property
    def keys_ranges(self):
        return self.data_shape[:-1]

    @property
    def num_data_points(self):
        return tf.reduce_prod(self.keys_ranges).numpy()

    @property
    def keys_dim(self):
        return tf.size(self.keys_ranges)

    @property
    def values_dim(self):
        return self.data_shape[-1]

    def make(self, shuffle=True):
        keys = tf.reshape(make_keys_grid(self.keys_ranges), [-1, self.keys_dim])
        values = tf.reshape(self.data, [-1, self.values_dim])
        if shuffle:
            tf.random.set_seed(0)
            keys = tf.random.shuffle(keys, seed=0)
            values = tf.random.shuffle(values, seed=0)
        keys_ds = tf.data.Dataset.from_tensor_slices(keys)
        values_ds = tf.data.Dataset.from_tensor_slices(values)
        return tf.data.Dataset.zip((keys_ds, values_ds))

class MulipleScenesDatasetFactory:

    def __init__(self, scenes_directory, cache_directory, verbose=False, take_random_scenes_up_to=None):
        self.scenes_directory = scenes_directory
        self.cache_directory = cache_directory
        self.verbose = verbose
        self.scene_paths = sorted([Path(s) for s in glob.glob(str(Path(scenes_directory) / "*")) if os.path.isdir(s)])
        self.scene_names = [p.stem for p in self.scene_paths]
        self.scene_shapes = [MulipleScenesDatasetFactory._get_scene_shape(p) for p in self.scene_paths]
        self.absolute_num_scenes = len(self.scene_paths)
        if take_random_scenes_up_to is None:
            self.num_scenes = self.absolute_num_scenes
        else:
            self.num_scenes = min(self.absolute_num_scenes, take_random_scenes_up_to)
        np.random.seed(0)
        self.absolute_indices_of_taken_scenes = np.random.choice(
            np.arange(self.absolute_num_scenes), self.num_scenes, replace=False)

    @property
    def max_keys_ranges(self):
        return tf.reduce_max(tf.stack(self.scene_shapes, 0), 0)

    def make(self, batch_size):
        self._cache_shuffled_data()

        def get_np_scene_from_name(scene_name):
            data_pattern = f"{self.cache_directory}/{Path(self.scenes_directory).name}/{scene_name}/*/*"
            filenames = sorted(glob.glob(data_pattern))
            return [np.load(filename, mmap_mode='r') for filename in filenames]
        scenes = [get_np_scene_from_name(n) for n in self.scene_names]
        get_generator = lambda: self._generate_dataset(scenes, batch_size)  # noqa: E731
        signature = (
            (tf.TensorSpec(shape=(None), dtype=tf.int32),
             tf.TensorSpec(shape=(None, 4), dtype=tf.int32)),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        return tf.data.Dataset.from_generator(get_generator, output_signature=signature)

    def _cache_shuffled_data(self):
        for shuffle_seed, scene_path in enumerate(self.scene_paths):
            scene_name = scene_path.stem
            scene_shape = MulipleScenesDatasetFactory._get_scene_shape(scene_path)
            info_string = f"{shuffle_seed:02d}-" + "-".join([f"{dim:04d}" for dim in list(scene_shape.numpy())])

            for data_type in ['values', 'keys']:
                data_path = (f"{self.cache_directory}/{Path(self.scenes_directory).name}/"
                             f"{scene_name}/{info_string}/{data_type}.npy")
                if Path(data_path).exists():
                    if self.verbose:
                        print(f"Shuffled array for `{scene_name}` {data_type} already exists in cache.")
                    continue
                print(f"Shuffling scene's {data_type}: `{scene_name}`")
                if data_type == 'values':
                    scene = SingleSceneFactory(scene_path).make()
                    value_dim = scene_shape[-1]
                    data = tf.reshape(scene, [-1, value_dim])
                else:
                    keys_shape = scene_shape[:-1]
                    key_dim = tf.size(keys_shape)
                    data = tf.reshape(make_keys_grid(keys_shape), [-1, key_dim])
                tf.random.set_seed(shuffle_seed)
                data = tf.random.shuffle(data, seed=shuffle_seed)
                Path(data_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(data_path, data)

    @staticmethod
    def _get_scene_shape(scene_path):
        image_paths = [s for s in sorted(glob.glob(str(scene_path / "*"))) if Path(s).suffix == ".png"]
        im_shape = SingleSceneFactory._get_image(image_paths[0]).shape
        angular_res = int(math.sqrt(len(image_paths)))
        return tf.gather(tf.concat([im_shape, 2 * [angular_res]], -1), [0, 1, 3, 4, 2])

    def _generate_dataset(self, scenes, batch_size):
        scene_sizes = [k.shape[0] for k, _ in scenes]
        batch_indices = [0 for _ in range(self.num_scenes)]
        scene_index = 0
        while True:
            absolute_scene_index = self.absolute_indices_of_taken_scenes[scene_index]
            keys, values = scenes[absolute_scene_index]
            batch_index = batch_indices[scene_index]
            scene_size = scene_sizes[absolute_scene_index]
            start = batch_index * batch_size
            batch_index += 1
            end = min(batch_index * batch_size, scene_size)
            scene_index_, k, v = [tf.convert_to_tensor(x) for x in [scene_index, keys[start:end], values[start:end]]]
            yield (scene_index_, k), v
            batch_indices[scene_index] = 0 if end == scene_size else batch_index
            scene_index = (scene_index + 1) % self.num_scenes

class MacroPixelImage8BitsDatasetFactory:

    def __init__(
        self,
        scenes_directory,
        cache_directory,
        limit_number_of_scenes=None
    ):
        self.scenes_directory = scenes_directory
        self.cache_directory = cache_directory
        self._limit_number_of_scenes = limit_number_of_scenes

        self._all_scene_paths = MacroPixelImage8BitsDatasetFactory._get_scene_paths(scenes_directory)

    @property
    def data_types(self):
        return ['keys_float32', 'values_uint8']

    @property
    def scene_shape(self):
        # TODO: remove hard-coded values
        return tf.constant([375, 540, 8, 8, 3])

    @property
    def scene_names(self):
        return [x.stem for x in self.scene_paths]

    @property
    def scene_shapes(self):
        return self.num_scenes * [self.scene_shape]

    @property
    def scene_size(self):
        return tf.reduce_prod(self.scene_shape[:-1]).numpy()

    @property
    def scene_paths(self):
        return self._all_scene_paths[:self._limit_number_of_scenes]

    @property
    def num_scenes(self):
        return len(self.scene_paths)

    @property
    def max_keys_ranges(self):
        return self.scene_shape

    @staticmethod
    def _get_scene_paths(scenes_directory):
        filenames = [x for x in glob.glob(f"{scenes_directory}/*")]
        extension = '.png'
        return sorted([Path(x) for x in filenames if x[-len(extension):] == extension])

    def make(self, batch_size):
        get_generator = lambda: self._generate_dataset(batch_size)  # noqa: E731
        signature = (
            (tf.TensorSpec(shape=(None), dtype=tf.int32),
             tf.TensorSpec(shape=(None, 4), dtype=tf.int32)),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
        return tf.data.Dataset.from_generator(
            get_generator,
            output_signature=signature)

    def _get_data_path(self, scene_name, data_type):
        return f"{self.cache_directory}/{Path(self.scenes_directory).name}/{scene_name}/{data_type}.npy"

    def _get_kv_pair(self, scene_path):
        return list(self._generate_kv(scene_path, hash(scene_path)))

    def _generate_kv(self, scene_path, shuffle_seed=0):

        for data_type in self.data_types:
            data_path = self._get_data_path(scene_path.stem, data_type)
            if not Path(data_path).exists():
                print(f"Shuffling scene's {data_type}: `{scene_path.stem}`")

                # keys
                if data_type == self.data_types[0]:
                    keys_shape = self.scene_shape[:-1]
                    key_dim = tf.size(keys_shape)
                    data = tf.reshape(make_keys_grid(keys_shape), [-1, key_dim])
                # values
                elif data_type == self.data_types[1]:
                    scene = self._get_scene(str(scene_path))
                    value_dim = self.scene_shape[-1]
                    data = tf.reshape(scene, [-1, value_dim])

                tf.random.set_seed(shuffle_seed)
                data = tf.random.shuffle(data, seed=shuffle_seed)
                Path(data_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(data_path, data)

            yield np.load(data_path, mmap_mode='r')

    def _get_scene(self, scene_path_as_str):
        # TODO: remove hard-coded values
        value = tf.io.read_file(scene_path_as_str)
        lf = tf.image.decode_image(value, channels=3)
        lf = lf[:self.scene_shape[0] * 14, :self.scene_shape[1] * 14]
        lf = tf.transpose(tf.reshape(lf, [self.scene_shape[0], 14, self.scene_shape[1], 14, 3]), [0, 2, 1, 3, 4])
        lf = lf[
            :, :,
            (14 // 2) - (self.scene_shape[2] // 2):(14 // 2) + (self.scene_shape[2] // 2),
            (14 // 2) - (self.scene_shape[3] // 2):(14 // 2) + (self.scene_shape[3] // 2), :]
        return lf

    def _generate_dataset(self, batch_size):
        kv_pairs = [self._get_kv_pair(path) for path in self.scene_paths]
        batch_indices = [0 for _ in range(len(kv_pairs))]
        while True:
            scene_index = random.randint(0, len(kv_pairs) - 1)
            keys, values_as_uint8 = kv_pairs[scene_index]
            batch_index = batch_indices[scene_index]
            start = batch_index * batch_size
            batch_index += 1
            end = min(batch_index * batch_size, self.scene_size)
            scene_index_, k, v = [tf.convert_to_tensor(x) for x in [scene_index, keys[start:end], values_as_uint8[start:end]]]  # noqa: E501
            yield (scene_index_, k), (tf.cast(v, tf.float32) / 255.) ** .5
            batch_indices[scene_index] = 0 if end == self.scene_size else batch_index

    def _generate_scene_keys_values(self, batch_size, batch_offset=0):
        kv_pairs = [self._get_kv_pair(path) for path in self.scene_paths]
        batches_per_scene = int(math.ceil(self.scene_size / batch_size))

        for batch_index in range(batch_offset, batches_per_scene):
            for scene_index, (keys, values) in enumerate(kv_pairs):
                start = batch_index * batch_size
                end = min(start + batch_size, self.scene_size)
                yield np.array(scene_index), keys[start:end], values[start:end]
