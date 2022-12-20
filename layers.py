import math
import tensorflow as tf

class FourierFeaturesInitializer(tf.keras.initializers.Initializer):

    def __init__(self, max_frequencies):
        self.max_frequencies = max_frequencies

    def __call__(self, shape, dtype=None):
        assert shape[0] == len(self.max_frequencies)
        x = [
            tf.random.uniform(
                shape[1:],
                minval=-max_frequency,
                maxval=+max_frequency
            )
            for max_frequency in self.max_frequencies
        ]
        return tf.stack(x, 0)

class FourierFeatures(tf.keras.layers.Layer):

    def __init__(self, max_frequencies, encoding_dim, keys_ranges, **kwargs):
        super().__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.keys_dim = len(max_frequencies)
        self.W = tf.Variable(FourierFeaturesInitializer(max_frequencies)([self.keys_dim, encoding_dim]))
        self.b = tf.Variable(tf.random.uniform([1, encoding_dim], maxval=math.pi * 2))
        self.offset = (tf.cast(keys_ranges, tf.float32) - 1) / 2

    def call(self, x):
        x = x - self.offset
        return tf.sin(x @ self.W + self.b)

class FourierFeaturesProvider(tf.keras.layers.Layer):

    def __init__(self, number_of_scenes, max_frequencies, encoding_dim, coordinates_range, **kwargs):
        super().__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.coordinates_dim = len(max_frequencies)
        self._weight_matrix_per_scene = tf.Variable(
            tf.stack([
                FourierFeaturesInitializer(max_frequencies)([self.coordinates_dim, encoding_dim])
                for _ in range(number_of_scenes)], 0)
        )
        self._bias_vector_per_scene = tf.Variable(
            tf.random.uniform([number_of_scenes, encoding_dim], maxval=math.pi * 2)
        )
        self._coordinates_offset = (tf.cast(coordinates_range, tf.float32) - 1) / 2

    def _get_weight_matrix(self, scene_id):
        slice_begin = tf.concat([scene_id, [0, 0]], 0)
        slice_size = tf.constant([1, self.coordinates_dim, self.encoding_dim])
        return tf.slice(self._weight_matrix_per_scene, slice_begin, slice_size)[0]

    def _get_bias_vector(self, scene_id):
        slice_begin = tf.concat([scene_id, [0]], 0)
        slice_size = tf.constant([1, self.encoding_dim])
        return tf.slice(self._bias_vector_per_scene, slice_begin, slice_size)

    def call(self, inputs):
        scene_id, coordinates = inputs[0], inputs[1]
        scene_id = tf.reshape(scene_id, [1])
        weight_matrix = self._get_weight_matrix(scene_id)
        bias = self._get_bias_vector(scene_id)
        coordinates = coordinates - self._coordinates_offset
        return tf.sin(coordinates @ weight_matrix + bias)


class ModelLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        residual=True,
        normalized=True,
        activation=tf.math.sin,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.residual = residual
        self.normalized = normalized
        self.activation = activation
        if self.normalized:
            self.LN = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, inputs):
        x, weight_matrix, bias_vector = inputs
        y = x @ weight_matrix + bias_vector
        y = self.activation(y)
        if self.residual:
            y = x + y
        if self.normalized:
            y = self.LN(y)
        return y

class LayerProvider(tf.keras.layers.Layer):

    def __init__(self, fan_in, fan_out, rank, number_of_scenes, omega_0, **kwargs):
        super().__init__(**kwargs)
        self.fan_out = fan_out
        self.rank = rank
        self.omega_0 = omega_0
        self._dictionary_in = self.add_weight(
            'dictionary_in', shape=[fan_in, rank], initializer=tf.keras.initializers.Orthogonal()
        )
        self._dictionary_out = self.add_weight(
            'dictionary_out', shape=[rank, fan_out], initializer=tf.keras.initializers.Orthogonal()
        )
        self._coefficient_vector_per_scene = self.add_weight(
            'log_coefficient_vector_per_scene', shape=[number_of_scenes, rank],
            initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=tf.sqrt(6.) / omega_0)
        )
        self._bias_vector_per_scene = self.add_weight(
            'bias_vector_per_scene', shape=[number_of_scenes, fan_out],
            initializer=tf.keras.initializers.Zeros()
        )

    def _get_weight_matrix(self, scene):
        coefficient_vector = self._get_coefficient_vector(scene)
        return self.omega_0 * (self._dictionary_in * coefficient_vector) @ self._dictionary_out

    def _get_coefficient_vector(self, scene_id):
        slice_begin = tf.concat([scene_id, [0]], 0)
        slice_size = [1, self.rank]
        return tf.slice(self._coefficient_vector_per_scene, slice_begin, slice_size)

    def _get_bias_vector(self, scene_id):
        slice_begin = tf.concat([scene_id, [0]], 0)
        slice_size = [1, self.fan_out]
        return tf.slice(self._bias_vector_per_scene, slice_begin, slice_size)

    @tf.function
    def call(self, scene_id):
        return self._get_weight_matrix(scene_id), self._get_bias_vector(scene_id)

class SVDSiren(tf.keras.layers.Layer):

    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        hidden_dim,
        rank,
        number_of_scenes,
        omega_0,
        **kwargs
    ):
        super().__init__(**kwargs)
        activations = (depth - 1) * [tf.sin] + [tf.sigmoid]
        are_normalizing = (depth - 1) * [True] + [False]
        are_residual = [False] + (depth - 2) * [True] + [False]
        fans_in = [input_dim] + (depth - 1) * [hidden_dim]
        fans_out = (depth - 1) * [hidden_dim] + [output_dim]
        self.layers = [
            ModelLayer(is_residual, is_normalizing, activation)
            for activation, is_normalizing, is_residual
            in zip(activations, are_normalizing, are_residual)
        ]
        self.parameters_provider = [
            LayerProvider(fan_in, fan_out, rank, number_of_scenes, omega_0)
            for fan_in, fan_out in zip(fans_in, fans_out)
        ]

    @tf.function
    def call(self, inputs):
        scene_id, x = inputs
        scene_id = tf.reshape(scene_id, [1])
        for parameters_provider, layer in zip(self.parameters_provider, self.layers):
            weight_matrix, bias_vector = parameters_provider(scene_id)
            x = layer([x, weight_matrix, bias_vector])
        return x
