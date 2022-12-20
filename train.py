import argparse
import os
from pathlib import Path
from enum import Enum

import tensorflow as tf

import layers
import callbacks
import data
from layers import SVDSiren, FourierFeaturesProvider
from callbacks import MultipleScenesImagesCallbackFactory
from data import MacroPixelImage8BitsDatasetFactory, MulipleScenesDatasetFactory

FOURIER_FEATURES_MAX_FREQUENCIES = [.2, .2, .06, .06]
FOURIER_FEATURES_DIM = 512

class ScenesDirectoryMode(Enum):
    SUBAPERTURE_IMAGES = 0
    MACROPIXEL_IMAGES = 1

class ModelFactory:

    @staticmethod
    def make(output_dim, depth, hidden_dim, rank, number_of_scenes, coordinates_range, omega_0):
        encoder = FourierFeaturesProvider(
            number_of_scenes, FOURIER_FEATURES_MAX_FREQUENCIES, FOURIER_FEATURES_DIM, coordinates_range)
        siren = SVDSiren(
            encoder.encoding_dim, output_dim, depth, hidden_dim, rank, number_of_scenes, omega_0)
        scene_id, coordinates = tf.keras.Input([], dtype=tf.int32), tf.keras.Input([encoder.coordinates_dim])
        codes = encoder([scene_id, coordinates])
        values = siren([scene_id, codes])
        return tf.keras.models.Model([scene_id, coordinates], values)

def get_args():
    parser = argparse.ArgumentParser()
    arg_name_to_default = {
        'epochs': 100,
        'steps_per_epoch': 80000,
        'batch_size': 4096,
        'hidden_dim': 512,
        'depth': 9,
        'rank': 2048,
        'omega_0': 30.,
        'verbose': 2,
        'lr_cosine_decay_initial_value': 1e-5,
        'lr_cosine_decay_final_value': 1e-8,
        'adam_epsilon': 1e-7,
        'scenes_directory': "datasets/Flowers",
        'cache_directory': "cache",
        'checkpoint_root': "models",
        'tensorboard_root': "tb"
    }
    for arg_name, default in arg_name_to_default.items():
        parser.add_argument(f'--{arg_name}', type=type(default), default=default)
    for arg_name, const in [
        ('supaperture_images', ScenesDirectoryMode.SUBAPERTURE_IMAGES),
        ('macropixel_images', ScenesDirectoryMode.MACROPIXEL_IMAGES)
    ]:
        parser.add_argument(
            f'--{arg_name}', dest='scenes_directory_mode', action='store_const',
            const=const, default=ScenesDirectoryMode.MACROPIXEL_IMAGES)
    parser.add_argument('--limit_number_of_scenes', type=int, default=None)
    args = parser.parse_args()
    return args

def train(args):
    checkpoint_directory = get_checkpoint_directory(args)
    checkpoint_path = checkpoint_directory + "{epoch:02d}.ckpt"
    tensorboard_path = os.path.expandvars(
        f"{args.tensorboard_root}/"
        f"{Path(checkpoint_directory).parent.name}/"
        f"{Path(checkpoint_directory).name}")

    ds_factory = get_ds_factory(args)
    ds = ds_factory.make(args.batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    model, initial_epoch = get_model_and_initial_epoch(checkpoint_directory, ds_factory, args)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(tensorboard_path)
    img_callback = MultipleScenesImagesCallbackFactory(tensorboard_path, model, ds_factory).make()
    callbacks = [ckpt_callback, tb_callback, img_callback]

    model.fit(
        ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=args.verbose
    )

def get_ds_factory(args):
    scenes_directory = os.path.expandvars(args.scenes_directory)
    cache_directory = os.path.expandvars(args.cache_directory)
    if args.scenes_directory_mode == ScenesDirectoryMode.SUBAPERTURE_IMAGES:
        return MulipleScenesDatasetFactory(
            scenes_directory,
            cache_directory,
            take_random_scenes_up_to=args.limit_number_of_scenes
        )
    elif args.scenes_directory_mode == ScenesDirectoryMode.MACROPIXEL_IMAGES:
        return MacroPixelImage8BitsDatasetFactory(
            scenes_directory,
            cache_directory,
            limit_number_of_scenes=args.limit_number_of_scenes
        )
    else:
        raise ValueError(args.scenes_directory_mode)

def get_checkpoint_directory(args):
    unexpanded_parent_path = (
        f"{args.checkpoint_root}/"
        f"{Path(os.path.expandvars(args.scenes_directory)).name}"
        f"{'' if args.limit_number_of_scenes is None else args.limit_number_of_scenes}"
    )
    model_name = "-".join(
        [f"{s}{x:.0e}" if type(x) == float else f"{s}{x}" for s, x in [
            ('epc', args.epochs),
            ('spe', args.steps_per_epoch),
            ('b', args.batch_size),
            ('w', args.hidden_dim),
            ('d', args.depth),
            ('rk', args.rank),
            ('om', args.omega_0),
            ('lri', args.lr_cosine_decay_initial_value),
            ('lrf', args.lr_cosine_decay_final_value),
        ]])
    return os.path.expandvars(unexpanded_parent_path + '/' + model_name + '/')

def get_model_and_initial_epoch(checkpoint_directory, ds_factory, args):
    model = make_model(ds_factory, args)
    schedule = get_schedule(args)
    optimizer = tf.keras.optimizers.Adam(schedule, epsilon=args.adam_epsilon, clipvalue=1.)
    model.compile(optimizer, tf.keras.losses.mean_squared_error)
    print(f"Looking for checkpoint in {checkpoint_directory}")
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory)
    if checkpoint_path is None:
        print("Found no checkpoint.")
        return model, 0
    print(f"Found checkpoint at {checkpoint_path}")
    initial_epoch = int(Path(checkpoint_path).stem)
    model.load_weights(checkpoint_path)
    return model, initial_epoch

def make_model(ds_factory, args):
    return ModelFactory.make(
        ds_factory.max_keys_ranges[-1].numpy(),
        args.depth,
        args.hidden_dim,
        args.rank,
        ds_factory.num_scenes,
        ds_factory.max_keys_ranges[:-1],
        args.omega_0)

def get_schedule(args):
    alpha = args.lr_cosine_decay_final_value / args.lr_cosine_decay_initial_value
    training_steps = args.epochs * args.steps_per_epoch
    return tf.keras.optimizers.schedules.CosineDecay(args.lr_cosine_decay_initial_value, training_steps, alpha)

def display(args):
    print('\n' + 128 * '*')
    for k, v in args._get_kwargs():
        print((f"{k}:" + 64 * ' ')[:32] + f"{v}")
    print(128 * '*' + '\n')

def test():
    # Model can be built.
    model = ModelFactory.make(3, 9, 512, 2048, 2000, tf.constant([375, 540, 8, 8]), 30.)
    model.summary()

    # Model can be called and produces finite values.
    scene_id = tf.constant([0])
    mock_coordinates = tf.random.uniform([4096, 4])
    mock_values = tf.random.uniform([4096, 3])
    predicted = model([scene_id, mock_coordinates])
    tf.debugging.assert_all_finite(predicted, 'Predicted values are not finite')

    # Model can be trained.
    model.compile('Adam', tf.keras.losses.mean_squared_error)
    mock_dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(mock_coordinates),
        tf.data.Dataset.from_tensor_slices(mock_values))
    ).batch(4096).map(lambda c, v: ((scene_id, c), v))
    model.fit(mock_dataset, epochs=1, verbose=0)

TESTING = False

if __name__ == "__main__":
    if TESTING:
        test()
    else:
        args = get_args()
        display(args)
        train(args)
