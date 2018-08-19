# MIT License
#
# Copyright (c) 2017 Tom Runia
# Modifications copyright (C) 2018  Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
# Date modified: 29 July 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Tom Runia
# Date Created: 2017-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import h5py
import numpy as np
import time
from datetime import datetime

from feature_extractor.feature_extractor import FeatureExtractor
import feature_extractor.utils as utils

CATEGORIES_ICWT = {'book': 0, 'cellphone': 1, 'mouse': 2, 'pencilcase': 3, 'ringbinder': 4,
              'hairbrush': 5, 'hairclip': 6, 'perfume': 7, 'sunglasses': 8, 'wallet': 9,
              'flower': 10, 'glass': 11, 'mug': 12, 'remote': 13, 'soapdispenser': 14,
              'bodylotion': 15, 'ovenglove': 16, 'sodabottle': 17, 'sprayer': 18, 'squeezer': 19}

SESSIONS_ICWT = {'MIX':1, 'ROT2D':2, 'ROT3D':3, 'SCALE':4, 'TRANSL':5}


DAYS_ICWT = {'day1': 1, 'day2': 2, 'day3': 3, 'day4': 4, 'day5': 5, 'day6': 6, 'day7': 7, 'day8': 8, 'day9': 9, 'day10': 10,
        'day11': 11, 'day12': 12, 'day13': 13, 'day14': 14, 'day15': 15}

CAMERAS_ICWT = {'left': 1, 'right': 2}

CORE50_ADDITIONAL_COLUMNS = 4
ICWT_ADDITIONAL_COLUMNS = 6
NICO_ADDITIONAL_COLUMNS = 4



def feature_extraction_queue(feature_extractor, image_path, layer_names,
                             batch_size, num_classes, num_images=1000000):
    '''
    Given a directory containing images, this function extracts features
    for all images. The layers to extract features from are specified
    as a list of strings. First, we seek for all images in the directory,
    sort the list and feed them to the filename queue. Then, batches are
    processed and features are stored in a large object `features`.

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param layer_names: list of str, list of layer names
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :param num_images: int, number of images to process (default=100000)
    :return:
    '''

    # Add a list of images to process, note that the list is ordered.
    image_files = utils.find_files(image_path, ("jpg", "png"))
    num_images = min(len(image_files), num_images)
    image_files = image_files[0:num_images]

    num_examples = len(image_files)
    num_batches = int(np.ceil(num_examples/batch_size))

    # Fill-up last batch so it is full (otherwise queue hangs)
    utils.fill_last_batch(image_files, batch_size)

    print("#"*80)
    print("Batch Size: {}".format(batch_size))
    print("Number of Examples: {}".format(num_examples))
    print("Number of Batches: {}".format(num_batches))

    # Add all the images to the filename queue
    feature_extractor.enqueue_image_files(image_files)

    # Initialize containers for storing processed filenames and features
    feature_dataset = {'filenames': []}
    for i, layer_name in enumerate(layer_names):
        layer_shape = feature_extractor.layer_size(layer_name)
        layer_shape[0] = len(image_files)  # replace ? by number of examples
        feature_dataset[layer_name] = np.zeros(layer_shape, np.float32)
        print("Extracting features for layer '{}' with shape {}".format(layer_name, layer_shape))

    print("#"*80)

    # Perform feed-forward through the batches
    for batch_index in range(num_batches):

        t1 = time.time()

        # Feed-forward one batch through the network
        outputs = feature_extractor.feed_forward_batch(layer_names)

        for layer_name in layer_names:
            start = batch_index*batch_size
            end   = start+batch_size
            feature_dataset[layer_name][start:end] = outputs[layer_name]

        # Save the filenames of the images in the batch
        feature_dataset['filenames'].extend(outputs['filenames'])

        t2 = time.time()
        examples_in_queue = outputs['examples_in_queue']
        examples_per_second = batch_size/float(t2-t1)

        print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Examples in Queue = {}, Examples/Sec = {:.2f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), batch_index+1,
            num_batches, batch_size, examples_in_queue, examples_per_second
        ))

    # If the number of pre-processing threads >1 then the output order is
    # non-deterministic. Therefore, we order the outputs again by filenames so
    # the images and corresponding features are sorted in alphabetical order.
    if feature_extractor.num_preproc_threads > 1:
        utils.sort_feature_dataset(feature_dataset)

    # We cut-off the last part of the final batch since this was filled-up
    feature_dataset['filenames'] = feature_dataset['filenames'][0:num_examples]
    for layer_name in layer_names:
        feature_dataset[layer_name] = feature_dataset[layer_name][0:num_examples]

    return feature_dataset


################################################################################
################################################################################
################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorFlow feature extraction")
    parser.add_argument("--network", dest="network_name", type=str, required=True, help="model name, e.g. 'resnet_v2_101'")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, required=True, help="path to pre-trained checkpoint file")
    parser.add_argument("--image_path", dest="image_path", type=str, required=True, help="path to directory containing images")
    # Start: Modifications (Vadym Gryshchuk).
    parser.add_argument("--out_features_h5", dest="out_file", type=str, default="./features.h5", help="path to save features (HDF5 file)")
    parser.add_argument("--out_features_csv", dest="out_features_file", type=str, default="./features.csv", help="path to save features (CSV file)")
    # End: Modifications (Vadym Gryshchuk).

    parser.add_argument("--layer_names", dest="layer_names", type=str, required=True, help="layer names separated by commas")
    parser.add_argument("--preproc_func", dest="preproc_func", type=str, default=None, help="force the image preprocessing function (None)")
    parser.add_argument("--preproc_threads", dest="num_preproc_threads", type=int, default=2, help="number of preprocessing threads (2)")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size (32)")
    parser.add_argument("--num_classes", dest="num_classes", type=int, default=1001, help="number of classes for the CNN model (1001)")
    # Start: Modifications (Vadym Gryshchuk).
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, default="nico", help="'iCubWorld' or 'CORe50'")
    # End: Modifications (Vadym Gryshchuk).
    args = parser.parse_args()

    # resnet_v2_101/logits,resnet_v2_101/pool4 => to list of layer names
    layer_names = args.layer_names.split(",")

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(
        network_name=args.network_name,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        preproc_func_name=args.preproc_func,
        preproc_threads=args.num_preproc_threads
    )

    # Print the network summary, use these layer names for feature extraction
    #feature_extractor.print_network_summary()

    # Feature extraction example using a filename queue to feed images
    feature_dataset = feature_extraction_queue(
        feature_extractor, args.image_path, layer_names,
        args.batch_size, args.num_classes)

    # Modifications (Vadym Gryshchuk). Start.
    features_temp = feature_dataset.get(layer_names[0])
    if features_temp.shape.__len__() == 4:
        features_temp1 = np.squeeze(features_temp, axis=2)
        features = np.squeeze(features_temp1, axis=1)
    else:
        features = feature_dataset.get(layer_names[0])

    labels = feature_dataset.get("filenames")

    if args.dataset_name == 'CORe50':
        features_array = np.zeros((features.shape[0], features.shape[1] + CORE50_ADDITIONAL_COLUMNS), dtype=np.float64)

        # Find the appropriate ids for each object.
        for index_label in range(len(labels)):

            # Get the current label
            label1 = labels[index_label].decode("utf-8").split('/')
            label2 = label1[len(label1) - 1]
            label = label2.split('_')

            # Get the features for this label
            temp = features[index_label]

            # Get object's id.
            object_id = int(str(label).split('_')[2]) - 1

            # Get session's id.
            session = int(str(label).split('_')[1])

            # Get image's id.
            image_id = int(str(label).split('_')[3].split('.')[0])

            if 0 <= object_id <= 4:
                category = 0
            if 5 <= object_id <= 9:
                category = 1
            if 10 <= object_id <= 14:
                category = 2
            if 15 <= object_id <= 19:
                category = 3
            if 20 <= object_id <= 24:
                category = 4
            if 25 <= object_id <= 29:
                category = 5
            if 30 <= object_id <= 34:
                category = 6
            if 35 <= object_id <= 39:
                category = 7
            if 40 <= object_id <= 44:
                category = 8
            if 45 <= object_id <= 49:
                category = 9

            temp1 = np.append(temp, image_id)
            temp2 = np.append(temp1, session)
            temp3 = np.append(temp2, category)
            temp4 = np.append(temp3, object_id)

            features_array[index_label] = temp4

        feature_dataset[layer_names[0]] = features_array

    if args.dataset_name == 'ICWT':

        features_array = np.zeros((features.shape[0], features.shape[1] + ICWT_ADDITIONAL_COLUMNS), dtype=np.float64)

        # Find the appropriate ids for each object.
        for index_label in range(len(labels)):

            # Get the current label
            splitted_path = labels[index_label].decode("utf-8").split('/')
            label = splitted_path[len(splitted_path) - 1].split('_')

            # Get the features for this label
            temp = features[index_label]

            temp1 = np.append(temp, CATEGORIES_ICWT.get(label[1]))
            temp2 = np.append(temp1, label[2])
            temp3 = np.append(temp2, SESSIONS_ICWT.get(label[3]))
            temp4 = np.append(temp3, DAYS_ICWT.get(label[4]))
            temp5 = np.append(temp4, CAMERAS_ICWT.get(label[5]))
            temp6 = np.append(temp5, label[6][:-4])

            features_array[index_label] = temp6

        feature_dataset[layer_names[0]] = features_array

    if args.dataset_name == 'NICO':

        # Extend the number of columns.
        features_array = np.zeros((features.shape[0], features.shape[1] + NICO_ADDITIONAL_COLUMNS), dtype=np.float64)

        # Find the object id, session id and category id for each object.
        for index_label in range(len(labels)):

            # Get the current label.
            label1 = labels[index_label].decode("utf-8").split('/')
            label2 = label1[len(label1) - 1]
            label3 = label2.split('_')

            # Get the features for this label.
            temp = features[index_label]
            temp0 = np.append(temp, int(label3[0]) - 1)   # category
            temp1 = np.append(temp0, int(label3[1]) - 1)  # object class
            temp2 = np.append(temp1, label3[2])  # session
            temp3 = np.append(temp2, label3[3][:-4])  # image id within the object class
            features_array[index_label] = temp3

        feature_dataset[layer_names[0]] = features_array

    # End: Modifications (Vadym Gryshchuk).

    # Write features to disk as HDF5 file
    utils.write_hdf5(args.out_file, layer_names, feature_dataset)
    print("Successfully written features to: {}".format(args.out_file))

    # Close the threads and close session.
    feature_extractor.close()
    print("Finished.")

    # Start: Modifications (Vadym Gryshchuk).

    f = h5py.File(args.out_file, 'r')
    dataset = f[args.layer_names][...]

    np.savetxt(args.out_features_file, dataset, delimiter=',')
    np.save
    print("Features are written also to: {}".format(args.out_features_file))
    print("Done.")

    f.close()

    # End: Modifications (Vadym Gryshchuk).
