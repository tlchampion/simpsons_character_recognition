# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from os.path import exists
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders


def split_image_folders(input, output):
    if os.path.isdir(output):
        print("Directory already exists.")
    else:
        print("Creating new folders and splitting images into 3 datasets")

        splitfolders.ratio(input, output, seed=42, ratio=(
            .75, .15, .10), move=False)


def create_datasets(traindir, testdir, test_labels):
    """
    Creates test, train and validation datasets from image directories and will
    generate a csv file with the test labels based on the test image filenames.

    Requires input of directory path to the train and test datasets as well as
    the path to a csv file for the test image labels.

    Returns the following:
    datagen: ImageDataGenerator instance to be used for train and validation datasets
    datagen_test: ImageDataGenerator instance for test dataset
    train: training dataset
    validation: validation dataset
    test: test dataset

    """

    file_exists = exists(test_labels)

    if file_exists == False:
        testfiles = pd.DataFrame(os.listdir(testdir), columns=['filename'])
        testlabels = testfiles['filename'].str.split(".", expand=True)
        testlabels = testlabels[0].str.rpartition('_', expand=True)
        data = [testfiles, testlabels]
        test_data_files = pd.concat(data, axis=1)
        test_data_files = test_data_files.drop(columns=[1, 2])
        test_data_files.rename(
            columns={'filename': 'filename', 0: 'label'}, inplace=True)
        test_data_files.to_csv(test_labels, index=False)
    else:
        test_data_files = pd.read_csv(test_labels)

    datagen = ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)

    datagen_test = ImageDataGenerator(rescale=1./255,
                                      validation_split=0.2,
                                      horizontal_flip=False,
                                      vertical_flip=False)

    train = datagen.flow_from_directory(directory=traindir,
                                        batch_size=32,
                                        target_size=(224, 224),
                                        class_mode='categorical',
                                        subset='training',
                                        seed=42,
                                        shuffle=True)

    validation = datagen.flow_from_directory(directory=traindir,
                                             batch_size=32,
                                             target_size=(224, 224),
                                             class_mode='categorical',
                                             subset='validation',
                                             seed=42,
                                             shuffle=True)

    test = datagen_test.flow_from_dataframe(
        dataframe=test_data_files,
        directory=testdir,
        X_col="location",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        seed=42,
        shuffle=False
    )

    return datagen, datagen_test, train, validation, test


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    datagen, dataget_test, train, validation, test = create_datasets()
