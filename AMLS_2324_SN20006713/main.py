import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random

# Load the model
model_pneumoniamnist = tf.keras.models.load_model("mobilenet_trained_model.h5")
print(model_pneumoniamnist.summary())

# model_pathmnist = tf.keras.models.load_model("mobilenet_trained_model_pathMNIST.h5")
# print(model_pathmnist.summary())

# Load the .npz file
pneumoniamnist_data = np.load('pneumoniamnist.npz')
pneumoniamnist_test = pneumoniamnist_data['test_images']


# pathmnist_data = np.load('pathmnist.npz')
# path_test = pathmnist_data['test_images']


def resize_and_convert(images, new_shape=(224, 224)):
    resized_images = []
    for img in images:
        if img.shape[-1] == 3:
            # Resize the image to the new shape
            resized = cv2.resize(img, new_shape)
            resized_images.append(resized)
        else:  # Image is grayscale or has a different number of channels
            # Resize the image to the new shape and add a third dimension for channels
            resized = cv2.resize(img, new_shape)
            resized = np.expand_dims(resized, axis=-1)
            resized_images.append(np.repeat(resized, 3, axis=-1))  # Convert to RGB
    return np.array(resized_images)


def normalize_images(images):
    # convert into float32
    images = images.astype('float32')
    images /= 255.0
    return images


def get_random_image(data):
    random_index = random.randint(0, len(data) - 1)
    return random_index


def get_binary_model_results(data, random_index, model):
    # Select the image and preprocess it
    image = data[random_index]
    preprocessed_image = resize_and_convert([image], new_shape=(224, 224))
    preprocessed_image = normalize_images(preprocessed_image)

    # Predict the label and probability
    res = model.predict(preprocessed_image, verbose=0)
    res = np.where(res > 0.5, 1, 0).reshape(1, -1)[0]

    # Define classes
    classes = ["Pneumonia", "Normal"]

    # Plot the image with labels
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {classes[res[0]]}")
    plt.axis('off')
    plt.show()


def get_multi_model_results(data, random_index, model):
    # Select the image and preprocess it
    image = data[random_index]
    preprocessed_image = resize_and_convert([image], new_shape=(224, 224))
    preprocessed_image = normalize_images(preprocessed_image)

    # Predict the label and probability
    res = model.predict(preprocessed_image, verbose=0)
    res = np.argmax(np.round(res), axis=1)

    # Define classes
    classes = ['Tissue_0',
               'Tissue_1',
               'Tissue_2',
               'Tissue_3',
               'Tissue_4',
               'Tissue_5',
               'Tissue_6',
               'Tissue_7',
               'Tissue_8']

    # Plot the image with labels
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {classes[res[0]]}")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Choose a random index to select an image
    random_index = get_random_image(pneumoniamnist_test)
    get_binary_model_results(pneumoniamnist_test, random_index, model_pneumoniamnist)

    # random_index = get_random_image(path_test)
    # get_multi_model_results(path_test, random_index, model_pathmnist)
