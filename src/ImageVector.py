import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
    return img

def get_image_feature_vectors() :

    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)

    # origin_image = "./images/origin.png"
    origin_image = "./images/target3.png"
    # origin_image = "./images/image_test1.png"
    img = load_img(origin_image)
    features = module(img)

    print(features)

    feature_set = np.squeeze(features)

    # print(feature_set)
    print("vector dmis : " , len(feature_set))

get_image_feature_vectors()
