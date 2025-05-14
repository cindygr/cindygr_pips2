import tensorflow as tf
import datetime, os
import scipy
from tensorflow.python.framework import tensor_util
from PIL import Image
import numpy as np
from collections import defaultdict, namedtuple
from typing import List
import cv2
from tensorflow.core.util import event_pb2

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

   #image_str = tf.placeholder(tf.string)
    image_str = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1


def extract_images_from_event(event_filename: str, image_tags: List[str]):
    topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for v in event.summary.value:
            if v.tag in image_tags:
                print(f"{v.tag} fpp")

                print(f"{v.tag} tensor")
                s = v.tensor.string_val[2]  # first elements are W and H

                tf_img = tf.image.decode_image(s)  # [H, W, C]
                np_img = tf_img.numpy()

                topic_counter[v.tag] += 1

                cnt = topic_counter[v.tag]
                #tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)

                yield np_img


def extract_images_from_event_file(event_file, output_dir):
    for event in tf.compat.v1.train.summary_iterator(event_file):
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    tag = value.tag
                    step = event.step
                    scalar_value = value.simple_value
                    print(f"Step: {step}, Tag: {tag}, Value: {scalar_value}")
                elif value.HasField('image'):
                    print(f"image: {value.tag}")
                    img = value.image
                    width = img.width
                    height = img.height
                    print(f"width: {width}, height: {height}")
                    image_data = tf.image.decode_gif(img.encoded_image_string).numpy()
                    img_array = np.array(image_data)
                    for indx in range(0, img_array.shape[0]):
                        im = img_array[indx, :, :, :].squeeze()
                        img_pil = Image.fromarray(im)
                        img_path = output_dir + f'im{indx}.png'
                        img_pil.save(img_path, lossless=True, quality=100)
        """
            tag =
        for value in event.summary.value:
            #if value .HasField('images'):
            if "trajs_on_rgbs" in value.tag:
                print(f"value {value}")
                img_path = os.path.join(output_dir, f'{value.tag}_{event.step}.webp')
                if os.path.isfile(img_path): continue

                t = value.tensor
                s = value.tensor.string_val  # first elements are W and H
                img = value.video
                image_data = tf.image.decode_gif(img.encoded_image_string).numpy()
                img_array = np.array(image_data)
                img_pil = Image.fromarray(img_array)
                img_pil.save(img_path, lossless=True, quality=100)
                print(img_path)
        """


if __name__ == '__main__':
    home_dir = "/Users/cindygrimm/PycharmProjects/cindygr_pips2/logs_demo/"
    event_file = home_dir + "tree_short_48_1024_de00_15:25:41/t/events.out.tfevents.1747088741.10-249-96-89.wireless.oregonstate.edu"
    output_dir = '/Users/cindygrimm/PycharmProjects/cindygr_pips2/logs_demo/'

    # save_images_from_event(fn=event_file, tag="camel", output_dir='/Users/grimmc/PycharmProjects/cindy_pips2/output/')

    extract_images_from_event_file(event_file=event_file, output_dir=output_dir)
    """return
    for indx, img in enumerate(
            extract_images_from_event(event_file, ["topic", "images", "trajs_on_rgbs", "first_step"])):
        cv2.imwrite(f"foo{indx}.png", img)
        print(indx)

        save_images_from_event("/Users/grimmc/PycharmProjects/cindy_pips2/logs_demo/camel_48_1024_de00_12:00:05",
                               tag="camel", output_dir='/Users/grimmc/PycharmProjects/cindy_pips2/output/')
    """
