#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gradcam.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse
from itertools import count

import tensorflow as tf
import numpy as np
from tensorcv.dataflow.image import ImageFromFile
from tensorcv.utils.viz import image_overlay, save_merge_images
import class_name as C
import sys
sys.path.append('../')
from lib.nets.vgg import VGG19_FCN
from lib.models.guided_backpro import GuideBackPro
from lib.models.grad_cam import ClassifyGradCAM
from lib.utils.viz import image_weight_mask

IM_PATH = '../data/'
SAVE_DIR = '../grad_CAM/'
VGG_PATH = '../lib/nets/pretrained/vgg19.npy'


# def image_weight_mask(image, mask):
#     """
#     Args:
#         image: image with size [HEIGHT, WIDTH, CHANNEL]
#         mask: image with size [HEIGHT, WIDTH, 1] or [HEIGHT, WIDTH]
#     """
#     image = np.array(np.squeeze(image))
#     mask = np.array(np.squeeze(mask))
#     assert len(mask.shape) == 2
#     assert len(image.shape) < 4
#     mask.astype('float32')
#     mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
#     mask = mask / np.amax(mask)

#     if len(image.shape) == 2:
#         return np.multiply(image, mask)
#     else:
#         for c in range(0, image.shape[2]):
#             image[:, :, c] = np.multiply(image[:, :, c], mask)
#         return image
def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-it', '--imtype', type=str, default='.jpg',
                        help='Image type! Default = .jpg')
    parser.add_argument('-cid', '--class_id', type=str, default=None,
                        help='Assign class id! Default = None')
    parser.add_argument('-t', '--top', type=int, default=2,
                        help='前幾大激活值得導向反向傳播，預設為1。')
    return parser.parse_args()

if __name__ == '__main__':
    
    FLAGS = get_parse()

    # merge several output images in one large image
    merge_im = 1
    grid_size = np.ceil(merge_im**0.5).astype(int)

    # class label for Grad-CAM generation
    # 355 llama 543 dumbbell 605 iPod 515 hat 99 groose 283 tiger cat
    # 282 tabby cat 233 border collie 242 boxer
    # class_id = [355, 543, 605, 515]
    # class_id = [283, 242]
    if FLAGS.class_id is not None:
        class_id = [int(string) for string in FLAGS.class_id.split(',')]
            

    # initialize Grad-CAM
    # using VGG19
    # #VGG19_FCN的model透過NIN網路(1*1的conv)取代全連接層，透過 4096個7*7*512 => 4096個1*1*4096 => 1000個1*1*4096 取代
    gcam = ClassifyGradCAM(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True))
    
    gbackprob = GuideBackPro(
        vis_model=VGG19_FCN(is_load=True,
                            pre_train_path=VGG_PATH,
                            is_rescale=True)
                            , top = FLAGS.top)

    # placeholder for input image
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    # create VGG19 model
    gcam.create_model(image)
    # 獲取不同類別的激活值與對應的特徵圖(最後一層特徵抽取捲基層的輸出)
    gcam.setup_graph()

    # generate class map and prediction label ops
    #map_op = gcam.get_visualization(class_id=class_id)
    label_op = gcam.pre_label

    back_pro_op = gbackprob.get_visualization(image)

    # initialize input dataflow
    # change '.png' to other image types if other types of images are used
    input_im = ImageFromFile(FLAGS.imtype, data_dir=IM_PATH,
                             num_channel=3, shuffle=False)
    input_im.set_batch_size(1)

    #writer = tf.summary.FileWriter(SAVE_DIR)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #writer.add_graph(sess.graph)

        cnt = 0
        merge_cnt = 0
        # weight_im_list = [[] for i in range(len(class_id))]
        o_im_list = []
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            b_map, label, o_im =\
                sess.run([back_pro_op, label_op, gcam.input_im],
                         feed_dict={image: im})
            
            if FLAGS.class_id is None:
                class_id = label[1][0].tolist()[:FLAGS.top]
            
            #map_op獲得grad-CAM熱力圖的operation
            map_op = gcam.get_visualization(class_id=class_id)
            #gcam_map: 熱力圖
            gcam_map = sess.run([map_op], feed_dict={image: im})
            
            print("\n第%d張影像前%d類分類結果："%(cnt, len(label[0][0])))
            for i in range(len(label[0][0])):
                print("置信度:%5.2f%% 類別(%3d):%s"%(label[0][0][i]*100, label[1][0][i], C.class_name[label[1][0][i]]))
            
            gcam_class_id = gcam_map[0][1]
            o_im_list.extend(o_im)
            for idx, cid, cmap, bmap in zip(count(), gcam_map[0][1], gcam_map[0][0], b_map[0]):
                #overlay_im: grad-cam的結果
                overlay_im = image_overlay(cmap, o_im)
                #weight_im: guided grad-cam的結果
                weight_im = image_weight_mask(bmap[0], cmap)
                
                try:
                    weight_im_list[idx].append(weight_im)
                    overlay_im_list[idx].append(overlay_im)
                except NameError:
                    weight_im_list = [[] for i in range(len(gcam_class_id))]
                    overlay_im_list = [[] for i in range(len(gcam_class_id))]
                    weight_im_list[idx].append(weight_im)
                    overlay_im_list[idx].append(overlay_im)
            merge_cnt += 1

            # Merging results
            if merge_cnt == merge_im:
                save_path = '{}oim_{}.png'.format(SAVE_DIR, cnt, cid)
                save_merge_images(np.array(o_im_list),
                                  [grid_size, grid_size],
                                  save_path)
                for w_im, over_im, cid in zip(weight_im_list,
                                              overlay_im_list,
                                              gcam_class_id):
                    # save grad-cam results
                    save_path = '{}gradcam_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(over_im), [grid_size, grid_size], save_path)
                    # save guided grad-cam results
                    save_path = '{}guided_gradcam_{}_class_{}.png'.\
                        format(SAVE_DIR, cnt, cid)
                    save_merge_images(
                        np.array(w_im), [grid_size, grid_size], save_path)
                weight_im_list = [[] for i in range(len(gcam_class_id))]
                overlay_im_list = [[] for i in range(len(gcam_class_id))]
                o_im_list = []
                merge_cnt = 0
                cnt += 1

        # Saving results
        if merge_cnt > 0:
            save_path = '{}oim_{}.png'.format(SAVE_DIR, cnt, cid)
            save_merge_images(np.array(o_im_list),
                              [grid_size, grid_size],
                              save_path)
            for w_im, over_im, cid in zip(weight_im_list,
                                          overlay_im_list,
                                          gcam_class_id):
                # save grad-cam results
                save_path = '{}gradcam_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(over_im), [grid_size, grid_size], save_path)
                # save guided grad-cam results
                save_path = '{}guided_gradcam_{}_class_{}.png'.\
                    format(SAVE_DIR, cnt, cid)
                save_merge_images(
                    np.array(w_im), [grid_size, grid_size], save_path)
    #writer.close()
