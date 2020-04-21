#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: deconv.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import argparse
import numpy as np
import tensorflow as tf
from tensorcv.dataflow.image import ImageFromFile

import config_path as config
import sys
sys.path.append('../')

from lib.nets.vgg import DeconvBaseVGG19, BaseVGG19
import lib.utils.viz as viz
import lib.utils.normalize as normlize
import lib.utils.image as uim


IM_SIZE = 224

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-it', '--imtype', type=str, default='.jpg',
                        help='Image type')
    parser.add_argument('-f','--feat', type=str, default='conv4_4',
                        help='Choose of feature map layer')
    parser.add_argument('-i','--id', type=int, default=None,
                        help='feature map id')

    return parser.parse_args()

def im_scale(im):
    return uim.im_rescale(im, [IM_SIZE, IM_SIZE])

if __name__ == '__main__':
    FLAGS = get_parse()
    
    input_im = ImageFromFile(FLAGS.imtype,
                             data_dir=config.im_path,
                             num_channel=3,
                             shuffle=False,
                             pf=im_scale,
                             )
    input_im.set_batch_size(1)

    
    vizmodel = DeconvBaseVGG19(config.vgg_path,
                           feat_key=FLAGS.feat,
                           pick_feat=FLAGS.id)
    
    
    #vizmap: 返回deconvolution的結果
    vizmap = vizmodel.layers['deconvim']
    #feat_op: 返回指定卷積層輸出的特徵圖，但僅保留整層特徵圖中，被激活函數激活後的最大值(單一值)，其餘皆改為0。
    feat_op = vizmodel.feats
    #max_act_op: 被激活函數激活後的最大值(單一值)。
    max_act_op = vizmodel.max_act

    act_size = vizmodel.receptive_size[FLAGS.feat]
    act_scale = vizmodel.stride[FLAGS.feat]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        max_act_list = []
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            max_act = sess.run(max_act_op, feed_dict={vizmodel.im: im})
            max_act_list.append(max_act)

        #max_list: 依所有測試圖片經指定卷積層後的激活值做排序，為激活值由大到小的測試圖片順序
        #im_file_list: 未經激活值大小排序的所有原測試圖片路徑
        max_list = np.argsort(max_act_list)[::-1]
        im_file_list = input_im.get_data_list()[0]

        feat_list = []
        im_list = []
        
        #最多執行激活值前9大的圖片
        for i in range(0, np.min([len(im_file_list), 9])):
            #im = input_im.next_batch()[0]
            #file_path = os.path.join(config.im_path, im_file_list[max_list[i]])
            
            #從激活值大到小依序讀入圖片。
            im = np.array([im_scale(scipy.misc.imread(im_file_list[max_list[i]], mode='RGB'))])

            '''
            cur_vizmap: 為feat_map做deconvolution的結果，shape為(1, 224, 224, 3)。
            
            feat_map: 為指定卷積層(如：conv4_4)輸出的特徵圖(如：(1, 28, 28, 512))，
                    但僅保留整層特徵圖中，被激活函數激活後的最大值，其餘皆改為0。
                    (如：保留第357張特徵圖第(12, 16)的值4915.98，其餘皆改為0)
                    
            max_act: 該層所有特徵圖中，被激活函數激活的最大值，如4915.98。
            '''
            cur_vizmap, feat_map, max_act = sess.run(
                [vizmap, feat_op, max_act_op], feed_dict={vizmodel.im: im})

            #act_ind: 激活最大值的位置，為一個tuple(Numpy array, Numpy array, Numpy array, Numpy array)，值為(0, 12, 16, 357)。
            act_ind = np.nonzero((feat_map))
            print('Location of max activation {}'.format(act_ind))
            # get only the first nonzero element
            act_c = (act_ind[1][0], act_ind[2][0])
            min_x = max(0, int(act_c[0] * act_scale - act_size / 2))
            max_x = min(IM_SIZE, int(act_c[0] * act_scale + act_size / 2))
            min_y = max(0, int(act_c[1] * act_scale - act_size / 2))
            max_y = min(IM_SIZE, int(act_c[1] * act_scale + act_size / 2))

            im_crop = im[0, min_x:max_x, min_y:max_y, :]
            act_crop = cur_vizmap[0, min_x:max_x, min_y:max_y, :]

            pad_size = (act_size - im_crop.shape[0], act_size - im_crop.shape[1])
            im_crop = np.pad(im_crop,
                             ((0, pad_size[0]), (0, pad_size[1]), (0, 0)),
                             'constant',
                             constant_values=0)
            act_crop = np.pad(act_crop,
                              ((0, pad_size[0]),(0, pad_size[1]), (0, 0)),
                              'constant',
                              constant_values=0)

            feat_list.append(act_crop)
            im_list.append(im_crop)

        viz.viz_filters(np.transpose(feat_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path + 'deconv/', '{}_feat.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)
        viz.viz_filters(np.transpose(im_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path + 'org/', '{}_im.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)
        
