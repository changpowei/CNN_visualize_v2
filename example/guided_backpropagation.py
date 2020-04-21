#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: guided_backpropagation.py
# Author: Qian Ge <geqian1001@gmail.com>

from scipy import misc
import argparse
import scipy.io
import numpy as np
import tensorflow as tf
from tensorcv.dataflow.image import ImageFromFile
import config_path as config
import class_name as C
import sys
sys.path.append('../')
from lib.nets.vgg import VGG19_FCN
from lib.models.guided_backpro import GuideBackPro

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-it', '--imtype', type=str, default='.jpg',
                        help='Image type! Default = .jpg')
    parser.add_argument('-cid', '--class_id', type=int, default=None,
                        help='Assign class id! Default = None')
    parser.add_argument('-t', '--top', type=int, default=1,
                        help='前幾大激活值得導向反向傳播，預設為1。')

    return parser.parse_args()

if __name__ == '__main__':
    
    FLAGS = get_parse()
    
    # placeholder for input image
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    # initialize input dataflow
    # change '.png' to other image types if other types of images are used
    input_im = ImageFromFile(FLAGS.imtype, data_dir = config.im_path,
                             num_channel=3, shuffle=False)
    # batch size has to be one
    input_im.set_batch_size(1)

    # initialize guided back propagation class
    # use VGG19 as an example
    # images will be rescaled to smallest side = 224 if is_rescale=True
    #class_id != None:會返回辨認為指定class的導向反向傳播的圖片
    #class_id = None:則會告知Top5(imagnet預設為1000類)的分類結果與對應的機率，並透過導向反向傳播解釋。
    model = GuideBackPro(vis_model=VGG19_FCN(is_load=True,
                                             pre_train_path = config.vgg_path,
                                             is_rescale=True)
                         , class_id = FLAGS.class_id
                         , top = FLAGS.top)

    # get op to compute guided back propagation map
    # final output respect to input image
    """
    back_pro_op: 一個tuple包含兩個list
    back_pro_op[0]: list中每個值，是激活值對輸入圖片透過導向反向傳播求導值，
                    為和輸入一樣大的矩陣，分別為激活值對每個像素的導數
    back_pro_op[1]: list中每個值，是該輸入圖所對應的類別，
                    有指定class_id: 是該class_id
                    沒指定class_id: 分類的結果
    """
    back_pro_op = model.get_visualization(image)

    #writer = tf.summary.FileWriter(config.save_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #writer.add_graph(sess.graph)

        cnt = 0
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            
            guided_backpro, label, o_im =\
                sess.run([back_pro_op, model.pre_label, model.input_im],
                         feed_dict={image: im})
            print("\n第%d張影像前%d類分類結果："%(cnt, len(label[0][0])))
            for i in range(len(label[0][0])):
                print("置信度:%5.2f%% 類別(%3d):%s"%(label[0][0][i]*100, label[1][0][i], C.class_name[label[1][0][i]]))
            
            
            for cid, guided_map in zip(guided_backpro[1], guided_backpro[0]):
                scipy.misc.imsave(
                    '{}map_{}_class_{}.png'.format(config.save_path_guided + 'guided/', cnt, cid),
                    np.squeeze(guided_map))
            scipy.misc.imsave('{}im_{}.png'.format(config.save_path_guided + 'org/', cnt),
                              np.squeeze(o_im))
            # scipy.io.savemat(
            #     '{}map_1_class_{}.mat'.format(SAVE_DIR, cid),
            #     {'mat': np.squeeze(guided_map)*255})
            cnt += 1

    #writer.close()
