#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: grad_cam.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool


class BaseGradCAM(object):
    def __init__(self, vis_model=None, num_channel=3):
        self._vis_model = vis_model
        self._nchannel = num_channel

    def create_model(self, inputs):
        self._create_model(inputs)

    def _create_model(self, inputs):
        pass

    def setup_graph(self):
        pass

    "_comp_feature_importance_weight: grad-CAM和CAM的差異之處在於取權重的方式"
    def _comp_feature_importance_weight(self, class_id):
        if not isinstance(class_id, list):
            class_id = [class_id]

        with tf.name_scope('feature_weight'):
            self._feature_w_list = []
            for idx, cid in enumerate(class_id):
                # 將輸出轉為指定類別為1，其餘為0，shape為(nclass, 1)的one hot 形式
                one_hot = tf.sparse_to_dense(
                    [[cid, 0]], [self._nclass, 1], 1.0)
                # _out_act: 每個類別的激活值，reshape to (1, nclass)
                out_act = tf.reshape(self._out_act, [1, self._nclass])
                #class_act(1, 1) = _out_act(1, nclass) * one_hot(nclass, 1)，代表僅保留該類別的激活值
                class_act = tf.matmul(out_act, one_hot,
                                      name='class_act_{}'.format(idx))
                #該類別的激活值(class_act)對特徵圖的每個像素(_conv_out)求導，con_out是conv5_4的卷積結果(最後一層卷積特徵圖)
                feature_grad = tf.gradients(class_act, self._conv_out,
                                            name='grad_{}'.format(idx))
                #將feature_grad刪除第0維度，shape會和con_out的shape一樣，con_out中每張特徵圖的每個像素值，都有對應的導數(權值)。
                feature_grad = tf.squeeze(
                    tf.convert_to_tensor(feature_grad), axis=0)
                #feature_w: 對feature_grad做GAP，即針對特徵圖中每個像素值的導數(加權值)取平均，得一張特徵圖一個權值
                feature_w = global_avg_pool(
                    feature_grad, name='feature_w_{}'.format(idx))
                self._feature_w_list.append(feature_w)

    def get_visualization(self, class_id=None):
        assert class_id is not None, 'class_id cannot be None!'

        with tf.name_scope('grad_cam'):
            # 取得con_out每張特徵圖的加權值
            self._comp_feature_importance_weight(class_id)
            conv_out = self._conv_out
            conv_c = tf.shape(conv_out)[-1]     #conv_out(feature maps) have 512 channels
            conv_h = tf.shape(conv_out)[1]      #最後一層捲基層輸出conv_out(即特徵圖)的長
            conv_w = tf.shape(conv_out)[2]      #最後一層捲基層輸出conv_out(即特徵圖)的寬
            conv_reshape = tf.reshape(conv_out, [conv_h * conv_w, conv_c])  #將feature map reshape為(heigh*width, 512)

            o_h = tf.shape(self.input_im)[1]    #原圖的長
            o_w = tf.shape(self.input_im)[2]    #原圖的寬

            classmap_list = []
            for idx, feature_w in enumerate(self._feature_w_list):
                feature_w = tf.reshape(feature_w, [conv_c, 1])              #feature_w reshape to (512, 1)
                classmap = tf.matmul(conv_reshape, feature_w)               #每張特徵圖乘上該特徵圖的權重 => (heigh*width, 1) = conv_reshape(heigh*width, 512) * feature_w(512, 1)
                classmap = tf.reshape(classmap, [-1, conv_h, conv_w, 1])    #reshap to (conv_h, conv_w)
                classmap = tf.nn.relu(
                    tf.image.resize_bilinear(classmap, [o_h, o_w]),         #resize to (o_h, o_w)
                    name='grad_cam_{}'.format(idx))
                classmap_list.append(tf.squeeze(classmap))

            return classmap_list, tf.convert_to_tensor(class_id)            #classmap_list為每個class id對應的grad-CAM熱力圖


class ClassifyGradCAM(BaseGradCAM):
    def _create_model(self, inputs):
        keep_prob = 1
        self._vis_model.create_model([inputs, keep_prob])

    def setup_graph(self):
        self.input_im = self._vis_model.layer['input']
        # layer['output']: 是最後一個1*1 conv的輸出(命名為fc8)，其深度為nclass
        # _out_act: 每個類別的激活值，用以代表每個類別的特徵圖
        self._out_act = global_avg_pool(self._vis_model.layer['output'])
        # layer['conv_out']: 是特徵抽取之捲基層的最後一層輸出(為'conv5_4'之輸出)
        self._conv_out = self._vis_model.layer['conv_out']
        self._nclass = self._out_act.shape.as_list()[-1]
        #pre_label: 為所有類別的激活值中，前5大的激活值與對應的類別號碼，並從大致小排序
        self.pre_label = tf.nn.top_k(tf.nn.softmax(self._out_act),
                                     k=5, sorted=True)
