#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: guided_backpro.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.layers import global_avg_pool


@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y


class GuideBackPro(object):
    def __init__(self, vis_model=None, class_id=None, top=1):
        assert vis_model is not None, 'vis_model cannot be None!'
        # assert not class_id is None, 'class_id cannot be None!'

        self._vis_model = vis_model
        if class_id is not None and not isinstance(class_id, list):
            class_id = [class_id]
        self._class_id = class_id
        self._top = top

    def _create_model(self, image):
        keep_prob = 1
        self._vis_model.create_model([image, keep_prob])
        self.input_im = self._vis_model.layer['input']
        
        #VGG19_FCN: 沒有全連接層，透過 4096個7*7*512 => 4096個1*1*4096 => 1000個1*1*4096 取代
        #layer['output']: 一個四維的tensor
        #_out_act: 為VGG19_FCN之輸出層(同樣命名為fc8)經過GAP的activate值 => 1000*1
        self._out_act = global_avg_pool(self._vis_model.layer['output'])
        
        #pre_label: 透過softmax將_out_act轉成機率型態，緊保留前k個機率最大的分類結果，並由大到小排序。
        self.pre_label = tf.nn.top_k(
            tf.nn.softmax(self._out_act), k=5, sorted=True)

    def _get_activation(self):
        with tf.name_scope('activation'):
            nclass = self._out_act.shape.as_list()[-1]
            act_list = []
            
            if self._class_id is None:
                #取前5大沒有經過softmax的激活值，並排序
                top_5_act = tf.nn.top_k(self._out_act, k=5, sorted=True)
                class_list = []
                #class_list: 沒指定class_id，則返回pre_label索引位置[0][0]最大的機率值
                #act_list: 對應的activate值，act_list為含有一個tensor的list                
                for i in range(self._top):
                    class_list.append(top_5_act.indices[0][i])
                    act_list.append(top_5_act.values[0][i])
                #class_list = [self.pre_label.indices[0][0]]
                #act_list = [tf.reduce_max(self._out_act)]
            
            else:
                class_list = self._class_id
                for cid in class_list:
                    #轉成one hot的形式(預設nclass為1000類)，指定類別的值為1，其餘為0，shape為(1000, 1)。
                    one_hot = tf.sparse_to_dense([[cid, 0]], [nclass, 1], 1.0)
                    #_out_act: reshape為(1, 1000)
                    self._out_act = tf.reshape(self._out_act, [1, nclass])
                    #class_act: 為一個(1, 1000) * (1000, 1)指定類別的單一激活值(type: tensor)
                    class_act = tf.matmul(self._out_act, one_hot)
                    #act_list: 為包有一個單一激活值的list
                    act_list.append(class_act)

            return act_list, tf.convert_to_tensor(class_list)

    def get_visualization(self, image):
        g = tf.get_default_graph()

        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            try:
                self._create_model(image)
            except ValueError:
                with tf.variable_scope(tf.get_variable_scope()) as scope:
                    scope.reuse_variables()
                    self._create_model(image)
            act_list, class_list = self._get_activation()

            with tf.name_scope('guided_back_pro_map'):
                guided_back_pro_list = []
                #act_list: 前5大類別的激活值(tensor)
                for class_act in act_list:
                    #guided_back_pro: 某類別的激活值對輸入圖片透過導向反向傳播求導
                    #返回值為和輸入一樣大的矩陣，分別為激活值對每個像素的導數
                    guided_back_pro = tf.gradients(
                        class_act, self._vis_model.layer['input'])
                    guided_back_pro_list.append(guided_back_pro)

                self.visual_map = guided_back_pro_list
                self.class_list = class_list
                return guided_back_pro_list, class_list
