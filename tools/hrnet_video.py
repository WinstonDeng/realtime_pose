'''
通过最新的deep-high-resolution作为2D关键点的获取, 实现高精度的端到端3D姿态重建
'''

import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')
sys.path.insert(0, main_path)

import numpy as np
import ipdb;pdb = ipdb.set_trace

from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import errno

from common.camera import *
from common.model import *
from common.simple_baseline import SingleFrameBaseline
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
import time
from tools.utils import show_skeleton
from tools.BankUtils import *

metadata={'layout_name': 'coco','num_joints': 17,'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]}
h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}
# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()

time0 = ckpt_time()

class skeleton():
    def parents(self):
        return np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    def joints_right(self):
        # return [1, 2, 3, 9, 10]
        return [1, 2, 3, 14, 15, 16]


def evaluate(test_generator, model_pos, action=None, return_predictions=False, long_bank=None, direction_mode='greedy'):

    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()

        N = 0

        for _, batch, batch_2d in test_generator.next_epoch():

            # Positional model
            if long_bank is None:
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = []

                for i in range(batch_2d.shape[1]-26):
                    short_term_2d = batch_2d[:, i:i+27, :, :]
                    short_term_rebuilded = np.zeros(short_term_2d.shape)
                    # for j in range(short_term_rebuilded.shape[1]):
                    #     short_term_rebuilded[:, j] = short_term_2d[:, 13]
                    # save_size = 8
                    # short_term_rebuilded[:, :save_size] = short_term_2d[:, short_term_2d.shape[1] // 2 - save_size:short_term_2d.shape[1] // 2].copy()
                    # short_term_rebuilded[:, -save_size:] = short_term_2d[:, short_term_2d.shape[1] // 2 + 1:short_term_2d.shape[1] // 2 + 1 + save_size].copy()
                    if direction_mode == 'simple':
                        short_term_rebuilded[0] = rebuild_short(batch_2d[0, i:i+27, :, :], long_bank, direction_mode=direction_mode, use_shift=True)
                    # elif direction_mode == 'optional':
                    #     short_term_rebuilded[0] = rebuild_short_combine(batch_2d[0, i:i + 27, :, :], long_bank,
                    #                                             use_shift=True)
                    # elif direction_mode == 'pyramid':
                    #     short_term_rebuilded[0] = rebuild_short_pyramid(batch_2d[0, i:i + 27, :, :], long_bank)

                    if test_generator.augment_enabled():
                        if direction_mode == 'simple':
                            short_term_rebuilded[1] = rebuild_short(batch_2d[1, i:i+27, :, :], long_bank, direction_mode=direction_mode, use_shift=True)
                        # elif direction_mode == 'optional':
                        #     short_term_rebuilded[1] = rebuild_short_combine(batch_2d[1, i:i + 27, :, :], long_bank,
                        #                                             use_shift=True)
                        # elif direction_mode == 'pyramid':
                        #     short_term_rebuilded[1] = rebuild_short_pyramid(batch_2d[1, i:i + 27, :, :], long_bank)

                    short_term_rebuilded = torch.from_numpy(short_term_rebuilded.astype('float32'))
                    if torch.cuda.is_available():
                        short_term_rebuilded = short_term_rebuilded.cuda()
                    predicted = model_pos(short_term_rebuilded)
                    predicted_3d_pos.append(predicted)
                predicted_3d_pos = torch.cat(predicted_3d_pos,1)
            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def show_bank(full_keypoints):
    long_bank = LongBank(bank_size=80, threshold=0.9)
    for i in range(full_keypoints.shape[0]):
        bank_item = BankItem(full_keypoints[i, :, :], i, y_flip=False)
        long_bank.insert_item(bank_item)
    t = []
    bank = long_bank.get_bank(sort=False)
    for j in range(len(bank)):
        # show_skeleton(bank[j].points2d, 'path/bank_lindan/{:d}.jpg'.format(bank[j].origin_index))
        t.append(bank[j].origin_index)
    print("bank size:", len(t))
    print(t)
    # sys.exit(1)
    return bank

def rebuild_short(short_term_2d, long_term, simi_threshold=0, direction_mode = 'simple', center_index=-1, use_shift=False):

    # 1. get useful item from long_term bank by similarity with mid of short
    useful_list_ = []
    useful_list = []
    short_term = short_term_2d.copy()

    if direction_mode == 'optional':
        op_flow = OptionalFlow()
        op_flow_list = op_flow.get_optional_flow_list(short_term)
        avg_bone_length_list = []
        for i in range(short_term.shape[0]):
            avg_bone_length_list.append(get_avg_bone_length(short_term[i]))
        for i in range(len(long_term)):
            useful_list.append(np.array([long_term[i], 0]))
            # for j in range(short_term.shape[0]):
            #     simi_rate = PositionSimilarityUtils(long_term[i].points2d, short_term[j]).get_similarity()
            #     if simi_rate > 0.9:
            #         useful_list.append(np.array([long_term[i], simi_rate]))
            #         break

    if direction_mode == 'simple':
        simi_short_list = []
        for i in range(short_term.shape[0]):
            simi_short_list.append(PositionSimilarityUtils(short_term[short_term.shape[0]//2], short_term[i]).get_similarity())
        for i in range(len(long_term)):
            simi_rate = PositionSimilarityUtils(short_term[short_term.shape[0]//2, :, :], long_term[i].points2d).get_similarity()
            if simi_rate >= simi_threshold:
                # print("index:", long_term[i].origin_index, "simi_rate:", simi_rate)
                useful_list_.append(np.array([long_term[i], simi_rate]))
        # sort by simi_rate
        rate_list = np.ones(len(useful_list_))
        for i in range(rate_list.shape[0]):
            rate_list[i] = useful_list_[i][1]
        sort_index = (-rate_list[:]).argsort()
        for i in range(sort_index.shape[0]):
            useful_list.append(useful_list_[sort_index[i]])
        # print("useful list:", len(useful_list_))
    if center_index >= 0:
        print("help for index:", center_index)


    insert_cnt = 0
    for i in range(len(useful_list)):
        # if insert_cnt > 13:
        #     break
        # 2. get direction
        if direction_mode == 'greedy':
            skip_flag = False
            for j in range(short_term.shape[0]//2, 0, -1):
                # find inserting position
                if greedy_direction(short_term[j-1], useful_list[i][0].points2d, short_term[j]):
                    translate = useful_list[i][0].points2d[7, :] - short_term[j-1, 7, :]
                    # move left
                    # print('left',  useful_list[i][0].origin_index, ' insert after', center_index-13+j-1)
                    for k in range(j-1):
                        short_term[k] = short_term[k + 1]
                    short_term[j-1] = useful_list[i][0].points2d - translate
                    skip_flag = True
                    break
            if not skip_flag:
                for j in range(short_term.shape[0]//2, short_term.shape[0] - 2, 1):
                    if greedy_direction(short_term[j], useful_list[i][0].points2d, short_term[j + 1]):
                        # move right
                        # print('right', useful_list[i][0].origin_index, ' insert after', center_index-13+j)
                        translate = useful_list[i][0].points2d[7, :] - short_term[j + 1, 7, :]
                        for k in range(short_term.shape[0] - 1, j + 1, -1):
                            short_term[k] = short_term[k - 1]
                        short_term[j + 1] = useful_list[i][0].points2d - translate
                        break
        elif direction_mode == 'optional':
            # if useful_list[i][1] < 0.6:
            #     continue
            skip_flag = True
            for j in range(short_term.shape[0] // 2, 0, -1):
                # for j in range(left_skip_index, 2, -1):
                # if left_skip_index is not -1 and j == left_skip_index:
                #     continue
                # translate = useful_list[i][0].points2d[7, :] - short_term[j, 7, :]
                translate = useful_list[i][0].points2d[7, :] - (short_term[j, 7, :] + short_term[j - 1, 7, :]) / 2
                # translate = 0
                insert_2d = useful_list[i][0].points2d - translate
                vel = op_flow.get_velocity(short_term[j - 1], insert_2d)
                # print(j)
                if op_flow.judge_direction(op_flow_list[j], vel, short_term[j - 1], insert_2d, short_term[j],
                                           avg_bone_length_list[j - 1]):
                    # if op_flow.judge_direction(op_flow_list[j], vel, avg_bone_length):
                    for k in range(j - 1):
                        short_term[k] = short_term[k + 1]
                        op_flow_list[k] = op_flow_list[k + 1]
                        avg_bone_length_list[k] = avg_bone_length_list[k + 1]
                    if center_index > 0:
                        print(useful_list[i][0].origin_index, 'left insert', center_index - 13 + j - 1)
                    short_term[j - 1] = insert_2d
                    op_flow_list[j - 1] = vel
                    avg_bone_length_list[j - 1] = get_avg_bone_length(insert_2d)
                    left_skip_index = j - 1
                    insert_cnt += 1
                    # skip_flag = False
                    break
                # if j == 1:
                #     import sys
                #     sys.exit(1)
            if skip_flag:
                for j in range(short_term.shape[0] // 2, short_term.shape[0] - 1, 1):
                    # for j in range(right_skip_index, short_term.shape[0] - 1, 1):
                    #     translate = useful_list[i][0].points2d[7, :] - short_term[j + 1, 7, :]
                    translate = useful_list[i][0].points2d[7, :] - (short_term[j, 7, :] + short_term[j + 1, 7, :]) / 2
                    # translate = 0
                    insert_2d = useful_list[i][0].points2d - translate
                    vel = op_flow.get_velocity(short_term[j], insert_2d)
                    # print(j)
                    if op_flow.judge_direction(op_flow_list[j + 1], vel, short_term[j], insert_2d, short_term[j + 1],
                                               avg_bone_length_list[j]):
                        # if op_flow.judge_direction(op_flow_list[j + 1], vel, avg_bone_length):
                        for k in range(short_term.shape[0] - 1, j + 1, -1):
                            short_term[k] = short_term[k - 1]
                            op_flow_list[k] = op_flow_list[k - 1]
                            avg_bone_length_list[k] = avg_bone_length_list[k - 1]
                        if center_index > 0:
                            print(useful_list[i][0].origin_index, 'right insert', center_index - 13 + j + 1)
                        short_term[j + 1] = insert_2d
                        op_flow_list[j + 1] = vel
                        avg_bone_length_list[j + 1] = get_avg_bone_length(insert_2d)
                        insert_cnt += 1
                        right_skip_index = j + 1
                        break
        else:
            if direction_mode == 'simple':
                direction = 'left' if i % 2 == 0 else 'right'
            else:
                direction = get_insert_direction(short_term[short_term.shape[0]//2-1:short_term.shape[0]//2+2], useful_list[i][0].points2d)

            # 3. get insert position and insert
            simi_bank_item = useful_list[i][1]
            # print("insert bank iterm:", useful_list[i][0].origin_index, "direction:", direction, "simi_bank:", simi_bank_item)
            if direction == 'left':
            # if direction_mode == 'simple':
                for j in range(short_term.shape[0] // 2 - 2, -1, -1):
                    # simi_short_item = PositionSimilarityUtils(short_term[short_term.shape[0]//2, :, :],
                    #                                           short_term[j, :, :]).get_similarity()
                    # print("left index:", j, "simi_rate:", simi_short_item)
                    if simi_bank_item > simi_short_list[j]:
                        # insert
                        if center_index>0:
                            print("useful index", useful_list[i][0].origin_index, "left update index:", center_index-13+j)
                        # print("bank:", simi_bank_item, " short:", simi_short_item)

                        translate = useful_list[i][0].points2d[7, :] - short_term[j, 7, :]
                        if use_shift:
                            for k in range(j):
                                short_term[k, :, :] = short_term[k + 1, :, :]
                                simi_short_list[k] = simi_short_list[k+1]
                        short_term[j, :, :] = useful_list[i][0].points2d - translate
                        simi_short_list[j] = simi_bank_item
                        insert_cnt += 1
                        break
            elif direction == 'right':
            # if direction_mode == 'simple':
                for j in range(short_term.shape[0] // 2 + 2, short_term.shape[0], 1):
                    # simi_short_item = PositionSimilarityUtils(short_term[short_term.shape[0] // 2, :, :],
                    #                                           short_term[j, :, :]).get_similarity()
                    # print("right index:", j, "simi_rate:", simi_short_item)
                    if simi_bank_item > simi_short_list[j]:
                        # insert
                        if center_index>0:
                            print("useful index", useful_list[i][0].origin_index, "right update index:", center_index-13+j)
                        translate = useful_list[i][0].points2d[7, :] - short_term[j, 7, :]
                        if use_shift:
                            for k in range(short_term.shape[0] - 1, j, -1):
                                short_term[k, :, :] = short_term[k - 1, :, :]
                                simi_short_list[k]=simi_short_list[k-1]
                        short_term[j, :, :] = useful_list[i][0].points2d - translate
                        simi_short_list[j]=simi_bank_item
                        insert_cnt += 1
                        break
    # if center_index == 1559:
    #     import sys
    #     sys.exit(1)
    return short_term

def rebuild_short_combine(short_term_2d, long_term, simi_threshold=0, center_index=-1, use_shift=False):
    useful_list_ = []
    useful_list = []

    short_term = short_term_2d.copy()
    # center item simi
    simi_short_list = []
    for i in range(short_term.shape[0]):
        simi_short_list.append(
            PositionSimilarityUtils(short_term[short_term.shape[0] // 2], short_term[i]).get_similarity())
    for i in range(len(long_term)):
        simi_rate = PositionSimilarityUtils(short_term[short_term.shape[0] // 2, :, :],
                                            long_term[i].points2d).get_similarity()
        if simi_rate >= simi_threshold:
            # print("index:", long_term[i].origin_index, "simi_rate:", simi_rate)
            useful_list_.append(np.array([long_term[i], simi_rate]))

    # sort by center_simi_rate
    rate_list = np.ones(len(useful_list_))
    for i in range(rate_list.shape[0]):
        rate_list[i] = useful_list_[i][1]
    sort_index = (-rate_list[:]).argsort()
    for i in range(sort_index.shape[0]):
        useful_list.append(useful_list_[sort_index[i]])

    if center_index >= 0:
        print("help for index:", center_index)

    insert_cnt = 0
    for i in range(len(useful_list)):
        if insert_cnt > 13:
            break
        # center simi -> simple
        direction = 'left' if i % 2 == 0 else 'right'

        # 3. get insert position and insert
        simi_bank_item = useful_list[i][1]
        # print("insert bank iterm:", useful_list[i][0].origin_index, "direction:", direction, "simi_bank:", simi_bank_item)
        if direction == 'left':
            # if direction_mode == 'simple':
            for j in range(short_term.shape[0] // 2 - 2, 6, -1):
                # simi_short_item = PositionSimilarityUtils(short_term[short_term.shape[0]//2, :, :],
                #                                           short_term[j, :, :]).get_similarity()
                # print("left index:", j, "simi_rate:", simi_short_item)
                if simi_bank_item > simi_short_list[j]:
                    # insert
                    if center_index > 0:
                        print("useful index", useful_list[i][0].origin_index, "left update index:",
                              center_index - 13 + j)
                    # print("bank:", simi_bank_item, " short:", simi_short_item)

                    translate = useful_list[i][0].points2d[7, :] - short_term[j, 7, :]
                    if use_shift:
                        for k in range(j):
                            short_term[k, :, :] = short_term[k + 1, :, :]
                            simi_short_list[k] = simi_short_list[k + 1]
                    short_term[j, :, :] = useful_list[i][0].points2d - translate
                    simi_short_list[j] = simi_bank_item
                    insert_cnt += 1
                    break
        elif direction == 'right':
            # if direction_mode == 'simple':
            for j in range(short_term.shape[0] // 2 + 2, short_term.shape[0]-7, 1):
                # simi_short_item = PositionSimilarityUtils(short_term[short_term.shape[0] // 2, :, :],
                #                                           short_term[j, :, :]).get_similarity()
                # print("right index:", j, "simi_rate:", simi_short_item)
                if simi_bank_item > simi_short_list[j]:
                    # insert
                    if center_index > 0:
                        print("useful index", useful_list[i][0].origin_index, "right update index:",
                              center_index - 13 + j)
                    translate = useful_list[i][0].points2d[7, :] - short_term[j, 7, :]
                    if use_shift:
                        for k in range(short_term.shape[0] - 1, j, -1):
                            short_term[k, :, :] = short_term[k - 1, :, :]
                            simi_short_list[k] = simi_short_list[k - 1]
                    short_term[j, :, :] = useful_list[i][0].points2d - translate
                    simi_short_list[j] = simi_bank_item
                    insert_cnt += 1
                    break

    op_flow = OptionalFlow()
    op_flow_list = op_flow.get_optional_flow_list(short_term)
    avg_bone_length_list = []
    useful_list_side = []
    for i in range(short_term.shape[0]):
        avg_bone_length_list.append(get_avg_bone_length(short_term[i]))
    for i in range(len(long_term)):
        for j in range(0, short_term.shape[0]//2-6):
            simi_rate = PositionSimilarityUtils(long_term[i].points2d, short_term[j]).get_similarity()
            if simi_rate > 0.9:
                useful_list_side.append(np.array([long_term[i], simi_rate]))
                break
        for j in range(short_term.shape[0]//2+7, short_term.shape[0]):
            simi_rate = PositionSimilarityUtils(long_term[i].points2d, short_term[j]).get_similarity()
            if simi_rate > 0.9:
                useful_list_side.append(np.array([long_term[i], simi_rate]))
                break

    for i in range(len(useful_list_side)):
        # side simi -> optional
        # if useful_list[i][1] < 0.6:
        #     continue
        skip_flag = True
        for j in range(short_term.shape[0] // 2-7, 2, -1):
            translate = useful_list_side[i][0].points2d[7, :] - short_term[j, 7, :]
            insert_2d = useful_list_side[i][0].points2d - translate
            vel = op_flow.get_velocity(short_term[j - 1], insert_2d)
            if op_flow.judge_direction(op_flow_list[j], vel, avg_bone_length_list[j - 1]):
                # if op_flow.judge_direction(op_flow_list[j], vel, avg_bone_length):
                for k in range(j - 1):
                    short_term[k] = short_term[k + 1]
                    op_flow_list[k] = op_flow_list[k + 1]
                    avg_bone_length_list[k] = avg_bone_length_list[k + 1]
                if center_index > 0:
                    print(useful_list_side[i][0].origin_index, 'left insert', center_index - 13 + j - 1)
                short_term[j - 1] = insert_2d
                op_flow_list[j - 1] = vel
                avg_bone_length_list[j - 1] = get_avg_bone_length(insert_2d)
                insert_cnt += 1
                # skip_flag = False
                break
        if skip_flag:
            for j in range(short_term.shape[0] // 2+7, short_term.shape[0] - 1, 1):
                translate = useful_list_side[i][0].points2d[7, :] - short_term[j + 1, 7, :]
                insert_2d = useful_list_side[i][0].points2d - translate
                vel = op_flow.get_velocity(short_term[j], insert_2d)
                if op_flow.judge_direction(op_flow_list[j + 1], vel, avg_bone_length_list[j]):
                    # if op_flow.judge_direction(op_flow_list[j + 1], vel, avg_bone_length):
                    for k in range(short_term.shape[0] - 1, j + 1, -1):
                        short_term[k] = short_term[k - 1]
                        op_flow_list[k] = op_flow_list[k - 1]
                        avg_bone_length_list[k] = avg_bone_length_list[k - 1]
                    if center_index > 0:
                        print(useful_list_side[i][0].origin_index, 'right insert', center_index - 13 + j + 1)
                    short_term[j + 1] = insert_2d
                    op_flow_list[j + 1] = vel
                    avg_bone_length_list[j + 1] = get_avg_bone_length(insert_2d)
                    insert_cnt += 1
                    break


    return short_term

def rebuild_short_pyramid(short_term_2d, long_term, simi_threshold=0, center_index=-1):
    short_term = short_term_2d.copy()
    # 对中心9帧，每帧都找两个最相似的，如果有就插入，没有就默认原来的临近帧相似，不插入，潜在的问题是找到的相似帧不如原来的
    short_term_out = np.zeros(short_term.shape)
    map_list = np.array([[1, 9], [4, 10], [7, 11], [10, 12], [13, 13], [16, 14], [19, 15], [22, 16], [25, 17]])
    for i in range(map_list.shape[0]):
        short_term_out[map_list[i, 0]] = short_term[map_list[i, 1]]
        useful_list_ = []
        useful_list = []
        simi_left = PositionSimilarityUtils(short_term[map_list[i, 1]], short_term[map_list[i, 1]-1]).get_similarity()
        simi_right = PositionSimilarityUtils(short_term[map_list[i, 1]], short_term[map_list[i, 1] + 1]).get_similarity()
        for j in range(len(long_term)):
            simi_rate = PositionSimilarityUtils(short_term[map_list[i, 1]],
                                            long_term[j].points2d).get_similarity()
            if simi_rate > simi_left or simi_rate > simi_right:
                # print("index:", long_term[i].origin_index, "simi_rate:", simi_rate)
                useful_list_.append(np.array([long_term[j], simi_rate]))
        # sort by center_simi_rate
        rate_list = np.ones(len(useful_list_))
        for j in range(rate_list.shape[0]):
            rate_list[j] = useful_list_[j][1]
        sort_index = (-rate_list[:]).argsort()
        for j in range(sort_index.shape[0]):
            useful_list.append(useful_list_[sort_index[j]])
        if len(useful_list) >= 2:
            translate = useful_list[0][0].points2d[7, :] - short_term[map_list[i, 1]-1, 7, :]
            short_term_out[map_list[i, 0]-1] = useful_list[0][0].points2d - translate
            translate = useful_list[1][0].points2d[7, :] - short_term[map_list[i, 1] + 1, 7, :]
            short_term_out[map_list[i, 0]+1] = useful_list[1][0].points2d - translate
        elif len(useful_list) == 1:
            if useful_list[0][1] > simi_left:
                translate = useful_list[0][0].points2d[7, :] - short_term[map_list[i, 1] - 1, 7, :]
                short_term_out[map_list[i, 0]-1] = useful_list[0][0].points2d - translate
                short_term_out[map_list[i, 0]+1] = short_term[map_list[i, 1]+1]
            elif useful_list[0][1] > simi_right:
                translate = useful_list[0][0].points2d[7, :] - short_term[map_list[i, 1] + 1, 7, :]
                short_term_out[map_list[i, 0] + 1] = useful_list[0][0].points2d - translate
                short_term_out[map_list[i, 0] - 1] = short_term[map_list[i, 1] - 1]
        else:
            short_term_out[map_list[i, 0]-1] = short_term[map_list[i, 1]-1]
            short_term_out[map_list[i, 0] + 1] = short_term[map_list[i, 1] + 1]

    return short_term_out

def main():
    args = parse_args()

    # 2D kpts loads or generate
    if not args.input_npz:
        # 通过hrnet生成关键点
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts
        video_name = args.viz_video
        print('generat keypoints by hrnet...')
        keypoints = generate_kpts(video_name)

        np.savez(video_name.split('.')[-2],kpts=keypoints)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts'] #(N, 17, 2)
    from tools.utils import show_skeleton,draw_2Dimg
    keypoints = coco2h36m(keypoints)

    # keypoints_symmetry = metadata['keypoints_symmetry']
    keypoints_symmetry = h36m_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  假设use the camera parameter
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    # for i in range(keypoints.shape[0]):
        #show_skeleton(keypoints[i], 'path/final/{:d}.jpg'.format(i))
    # show_skeleton(keypoints[87], 'path/final/{:d}.jpg'.format(87))
    # show_skeleton(keypoints[89], 'path/final/{:d}.jpg'.format(89))
    # print(PositionSimilarityUtils(keypoints[89], keypoints[87]).get_similarity())
    # sys.exit(1)
    # for i in range(100, 127):
    #     print(PositionSimilarityUtils(keypoints[i], keypoints[113]).get_similarity())
    # print("468 323", PositionSimilarityUtils(keypoints[468], keypoints[323]).get_similarity())
    # short_term = keypoints[100:127, :, :]
    # op_flow = OptionalFlow()
    # op_flow_list = op_flow.get_optional_flow_list(short_term)
    # for i in range(2, short_term.shape[0]-2, 1):
    #     # translate = short_term[i+2, 7, :] - short_term[i+1, 7, :]
    #     # temp = short_term[i+2] - translate
    #     # vel = op_flow.get_velocity(short_term[i], temp, 1)
    #     avg_bone_length = get_avg_bone_length(short_term[i])
    #     # print(i, "dire", op_flow.judge_direction(vel, op_flow_list[i+1], avg_bone_length))
    #     print(i, "dire", op_flow.judge_direction(op_flow_list[i], op_flow_list[i + 1], avg_bone_length))
    # sys.exit(1)
    # bank
    bank = show_bank(keypoints.copy())
    # short_term = keypoints[408:435, :, :]
    # short_term = rebuild_short(short_term, bank,center_index=421, use_shift=True)
    # sys.exit(1)
    # model_pos = TemporalModel(17, 2, 17,filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
    #                             dense=args.dense)
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3], causal=args.causal, dropout=args.dropout,
                              channels=args.channels, dense=args.dense)
    model_pos_crop = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3], causal=args.causal, dropout=args.dropout,
                              channels=args.channels, dense=args.dense)
    # model_pos_sf = SingleFrameBaseline(17, 2, 17, dropout=args.dropout)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        model_pos_crop = model_pos_crop.cuda()
        # model_pos_sf = model_pos_sf.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('------- load data spends {:.2f} seconds'.format(ckpt))


    # load trained model
    # chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # print('Loading checkpoint', os.path.join(main_path,chk_filename))
    # checkpoint = torch.load(os.path.join(main_path,chk_filename), map_location=lambda storage, loc: storage)# 把loc映射到storage
    #coco_crop6_base  ckpt_coco_27
    checkpoint = torch.load('checkpoint/ckpt_base.bin',
                            map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])
    checkpoint_crop = torch.load('checkpoint/ckpt_crop5.bin',
                            map_location=lambda storage, loc: storage)  # 把loc映射到storage
    model_pos_crop.load_state_dict(checkpoint_crop['model_pos'])

    ckpt, time2 = ckpt_time(time1)
    print('------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    # receptive_field = 27
    pad = (receptive_field - 1) // 2 # Padding on each side
    # pad = 0
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    prediction = evaluate(gen, model_pos, return_predictions=True)
    # prediction_crop = evaluate(gen, model_pos_crop, return_predictions=True)
    prediction_bank = evaluate(gen, model_pos, return_predictions=True, long_bank=bank, direction_mode='simple')
    # prediction_bank_optional = evaluate(gen, model_pos, return_predictions=True, long_bank=bank, direction_mode='optional')

    # sys.exit(1)
    # gen_sf = UnchunkedGenerator(None, None, [input_keypoints],
    #                          pad=0, causal_shift=causal_shift, augment=args.test_time_augmentation,
    #                          kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    # prediction_sf = evaluate(gen_sf, model_pos_sf, return_predictions=True)

    # rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype=np.float32)
    # t = np.array([1841.1070556640625, 4955.28466796875, 1563.4454345703125], dtype=np.float32)
    t=0
    prediction = camera_to_world(prediction, R=rot, t=t)
    # prediction_crop = camera_to_world(prediction_crop, R=rot, t=t)
    prediction_bank = camera_to_world(prediction_bank, R=rot, t=t)
    # prediction_bank_optional = camera_to_world(prediction_bank_optional, R=rot, t=t)
    # prediction_sf = camera_to_world(prediction_sf, R=rot, t=t)
    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    # prediction_crop[:, :, 2] -= np.min(prediction_crop[:, :, 2])
    prediction_bank[:, :, 2] -= np.min(prediction_bank[:, :, 2])
    # prediction_bank_optional[:, :, 2] -= np.min(prediction_bank_optional[:, :, 2])
    # prediction_sf[:, :, 2] -= np.min(prediction_sf[:, :, 2])

    anim_output = {'VideoPose': prediction}
    # anim_output = {'Ours': prediction_bank}
    anim_output['Ours'] = prediction_bank
    # anim_output['bank_optional'] = prediction_bank_optional
    # anim_output['crop'] = prediction_crop
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))
    # sys.exit(1)
    if not args.viz_output:
        args.viz_output = 'outputs/hrnet_result.mp4'

    from common.visualization import render_animation, render_animation_origin, render_animation_diff, render_animation_diff_
    # render_animation(input_keypoints,  h36m_metadata, anim_output,
    #                         skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
    #                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    #                         input_video_path=args.viz_video, viewport=(1000, 1002),
    #                         input_video_skip=args.viz_skip)
    render_animation_origin(input_keypoints, h36m_metadata, anim_output,
                        skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                        limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                        input_video_path=args.viz_video, viewport=(1000, 1002),
                        input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    openVideoCommand="xdg-open " + args.viz_output
    os.system(openVideoCommand)
    print('total spend {:2f} second'.format(ckpt))


def coco2h36m(coco_2d):
    out_2d = []
    for i in range(coco_2d.shape[0]):
        frame_2d = []
        temp = coco_2d[i, :, :]
        frame_2d.append(np.expand_dims((temp[11, :]+temp[12, :])/2, 0))
        frame_2d.append(np.expand_dims(temp[12, :], 0))
        frame_2d.append(np.expand_dims(temp[14, :], 0))
        frame_2d.append(np.expand_dims(temp[16, :], 0))
        frame_2d.append(np.expand_dims(temp[11, :], 0))
        frame_2d.append(np.expand_dims(temp[13, :], 0))
        frame_2d.append(np.expand_dims(temp[15, :], 0))
        frame_2d.append(np.expand_dims(((temp[11, :]+temp[12, :])/2+(temp[5, :]+temp[6, :])/2)/2, 0))
        frame_2d.append(np.expand_dims((temp[5, :]+temp[6, :])/2, 0))
        frame_2d.append(np.expand_dims(temp[0, :], 0))
        frame_2d.append(np.expand_dims((temp[1, :]+temp[2, :])/2, 0))
        frame_2d.append(np.expand_dims(temp[5, :], 0))
        frame_2d.append(np.expand_dims(temp[7, :], 0))
        frame_2d.append(np.expand_dims(temp[9, :], 0))
        frame_2d.append(np.expand_dims(temp[6, :], 0))
        frame_2d.append(np.expand_dims(temp[8, :], 0))
        frame_2d.append(np.expand_dims(temp[10, :], 0))
        out_2d.append(np.expand_dims(np.concatenate(frame_2d, 0), 0))
    return np.concatenate(out_2d, 0)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
