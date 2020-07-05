import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')


class common():
    keypoints_symmetry = [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]
    rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)

    c3dpo_rot = np.array([ 0.74070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
    skeleton_parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    pairs = [(1,2), (5,4),(6,5),(8,7),(8,9),(10,1),(11,10),(12,11),(13,1),(14,13),(15,14),(16,2),(16,3),(16,4),(16,7)]

    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    # pad = (243 - 1) // 2 # Padding on each side
    pad = (27 - 1) // 2  # Padding on each side
    causal_shift = 0
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]]

    my_connections = ((1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (1, 7), (4, 10), (7, 10), (7, 8), (8, 9), (10, 11), (11, 12))


def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        if W>H:
            W_resize = max_length
            H_resize = int(H * max_length / W)
        else:
            H_resize = max_length
            W_resize = int(W * max_length / H)
        frame = cv2.resize(frame, (W_resize, H_resize), interpolation=cv2.INTER_AREA)
        return frame, W_resize, H_resize

    else:
        return frame, W, H


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def draw_2Dimg(img, kpt, display=None):
    # kpts : (17, 3)  3-->(x, y, score)
    im = img.copy()
    joint_pairs = common.joint_pairs
    for item in kpt:
        score = item[-1]
        if score > 0.1:
            x, y = int(item[0]), int(item[1])
            cv2.circle(im, (x, y), 1, (255, 5, 0), 5)
    for pair in joint_pairs:
        j, j_parent = pair
        pt1 = (int(kpt[j][0]), int(kpt[j][1]))
        pt2 = (int(kpt[j_parent][0]), int(kpt[j_parent][1]))
        cv2.line(im, pt1, pt2, (0,255,0), 2)

    if display:
        cv2.imshow('im', im)
        cv2.waitKey(3)
    return im

def draw_3Dimg(pos, image, display=None, kpt2D=None):
    from mpl_toolkits.mplot3d import Axes3D  # projection 3D 必须要这个
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)

    # 2D
    fig.add_subplot(121)
    if isinstance(kpt2D, np.ndarray):
        plt.imshow(draw_2Dimg(image, kpt2D))
    else:
        plt.imshow(image)

    # 3D
    ax = fig.add_subplot(122, projection='3d')
    radius = 1.7
    ax.view_init(elev=15., azim=70.)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    parents = common.skeleton_parents
    joints_right = common.joints_right

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in joints_right else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    if display:
        cv2.imshow('im', image)
        cv2.waitKey(3)

    return image


def draw_3Dimg_adjust(pos, image, display=None, kpt2D=None, shapes=None):
    from mpl_toolkits.mplot3d import Axes3D  # projection 3D 必须要这个
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)

    # 2D
    fig.add_subplot(131)
    if isinstance(kpt2D, np.ndarray):
        plt.imshow(draw_2Dimg(image, kpt2D))
    else:
        plt.imshow(image)

    # nrsfm 矫正
    # if shapes is not None:
    #     index_list = [0, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    #     pos[index_list] = shapes[-1]
    # 3D
    ax = fig.add_subplot(132, projection='3d')
    radius = 1.7
    ax.view_init(elev=15., azim=70.)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.dist = 7.5
    parents = common.skeleton_parents
    joints_right = common.joints_right

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in joints_right else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    # nrsfm
    bx = fig.add_subplot(133, projection='3d')
    bx.view_init(elev=15., azim=70.)
    bx.set_xlim3d([-radius / 2, radius / 2])
    bx.set_zlim3d([0, radius])
    bx.set_ylim3d([-radius / 2, radius / 2])
    bx.set_aspect('equal')
    # to_world
    from common.camera import camera_to_world
    shapes = np.array(shapes)
    shapes = shapes[:, :, [0, 2, 1]]
    shapes = camera_to_world(shapes, R=common.rot, t=0)
    shapes[:, :, 2] -= np.min(shapes[:, :, 2])
    one_shape = shapes[-1]
    # one_shape = one_shape[:, [0, 2, 1]]


    for j, j_parent in common.my_connections:
        bx.plot([one_shape[j, 0], one_shape[j_parent, 0]],
                [one_shape[j, 1], one_shape[j_parent, 1]],
                [one_shape[j, 2], one_shape[j_parent, 2]], zdir='z', c='black')

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    if display:
        cv2.imshow('im', image)
        cv2.waitKey(1)

    return image


def rebuild(list_a):
    """
        17 -> 13
    """
    index_list = [0, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    list_b = list_a[index_list]
    return list_b


def draw_3Dimg_c3dpo(pos, image, display=None, kpt2D=None, shape=None):
    from mpl_toolkits.mplot3d import Axes3D  # projection 3D 必须要这个
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)

    # 2D
    fig.add_subplot(131)
    if isinstance(kpt2D, np.ndarray):
        plt.imshow(draw_2Dimg(image, kpt2D))
    else:
        plt.imshow(image)

    # c3dpo 矫正
    # if shape is not None:
    #     index_list = [0, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
    #     pos[index_list] = shape
    # 3D
    ax = fig.add_subplot(132, projection='3d')
    radius = 1.7
    ax.view_init(elev=15., azim=70.)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.dist = 7.5
    parents = common.skeleton_parents
    joints_right = common.joints_right


    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in joints_right else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    # c3dpo
    bx = fig.add_subplot(133, projection='3d')
    bx.view_init(elev=15., azim=15.)
    bx.set_xlim3d([-radius / 2, radius / 2])
    bx.set_zlim3d([0, radius])
    bx.set_ylim3d([-radius / 2, radius / 2])
    bx.set_aspect('equal')
    from common.camera import camera_to_world
    shape = camera_to_world(shape, R=common.c3dpo_rot, t=0)
    shape[:, 2] -= np.min(shape[:, 2])
    order_pair = ((1, 2), (2, 3), (4, 5), (5, 6), (0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (11, 12), (12, 13), (14, 15),
                  (15, 16), (11, 8), (14, 8))
    # scale
    # shape *= 0.7
    for j, j_parent in order_pair:

        bx.plot([shape[j, 0], shape[j_parent, 0]],
                [shape[j, 1], shape[j_parent, 1]],
                [shape[j, 2], shape[j_parent, 2]], zdir='z', c='red')
    # bx.scatter(shape[:, 0], shape[:, 1], shape[:, 2])

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    if display:
        cv2.imshow('im', image)
        cv2.waitKey(1)

    return image


def evaluate(test_generator, model_pos, action=None, return_predictions=False):
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def videoInfo(VideoName):
    cap = cv2.VideoCapture(VideoName)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, length

def videopose_model_load():
    # load trained model
    from common.model import TemporalModel
    # chk_filename = main_path + '/checkpoint/cpn-pt-243.bin'
    chk_filename = main_path + '/checkpoint/ckpt_base.bin'
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)# 把loc映射到storage
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3] , causal=False, dropout=False, channels=1024, dense=False)
    model_pos = model_pos.cuda()
    model_pos.load_state_dict(checkpoint['model_pos'])
    receptive_field = model_pos.receptive_field()
    return model_pos

def interface(model_pos, keypoints, W, H):
    # input (N, 17, 2) return (N, 17, 3)
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    from common.camera import normalize_screen_coordinates_new, camera_to_world, normalize_screen_coordinates
    #  keypoints = normalize_screen_coordinates_new(keypoints[..., :2], w=W, h=H)
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    input_keypoints = keypoints.copy()
    # test_time_augmentation True
    from common.generators import UnchunkedGenerator
    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=common.pad, causal_shift=common.causal_shift, augment=True, kps_left=common.kps_left, kps_right=common.kps_right, joints_left=common.joints_left, joints_right=common.joints_right)
    prediction = evaluate(gen, model_pos, return_predictions=True)
    prediction = camera_to_world(prediction, R=common.rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction

def show_skeleton(inputs2d, filename=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([-0.6, 0.5])
    # if np.all(inputs2d[:, 1] < 0):
    #     inputs2d[:, 1] *= -1
    inputs2d[:, 1] *= -1
    # 17,2
    ax.scatter(inputs2d[:, 0], inputs2d[:, 1])
    for i in range(inputs2d.shape[0]):
        ax.text(inputs2d[i, 0], inputs2d[i, 1], i)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()