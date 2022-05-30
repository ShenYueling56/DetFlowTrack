import mayavi.mlab as mlab
import numpy as np
import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3],
                          scale_mode="none", colormap="copper", scale_factor=0.15, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2],
                          colormap='gnuplot', scale_factor=0.15, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig

def draw_two_scenes(points, ref_points=None, gt_boxes=None, ref_boxes=None):
    if not isinstance(points, np.ndarray):
        points = points.detach().cpu().numpy()
    if ref_points is not None and not isinstance(ref_points, np.ndarray):
        ref_points = ref_points.detach().cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.detach().cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))

    if ref_points is not None:
        mlab.points3d(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], mode='point',
                          color=(1, 0, 0), colormap='gnuplot', scale_factor=1, figure=fig)

    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100) #蓝色

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), max_num=100) #绿色
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig

def draw_three_scenes(points, points2=None, points3=None, boxes1=None, boxes2=None, boxes3=None, label=None, label2=None):
    if not isinstance(points, np.ndarray):
        points = points.detach().cpu().numpy()
    if points2 is not None and not isinstance(points2, np.ndarray):
        points2 = points2.detach().cpu().numpy()
    if points3 is not None and not isinstance(points3, np.ndarray):
        points3 = points3.detach().cpu().numpy()
    if boxes1 is not None and not isinstance(boxes1, np.ndarray):
        boxes1 = boxes1.detach().cpu().numpy()
    if boxes2 is not None and not isinstance(boxes2, np.ndarray):
        boxes2 = boxes2.detach().cpu().numpy()
    if boxes3 is not None and not isinstance(boxes3, np.ndarray):
        boxes3 = boxes3.detach().cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))

    if points2 is not None:
        mlab.points3d(points2[:, 0], points2[:, 1], points2[:, 2],
                          color=(1, 0, 0), colormap='gnuplot', scale_factor=0.15, figure=fig) # 红色

    if points3 is not None:
        mlab.points3d(points3[:, 0], points3[:, 1], points3[:, 2],
                          color=(0, 1, 0), colormap='gnuplot', scale_factor=0.15, figure=fig) # 绿色
        # mlab.points3d(points3[:, 0], points3[:, 1], points3[:, 2], points3[:, 3],
        #               scale_mode = "none", colormap = "copper", scale_factor=0.3, figure=fig)  # 每个点指定颜色

    if boxes1 is not None:
        if boxes1.shape[1] > 7:
            for i in range(boxes1.shape[0]):
                score = str(boxes1[i, 7])
                mlab.text3d(boxes1[i, 0], boxes1[i, 1], boxes1[i, 2], score, scale=1, color=(1, 0, 0), figure=fig,
                )
        corners3d = boxes_to_corners_3d(boxes1[:, 0:7])
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100) #蓝色框,第一帧检测框

    if boxes2 is not None and len(boxes2) > 0:
        if boxes2.shape[1] > 7:
            for i in range(boxes2.shape[0]):
                score = str(boxes2[i, 7])
                mlab.text3d(boxes2[i, 0], boxes2[i, 1], boxes2[i, 2], score, scale=1, color=(1, 0, 0), figure=fig,
                )
        corners3d2 = boxes_to_corners_3d(boxes2)
        fig = draw_corners3d(corners3d2, fig=fig, color=(0, 1, 0), max_num=100) #绿色框,第二帧检测框

    if boxes3 is not None and len(boxes3) > 0:
        if boxes3.shape[1] > 7:
            for i in range(boxes3.shape[0]):
                score = str(boxes3[i, 7])
                mlab.text3d(boxes3[i, 0], boxes3[i, 1], boxes3[i, 2], score, scale=1, color=(1, 0, 0), figure=fig,
                )
        corners3d3 = boxes_to_corners_3d(boxes3)
        fig = draw_corners3d(corners3d3, fig=fig, color=(1, 0, 0), max_num=100) #红色框,检测真实值

    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig

def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig
