import os, time, cv2
import argparse
import math
import numpy as np
import csv
import torch
import torch.nn.functional as F
from visdom import Visdom
import scipy.signal as signal

from models.Detect.DBFaceSmallH import DBFace
from models.EmotionRecoginze import Resnet, aVIT

from uties import common, process

# tensorboard --logdir ./runs/128-0.00037-11.17.10.44
labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vs', '--video_source', default="..\\Data\\视频数据集\\207", help='runs path')
    parser.add_argument('-fn', '--file_name', default="2.mp4", help='runs path')
    parser.add_argument('-em', '--emotion_model', type=str, default="vit", help='Batch size.')
    parser.add_argument('-empt', '--emodel_pretrain', type=bool, default=True, help='Batch size.')
    parser.add_argument('-cn', '--class_num', type=int, default=7, help='Pytorch checkpoint file path')
    parser.add_argument('-s', '--image_size', type=int, default=128, help='Image size.')
    parser.add_argument('-ip', '--isplot', type=bool, default=True, help='Image size.')
    parser.add_argument('-is', '--isshow', type=bool, default=True, help='Image size.')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Image size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('-p', '--plot_cm', default=True, action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()


def detect(model, image, threshold=0.3, nms_iou=0.3):
    def nms(objs, iou=0.5):
        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):

            if flags[index] != 0:
                continue

            keep.append(obj)
            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and obj.iou(objs[j]) > iou:
                    flags[j] = 1
        return keep

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]

    torch_image = torch_image.to(args.device)

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def warp_affine(image, face, scale=1.0):
    eye_center = ((face.landmark[0][0] + face.landmark[1][0]) / 2.0,
                  (face.landmark[0][1] + face.landmark[1][1]) / 2.0)
    dy = face.landmark[1][1] - face.landmark[0][1]
    dx = face.landmark[1][0] - face.landmark[0][0]
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
    rot_img = cv2.warpAffine(image.copy(), rot, dsize=(image.shape[1], image.shape[0]))
    # cv2.imshow("align_face", rot_img)
    return rot_img


def draw_vis(outputs):
    vis_show.bar(X=outputs,
                 win='emotion',
                 opts=dict(
                     stacked=False,
                     rownames=['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"],
                     ymax=1.0,
                     ymin=1.0,
                     title='emotion',
                     ylabel='Confidence',  # y轴名称
                     xtickmin=0.4,  # x轴左端点起始位置
                     xtickstep=0.4,  # 每个柱形间隔距离
                     ytickmax=1.0
                 ))
    vis_show.line([outputs.cpu().detach().numpy()],
                  [cap.get(0) // 1000],
                  win=args.emotion_model + "-" + args.video_source,
                  opts=dict(title=args.emotion_model + "-" +args.video_source.split("视频数据集/")[-1]+"1",
                            ymax=1.0,
                            legend=['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]),
                  update='append')


def create_kalman():
    """Creates kalman filter."""
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    kalman = KalmanFilter(dim_x=7, dim_z=7)
    # kalman.Q = Q_discrete_white_noise(dim=7, dt=0.1, var=1e-2)  # 过程（系统）噪声
    return kalman


if __name__ == "__main__":
    args = parse_args()
    vis_show = Visdom(env='show')
    ### model
    # detect
    DBface = DBFace()
    DBface.eval().to(args.device)
    DBface.load("models/Checkpoint/dbfaceSmallH.pth")
    csv_buffer = []
    # analy
    if args.emotion_model == "resnet":
        Emodel = Resnet.Res18(pretrained=True, inplanes=224, num_classes=args.class_num, drop_rate=0)
        checkpoint = torch.load("models/models/Checkpoint/resnet-224.pth")
    else:
        Emodel = aVIT.ViT(
            image_size=128,
            patch_size=2,
            num_classes=7,
            dim=64,
            depth=3,
            heads=3,
            mlp_dim=16,
            dropout=0,
            emb_dropout=0
        )
        checkpoint = torch.load("models/Checkpoint/epoch95_acc0.8911.pth")
    if args.emodel_pretrain:
        Emodel.load_state_dict(checkpoint["model_state_dict"], strict=False)
    Emodel.to(args.device).eval()

    # cv
    # path = "..\\Data\\视频数据集"
    # for root, dirs, files in os.walk(path):
    # for file_name in files:
    # videopath = os.path.join(root, file_name)
    # print(videopath)
    dname,vname= args.video_source.split("视频数据集\\")[-1], args.file_name.split(".")[0]
    cap = cv2.VideoCapture(args.video_source+"\\"+ args.file_name)
    cap_width, cap_height = cap.get(4), cap.get(3)
    predicts_buff = None
    trace_x, trace_y = None, None
    emotion = "NE"
    box_scle = 1.2
    # Read first frame
    # while (frame := cap.read()[1]) is not None:
    frame = cap.read()[1]
    while (cap.read()[1]) is not None:
        faces = detect(DBface, frame)
        for face in faces:
            rot_img = warp_affine(frame, face, 1.0)
            [xcenter, ycenter] = face.center
            # trace location

            if not trace_x or not trace_y:
                trace_x, trace_y = xcenter, ycenter

            # if face.score >= 0.75 and (trace_check := abs(xcenter - trace_x) < 40 and abs(ycenter - trace_y) < 40):
            if face.score >= 0.75 and ( abs(xcenter - trace_x) < 40 and abs(ycenter - trace_y) < 40):
                trace_x, trace_y = xcenter, ycenter

                xy1 = (int(xcenter - face.width * box_scle / 2.0), int(ycenter - face.width * box_scle / 2.0))
                xy2 = (int(xcenter + face.width * box_scle / 2.0), int(ycenter + face.width * box_scle / 2.0))
                face_img = rot_img[xy1[1]:xy2[1], xy1[0]:xy2[0], :]  ##变换处理
                # face_img = frame[xy1[1]:xy2[1], xy1[0]:xy2[0], :]

                img_input, img_tensor = process.input_porocess(face_img, args.image_size, "cuda:0")
                outputs, _ = Emodel(img_input)
                predicts_buff = outputs if predicts_buff is None else torch.cat((predicts_buff, outputs), 0)

                banch_size = 20

                if cap.get(1) % banch_size == 0:
                    outputs = torch.mean(predicts_buff, dim=0)##均值滤波
                    # outputs,_ = torch.max(predicts_buff, dim=0) ##最大值滤波
                    csv_buffer.append(outputs.cpu().detach().numpy().tolist())
                    predicts_buff = None
                    draw_vis(outputs)
                    _, predicts = torch.max(outputs, 0)
                    emotion = labels_name[predicts]
                if args.isplot:
                    cv2.imshow("Face", face_img)
                    common.drawbbox(lable=emotion, image=frame, bbox=face)
        if args.isplot:
            cv2.imshow("demo DBFace", frame)
            # if (key := cv2.waitKey(1) & 0xFF) == ord('q'):
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break



    print(args.video_source)
    if True:
        if not os.path.exists('outputs\\{}'.format(dname)):
            os.makedirs('outputs\\{}'.format(dname))
        with open('outputs\\{}\\{}.csv'.format(dname,vname.split(".")[0]), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"])
            writer.writerows(csv_buffer)
    csv_buffer = []
    cap.release()
    cv2.destroyAllWindows()
