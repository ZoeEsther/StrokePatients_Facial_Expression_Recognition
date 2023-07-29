import os, time, cv2
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.Detect.DBFaceSmallH import DBFace
from models.EmotionRecoginze import Resnet,aVIT

from uties import common,process
# tensorboard --logdir ./runs/128-0.00037-11.17.10.44
device = 'cuda:0'
labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]

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


def detect(model, image, threshold=0.3, nms_iou=0.3):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]

    torch_image = torch_image.to(device)

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
    eye_center = ((face.landmark[0][0] + face.landmark[1][0]) / 2,
                  (face.landmark[0][1] + face.landmark[1][1]) / 2)
    dy = face.landmark[1][1] - face.landmark[0][1]
    dx = face.landmark[1][0] - face.landmark[0][0]
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
    rot_img = cv2.warpAffine(image.copy(), rot, dsize=(image.shape[1], image.shape[0]))
    # cv2.imshow("align_face", rot_img)
    return rot_img



if __name__ == "__main__":
    filepath = "pain1"
    ### log
    curtime = time.strftime("%m.%d.%H.%M")
    runPath = 'runs/'+ filepath
    print("tbPath:"+runPath)
    bwriter_pos = SummaryWriter(runPath + '/tboard' + '/pos')
    bwriter_em0 = SummaryWriter(runPath + '/tboard/' + labels_name[0])
    bwriter_em1 = SummaryWriter(runPath + '/tboard/' + labels_name[1])
    bwriter_em2 = SummaryWriter(runPath + '/tboard/' + labels_name[2])
    bwriter_em3 = SummaryWriter(runPath + '/tboard/' + labels_name[3])
    bwriter_em4 = SummaryWriter(runPath + '/tboard/' + labels_name[4])
    bwriter_em5 = SummaryWriter(runPath + '/tboard/' + labels_name[5])
    bwriter_em6 = SummaryWriter(runPath + '/tboard/' + labels_name[6])

    ### model
    # detect
    DBface = DBFace()
    DBface.eval().to(device)
    DBface.load("models/Detect/dbfaceSmallH.pth")
    # analy
    is_pretrain = True
    IMAGE_SIZE = 128
    CLASS_NUM = 7
    # Emodel = Resnet.Res18(pretrained=True, inplanes=IMAGE_SIZE, num_classes=CLASS_NUM, drop_rate=0)
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
    if is_pretrain:
        checkpoint = torch.load("models/EmotionRecoginze/epoch91_acc0.8853.pth")
        Emodel.load_state_dict(checkpoint["model_state_dict"], strict=False)
    Emodel.to("cuda:0").eval()


    ### cv
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture()
    cap.open("testdata/"+filepath+".avi")
    if cap.isOpened() != True:
        os._exit(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frameCount = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:break
        frameCount+=1
        frame0 = frame.copy()
        faces = detect(DBface, frame)
        scle = 1.2
        for face in faces:
            rot_img = warp_affine(frame,face,1.0)
            [x1, y1, x2, y2] = face.box
            [xcenter,ycenter] = face.center
            if face.score>=0.75:
                bwriter_pos.add_scalar("posX", xcenter, frameCount)
                bwriter_pos.add_scalar("posY", ycenter, frameCount)
                xy1 = (int(xcenter-face.width*scle//2), int(ycenter-face.width*scle//2))
                # xy1 = (int(x1), int(y1))
                xy2 = (int(xcenter+face.width*scle//2), int(ycenter+face.width*scle//2))
                # cv2.rectangle(frame, xy1, xy2, (0, 255, 0), 3)

                face_img = rot_img[xy1[1]:xy2[1], xy1[0]:xy2[0],:]  ##变换处理
                # face_img = frame0[xy1[1]:xy2[1], xy1[0]:xy2[0], :]

                img_input, img_tensor=process.input_porocess(face_img, IMAGE_SIZE,"cuda:0")

                banch_size = 1
                if frameCount%banch_size==0:
                    outputs,_ = Emodel(img_input)
                    _, predicts = torch.max(outputs, 1)
                    predicts = labels_name[predicts]
                    a0 = outputs[0][0]
                    a1 = outputs[0][1]
                    a2 = outputs[0][2]
                    a3 = outputs[0][3]
                    a4 = outputs[0][4]
                    a5 = outputs[0][5]
                    a6 = outputs[0][6]
                    bwriter_em0.add_scalar("em", outputs[0][0], frameCount)
                    bwriter_em1.add_scalar("em", outputs[0][1], frameCount)
                    bwriter_em2.add_scalar("em", outputs[0][2], frameCount)
                    bwriter_em3.add_scalar("em", outputs[0][3], frameCount)
                    bwriter_em4.add_scalar("em", outputs[0][4], frameCount)
                    bwriter_em5.add_scalar("em", outputs[0][5], frameCount)
                    bwriter_em6.add_scalar("em", outputs[0][6], frameCount)
                    common.drawbbox(lable=predicts, image = frame, bbox= face)
                cv2.imshow("Face", face_img)
        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        ok, frame = cap.read()
    bwriter_pos.close()
    cap.release()
    cv2.destroyAllWindows()