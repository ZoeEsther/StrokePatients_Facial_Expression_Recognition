import os, time, cv2
import argparse
import math
import numpy as np
import csv
import torch
import torch.nn.functional as F
from visdom import Visdom
import scipy.signal as signal
import PIL.Image as image
from torchvision import transforms
from models.Detect.DBFaceSmallH import DBFace
from models.EmotionRecoginze import Resnet, aVIT

from uties import common, process


class Video_FER:
    def __init__(self, source="0", device='cuda:0', model_name='vit', out_path='outputs',
                 has_vis=True, has_plt=True, has_csv=True):  # 构造函数，类中的每一个方法的第一参数必须是self，使用或者不使用都可以
        self.hasPlt = has_plt
        self.hasVis = has_vis
        self.hasCsv = has_csv
        self.Device = device
        self.outPath = out_path
        self.Vis = Visdom(env="show")
        self.dbFace = DBFace()
        self.dbFace.load("models/Checkpoint/dbfaceSmallH.pth")
        self.dbFace.eval().to(self.Device)
        # self.labLst = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]
        # self.labLst = ['SU', 'HA', 'SA', 'AN', 'NE']
        self.labLst = ['PA', 'TI', 'ST', 'NE','UNKNOW']
        self.lab2va = {}
        for i,lab in enumerate(self.labLst):
            self.lab2va[lab] = i
        self.emotionLab = "NE"
        self.processBuffer = None
        self.outputBuffer = []
        self.faceImg = None

        self.model_name = model_name
        self.Emodel = None
        if self.model_name == "resnet":
            self.Emodel = Resnet.Res18(pretrained=True, inplanes=224, num_classes=7, drop_rate=0)
            checkpoint = torch.load("models/Checkpoint/resnet-224.pth")
            self.Emodel.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.imgSize = 224
        else:
            from models.EmotionRecoginze.kvit import kvit_pretrained
            self.Emodel = kvit_pretrained()
            checkpoint = torch.load("models/Checkpoint/kvit-epoch=09-val_acc=0.9921.ckpt")
            self.Emodel.load_state_dict(checkpoint["state_dict"], strict=True)
            # self.Emodel = aVIT.ViT(
            #     image_size=128,
            #     patch_size=2,
            #     num_classes=7,
            #     dim=64,
            #     depth=3,
            #     heads=3,
            #     mlp_dim=16,
            #     dropout=0,
            #     emb_dropout=0
            # )
            # checkpoint = torch.load("models/Checkpoint/kvit-epoch=27-val_acc=0.9791.ckpt")
            # self.Emodel.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.imgSize = 128
        self.Emodel.to(self.Device).eval()

        ###First frame test
        self.Source = source
        if self.Source == "0":
            self.Cap = cv2.VideoCapture(0)
            self.filePath = time.strftime("%m-%d-%H-%M", time.localtime())
            self.dirPath = "cap//"
        else:
            self.Cap = cv2.VideoCapture(self.Source)
            sourcePath = self.Source.split("视频数据集\\")[-1].split("\\")
            self.filePath = sourcePath.pop(-1).split(".")[0]
            # self.dirPath = "-".join(sourcePath)
            self.dirPath = ""
            while (sourcePath):
                self.dirPath = self.dirPath + sourcePath.pop(0) + "\\"

        self.frameWidth, self.frameHeight = self.Cap.get(3), self.Cap.get(4)

        self.frame = None
        self.frameCount = 0
        self.frameRate = self.Cap.get(5)
        self.Face = None
        self.xTrace, self.yTrace = None, None

    def warp_affine(self, image, face, scale=1.0):
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

    def draw_vis(self, predict, outputs, count):
        self.Vis.bar(X=outputs,
                     win='current confidence',
                     opts=dict(
                         stacked=False,
                         rownames=self.labLst,
                         title='real time analysis',
                         ylabel='Confidence',  # y轴名称
                         xtickmin=0.4,  # x轴左端点起始位置
                         xtickstep=0.4,  # 每个柱形间隔距离
                         ytickmax=1.0
                     ))
        self.Vis.line([outputs.cpu().detach().numpy()],
                      [count],
                      win="{}:{}\\{}".format(self.model_name, self.dirPath, self.filePath),
                      opts=dict(title="{}:{}\\{}".format(self.model_name, self.dirPath, self.filePath),
                                ymax=1.0,
                                legend=self.labLst),
                      update='append')

        self.Vis.line(Y=[predict.cpu().detach().numpy()],
                      X=[count],
                      win="pred:{}:{}\\{}".format(self.model_name, self.dirPath, self.filePath),
                      opts=dict(title="{}:{}\\{}".format(self.model_name, self.dirPath, self.filePath)),
                      update='append')

    def faceDetect(self, image, threshold=0.3, nms_iou=0.3):
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

        torch_image = torch_image.to(self.Device)

        hm, box, landmark = self.dbFace(torch_image)
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

    def emAnalysis(self, face_img, image_size=128):
        def imgProcess(np_img, imgSize, device):
            test_transformer = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
            ])
            return test_transformer(np_img).to(device)

        img_input = imgProcess(face_img, image_size, self.Device)
        img_input = img_input.unsqueeze(0)
        outputs = self.Emodel(img_input)
        return outputs

    def testDetect(self):
        self.frame = self.Cap.read()[1]
        if self.frame is None:
            return False
        faces = self.faceDetect(self.frame)
        faces.sort(key=lambda x: abs(x.center[0] - self.frameWidth // 2) + abs(x.center[1] - self.frameHeight // 2))
        for face in faces:
            [xcenter, ycenter] = face.center
            if face.score >= 0.75:
                self.Face = face
                [self.xTrace, self.yTrace] = self.Face.center
                return True
        return False

    def normProcess(self, input, banch_size=20, method="means"):
        self.processBuffer = input if self.processBuffer is None else torch.cat((self.processBuffer, input), 0)
        if self.frameCount % self.frameRate == 0:
            if method == "means":
                outputs = torch.mean(self.processBuffer, dim=0)
            elif method == "max":
                outputs = torch.mean(self.processBuffer, dim=0)
            # outputs,_ = torch.max(predicts_buff, dim=0) ##最大值滤波
            _, predicts = torch.max(outputs, 0)
            self.emotionLab = self.labLst[predicts]
            VaValue = self.lab2va[self.emotionLab]
            # self.outputBuffer.append([self.frameCount // self.frameRate] + outputs.cpu().detach().numpy().tolist())
            self.outputBuffer.append([self.frameCount // self.frameRate] + [VaValue])
            self.processBuffer = None

            if self.hasVis:
                self.draw_vis(predicts, outputs, self.frameCount // self.frameRate)
        if self.hasPlt:
            cv2.imshow("Face", self.faceImg)
            common.drawbbox(lable=self.emotionLab, image=self.frame, bbox=self.Face)
        return

    def faceTrace(self, faces, box_scle=1.2, use_affine=True):
        for face in faces:
            [xcenter, ycenter] = face.center
            boxSize = face.width
            xy1 = (int(xcenter - boxSize * box_scle / 2.0), int(ycenter - boxSize * box_scle / 2.0))
            xy2 = (int(xcenter + boxSize * box_scle / 2.0), int(ycenter + boxSize * box_scle / 2.0))
            safecheck = 0 <= xy1[0] and 0 <= xy1[1] and xy2[0] <= self.frameWidth and xy2[1] <= self.frameHeight
            tracecheck = abs(xcenter - self.xTrace) < 40 and abs(ycenter - self.yTrace) < 40
            if face.score >= 0.75 and tracecheck and safecheck:
                self.Face = face
                self.xTrace, self.yTrace = xcenter, ycenter

                if use_affine:
                    img0 = self.warp_affine(self.frame, self.Face, 1.0)
                else:
                    img0 = self.frame[xy1[1]:xy2[1], xy1[0]:xy2[0], :]
                self.faceImg = img0[xy1[1]:xy2[1], xy1[0]:xy2[0], :]
                return True
        return False

    def saveCsv(self, input_Buffer, root_path, dir_path, file_name):
        dirPath = '{}\\{}'.format(root_path, dir_path)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        with open('{}\\{}.csv'.format(dirPath, file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['frame', 'SU', 'FE', 'DI', 'HA', 'SA', 'AN', 'NE'])
            writer.writerow(['frame', 'predict'])
            writer.writerows(input_Buffer)

    def fftAnalysis(self, input_data):
        Path = '{}\\{}'.format(self.dirPath, self.filePath)
        tdata = np.array(input_data).T[1:]
        fdata = []
        for td in tdata:
            fdata.append(np.abs((np.fft.rfft(td) / len(td))).tolist())
        fdata = np.array(fdata).T
        self.Vis.line(Y=fdata[1:], X=[i / 2 / len(fdata) for i in range(len(fdata) - 1)],
                      win=Path,
                      opts={
                          "title": '频谱图:' + Path,
                          "xlabel": 'a',
                          "ylabel": 'frequency',
                          "legend": self.emotionLab})

    def Detect(self):
        while (self.frame is not None):
            if self.Source == "0":
                self.frameCount += 1
            else:
                self.frameCount = self.Cap.get(1)
            faces = self.faceDetect(self.frame)
            self.faceTrace(faces, box_scle=1.2, use_affine=True)
            predicts = self.emAnalysis(self.faceImg, image_size=self.imgSize)
            self.normProcess(predicts, banch_size=self.frameRate, method="means")
            if self.hasPlt:
                cv2.imshow("Expression analysis", self.frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            self.frame = self.Cap.read()[1]
        if self.hasCsv:
            self.saveCsv(self.outputBuffer, self.outPath, self.dirPath, self.filePath)
        # self.fftAnalysis(self.outputBuffer)



if __name__ == "__main__":
    # for root, dirs, files in os.walk("..\\Data\\视频数据集"):
    #     for file in files:
    #         if "1.mp4" == file or "2.mp4" == file or "3.mp4" == file or "4.mp4" == file or  "5.mp4" == file :
    #             filepath = os.path.join(root, file)
    #             print(filepath)
    #             ferdetect = Video_FER(source=filepath, device='cuda:0', model_name='vit',
    #                                   has_vis=False, has_plt=False, has_csv=True)
    #             print("寻找患者")
    #             if ferdetect.testDetect():
    #                 print("患者检测成功")
    #                 if not (ferdetect.Detect()):
    #                     print("患者检测失败")
    #                     continue

    ferdetect = Video_FER(source="./testdata/pain1.avi", device='cuda:0', model_name='vit',
                          has_vis=True, has_plt=True, has_csv=True)
    print("寻找患者")
    if ferdetect.testDetect():
        print("患者检测成功")
        ferdetect.Detect()
    else:
        print("患者检测失败")
