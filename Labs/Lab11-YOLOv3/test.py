import cv2
import torch
import numpy as np
from yolov3.models import Darknet
from yolov3.utils.utils import non_max_suppression
def img2tensor(img):
    h, w, _ = img.shape
    img = np.pad(img, 
                ([max((w-h)//2, 0), max(w-h-(w-h)//2, 0)],
                 [max((h-w)//2, 0), max(h-w-(h-w)//2, 0)],
                 [0, 0]), 
                'constant', 
                constant_values = (0, 0))
    img = cv2.resize(img, (416, 416))
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    return img.float(), [h, w]

# if torch.cuda.is_available():
#     device = torch.device('cuda', 0)
#     torch.cuda.current_device()
# else:
#     device = torch.device('cpu')
device = torch.device('cpu')
# set up model
model = Darknet('yolov3/config/yolov3.cfg', img_size=416).float().to(device)
model.load_darknet_weights('yolov3/weights/yolov3.weights')
model.eval();  # model.train(False)

cap = cv2.VideoCapture('test.mp4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale = max(w, h) / 416.0

colors = {0:(0,0,255), 1:(0,255,0), 2:(255,0,0), 9:(255,0,255)}

while 1:
    ret, frame = cap.read()
    if ret == False:
        break
    img = frame[:,:,::-1] / 255.0
    img = img2tensor(img)[0]
    detections = model(img.to(device))
    detections = non_max_suppression(detections, conf_thres=0.3, nms_thres=0.5)
    det = [x.detach().cpu().numpy() for x in detections][0]
    det[:,0:4] *= scale
    det[:,0:2] -= np.asarray([max((h-w)//2, 0), max((w-h)//2, 0)])
    det[:,2:4] -= np.asarray([max((h-w)//2, 0), max((w-h)//2, 0)])
    for j in range(det.shape[0]):
        cv2.rectangle(frame, (det[j,0], det[j,1]), (det[j,2], det[j,3]), 
                      color=colors[int(det[j][-1])] if int(det[j][-1]) in colors else (0,255,255),
                      thickness=2)

    cv2.imshow("test",frame)
    if not cap.isOpened():
        break
    # if cv2.waitKey()==27:
    #     break

cap.release()
cv2.destroyWindow("test")