import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import re


def xyxy_to_xywh(xyxy, input_size, output_shape):
    """
    xyxy - an array of xyxy detections in input_size x input_size coordinates.
    input_size - length of each input dimension.
    output_shape - shape of output array (height, width)
    """

    pad_x = max(output_shape[0] - output_shape[1], 0) * (input_size / max(output_shape))
    pad_y = max(output_shape[1] - output_shape[0], 0) * (input_size / max(output_shape))
    unpad_h = input_size - pad_y
    unpad_w = input_size - pad_x

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    box_h = ((y2 - y1) / unpad_h) * output_shape[0]
    box_w = ((x2 - x1) / unpad_w) * output_shape[1]
    y1 = ((y1 - pad_y // 2) / unpad_h) * output_shape[0]
    x1 = ((x1 - pad_x // 2) / unpad_w) * output_shape[1]
    
    return torch.stack([x1, y1, box_w, box_h], dim=1)


def calc_centroid(pred_tensor):
    x1 = pred_tensor[:, 0]
    y1 = pred_tensor[:, 1]
    box_w = pred_tensor[:, 2]
    box_h = pred_tensor[:, 3]

    return torch.stack([x1+(box_w//2), y1+(box_h//2)], dim=1).numpy()


def hsv_to_rgb(H):
    # assumes saturation (S) is 1 and value (V) is 1
    C = 1
    X = C*(1-np.abs((H/60)%2-1))
    m=1-C 
    
    if H >= 0 and H < 60:
        r,g,b = C,X,0
    elif H >= 60 and H < 120:
        r,g,b = X,C,0
    elif H >= 120 and H < 180:
        r,g,b = 0,C,X
    elif H >= 180 and H < 240:
        r,g,b = 0,X,C
    elif H >= 240 and H <300:
        r,g,b = X,0,C
    else:
        r,g,b = C,0,X
    
    roun = lambda x: int(round(x))
    
    # return b,g,r
    return roun((b+m)*255),roun((g+m)*255),roun((r+m)*255)


def vec_to_bgr(vec):
    # takes 2d vector, returns bgr color using HSV formula +++remember OPENCV is BGR
    
    # transform to [-pi,pi] and then to degrees
    angle = np.arctan2(vec[1],vec[0])*180/np.pi
    
    # transform to [0,360]
    angle = (angle + 360) % 360
    return hsv_to_rgb(angle)

def compose_torch_transform(width,height,detector):
    img_size = detector.img_size
    # scale and pad image
    ratio = min(img_size/width, img_size/height)
    imw = round(width * ratio)
    imh = round(height * ratio)
    return transforms.Compose([transforms.Resize((imh,imw)),
                                       transforms.Pad((max(int((imh-imw)/2),0), 
                                                       max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
                                                       max(int((imw-imh)/2),0)), (128,128,128)),
                                       transforms.ToTensor()])

def resize_image(img, img_transforms):
    """
    resize PIL/ndarray image to fit the yolo detector and return a tensor.
    """
    # convert image to Tensor
    return img_transforms(img)


def draw_arrow(frame, frameCounter, centroids, windowSize=1, scale=4):
    # initial arrow
    if frameCounter<windowSize:
        return
    
    # no prediction at t - windowSize
    if centroids[frameCounter-windowSize][0] == 0 or centroids[frameCounter][0] == 0:
        return

    arrowBase = tuple(centroids[frameCounter-windowSize].astype(int))
    arrowHead = tuple(centroids[frameCounter].astype(int))
    
    # scale head for better visibility
    extend_x = scale*(arrowHead[0]-arrowBase[0])
    extend_y = scale*(arrowHead[1]-arrowBase[1])
    
    new_x = arrowHead[0] + extend_x
    new_y = arrowHead[1] + extend_y
    
    if new_x<0:
        new_x=0
    if new_x > frame.shape[1]:
        new_x = frame.shape[1]
        
    if new_y<0:
        new_y = 0
    if new_y>frame.shape[0]:
        new_y=frame.shape[0]
    
    arrowHead = (new_x,new_y)
    
    vec_color = vec_to_bgr([arrowHead[0]-arrowBase[0],arrowHead[1]-arrowBase[1]])
    
    cv.arrowedLine(frame,arrowBase,arrowHead,color=vec_color, thickness=2,tipLength=0.2)

    
def draw_bounding_boxes(frame, detections, detector, colors):
    # Get bounding-box colors
   
    font = cv.FONT_HERSHEY_COMPLEX
    scale = 0.4
    thickness = cv.FILLED
    margin = 4

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()

        # browse detections and draw bounding boxes
        for x1, y1, box_w, box_h, conf, cls_conf, cls_pred in detections:
            cls = detector.classes[int(cls_pred)]
            color = colors[int(cls_pred) % len(colors)]
            color = [i * 255 for i in color]
            text = cls + " " + str(round(conf.item(), 2))
            txt_size = cv.getTextSize(text, font, scale, thickness)
            end_x = x1 + txt_size[0][0] + margin
            end_y = y1 + txt_size[0][1] + margin
            cv.rectangle(frame,(x1,y1),(x1+box_w, y1+box_h),color,2)
            cv.rectangle(frame, (x1, y1), (end_x, end_y), color, thickness)
            cv.putText(frame, text, (x1, end_y - margin), font, scale, (255,255,255), 1, cv.LINE_AA)


def update_centroids(detections,detections_xyxy,centroids,frameCounter):

    detections = torch.cat([detections, detections_xyxy[:, 4:]], dim=1)

    if detections.shape[0] > 1 and frameCounter > 1:
        prev = centroids[frameCounter - 1]
        detected_centroids = calc_centroid(detections)
        deltas = prev - detected_centroids
        dists = np.linalg.norm(deltas, axis=1)
        arg_best = np.argmin(dists)
        centroid = detected_centroids[arg_best]
    else:
        centroid = calc_centroid(detections)[0]

    centroids[frameCounter][0] = centroid[0]
    centroids[frameCounter][1] = centroid[1]

    return detections


def draw_k_arrows(frame,frameCounter,arrowWindow,centroids,windowSize,scale=5):
    for k in range(arrowWindow):
        draw_arrow(frame,frameCounter-k,centroids,windowSize,scale)

def save_pred_video(video_path,
                    output_path,
                    detector,
                    conf_thres=0.9,
                    nms_thres=0.5,
                    start_frame=0,
                    num_frames=None,
                    windowSize=1,
                    arrowWindow=20):
    print("saving to: ",output_path)
    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width  = int(vcap.get(3))
    height = int(vcap.get(4))
    frame_rate = vcap.get(cv.CAP_PROP_FPS)

    img_transforms = compose_torch_transform(width,height,detector)
    
    #videowriter = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'),
                                 #frame_rate, (width, height))

    frameCounter = 0
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    # TODO delete
    missed_counter = 0
    vid_time = re.search('20200521-(\d+)',output_path).group(1)
    
    # while vcap.isOpened(): # while the stream is open
    #inference_time = np.zeros(num_frames)
    centroids = np.zeros((num_frames,2))
    
    
    ###############################
    times = dict()
    for key in ['Read','BGR2RGB','fromarray','resize_image','Inference','Detect_draw','Write']:
        times[key] = np.zeros(num_frames)    
    ################################
    
    for frameCounter in tqdm(range(num_frames)):
        start_time = time.time() ##
        ret, frame = vcap.read()
        times['Read'][frameCounter] = time.time() - start_time ##

        if not ret:
            print("error reading frame")
            break
        
        start_time = time.time() ##
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        times['BGR2RGB'][frameCounter] = time.time() - start_time ##
        
        start_time = time.time() ##
        PIL_im = Image.fromarray(frame_rgb)
        times['fromarray'][frameCounter] = time.time() - start_time ##
        
        start_time = time.time() ##
        resized_frame = resize_image(PIL_im, img_transforms)
        times['resize_image'][frameCounter] = time.time() - start_time ##
        
        
        start_time = time.time() ##
        detections_xyxy = detector.detect_image(resized_frame, conf_thres=conf_thres, nms_thres=nms_thres)
        times['Inference'][frameCounter] = time.time() - start_time ##
        
        start_time = time.time() ##
        
        if detections_xyxy is not None:
            """
            detections = xyxy_to_xywh(detections_xyxy, detector.img_size, frame_rgb.shape)
            detections = update_centroids(detections,detections_xyxy,centroids,frameCounter)
            draw_bounding_boxes(frame, detections, detector, colors)
            draw_k_arrows(frame,frameCounter,arrowWindow,centroids,windowSize,scale=5)
            """
        else:
            cv.imwrite('./21_05_missed/21_05_'+vid_time+'_'+str(missed_counter)+'.jpg',frame_rgb)
            missed_counter+=1
        

        times['Detect_draw'][frameCounter] = time.time() - start_time ##
        
        
        start_time = time.time()##
        #videowriter.write(frame)
        times['Write'][frameCounter] = time.time() - start_time ##
    
    #######################################
    #for key in times.keys():
    #    print(key,': ',times[key].mean())
    ########################################
    print("missed: ",missed_counter)
    vcap.release()
    #videowriter.release()

    return times
