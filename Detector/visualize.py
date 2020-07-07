import os
import re
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from torchvision import transforms
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
import pandas as pd
import scipy.stats as st


def calc_centroid(pred_tensor):
    x1 = pred_tensor[:, 0]
    y1 = pred_tensor[:, 1]
    box_w = pred_tensor[:, 2]
    box_h = pred_tensor[:, 3]

    return np.stack([x1+(box_w//2), y1+(box_h//2)], axis=1)


def hsv_to_rgb(H, S, V):
    """
    transform angle to BGR color as 3-tuple
    """
    C = S*V
    X = C*(1-np.abs((H/60)%2-1))
    m=V-C
    
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
    """
    input: 2D vector
    return: 3-tuple, specifying BGR color selected using HSV formula.
    """
    
    # transform to [-pi,pi] and then to degrees
    angle = np.arctan2(vec[1], vec[0])*180/np.pi
    
    # transform to [0,360]
    angle = (angle + 360) % 360
    return hsv_to_rgb(angle, 1, 1)


def time_to_bgr(k,arrowWindow): # DOES NOT WORK - TODO 
    # map the relative position of the frame to [0,360] angle and then to hue
    rel = (arrowWindow-k)/arrowWindow
    return hsv_to_rgb(0,1,rel)
    

def draw_arrow(frame,
               frameCounter,
               centroids,
               arrowWindow,
               k,
               vis_angle=True,
               windowSize=1,
               scale=2.5):
    """
    draws the direction of the velocity vector from (arrowWindow) frames back
    directions based on the first discrete derivative of the 2D coordinates of
    windowSize consecutive centroids of the detecions, if both exist
    """
    
    # initial arrow
    if frameCounter<windowSize:
        return
    
    # if no prediction at t - windowSize, bo drawing
    if np.isnan(centroids[frameCounter-windowSize, 0]) or np.isnan(centroids[frameCounter, 0]):
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
    
    # compute color based on angle or time
    if vis_angle:
        vec_color = vec_to_bgr([arrowHead[0]-arrowBase[0],arrowHead[1]-arrowBase[1]])
    else:
        vec_color = time_to_bgr(k, arrowWindow)
    
    cv.arrowedLine(frame,arrowBase,arrowHead,color=vec_color, thickness=2,tipLength=0.2,line_type=cv.LINE_AA)


def draw_bounding_boxes(frame, detections, color=(0, 0, 255)):
    """
    frame - a numpy array representing the image.
    detections - [(x, y, w, h, conf)...] bounding boxes array.
    
    draws bounding boxes on frame (in place).
    """

    font = cv.FONT_HERSHEY_COMPLEX
    scale = 0.4
    thickness = cv.FILLED
    margin = 4

    if detections is not None:
        for x1, y1, box_w, box_h, conf in detections:
            x1 = int(x1)
            y1 = int(y1)
            box_w = int(box_w)
            box_h = int(box_h)

            text = str(round(conf, 2))
            txt_size = cv.getTextSize(text, font, scale, thickness)
            end_x = int(x1 + txt_size[0][0] + margin)
            end_y = int(y1 + txt_size[0][1] + margin)

            cv.rectangle(frame, (x1, y1), (end_x, end_y), color, thickness)
            cv.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
            cv.putText(frame, text, (x1, end_y - margin), font, scale,
                       (255, 255, 255), 1, cv.LINE_AA)


def update_centroids(detections, centroids, frameCounter):
    """
    update the detection centroids array
    """
    if detections.shape[0] > 1 and frameCounter > 1:
        prev = centroids[frameCounter - 1][:2]
        detected_centroids = calc_centroid(detections)
        deltas = prev - detected_centroids
        dists = np.linalg.norm(deltas, axis=1)
        arg_best = np.argmin(dists)
        centroid = detected_centroids[arg_best]
        detection = detections[arg_best:arg_best+1]
    else:
        centroid = calc_centroid(detections)[0]
        detection = detections

    centroids[frameCounter][0] = centroid[0]
    centroids[frameCounter][1] = centroid[1]
    centroids[frameCounter][2] = detection[0][4]

    return detection


def draw_k_arrows(frame,frameCounter,centroids,arrowWindow,visAngle,windowSize,scale=5):
    for k in range(arrowWindow):
        draw_arrow(frame,frameCounter-k,centroids,arrowWindow,k,visAngle,windowSize,scale)


def draw_k_centroids(frame,frameCounter,centroids,k):
    if k > frameCounter:
        k = frameCounter-1
    
    for j in range(k):
        if np.isnan(centroids[frameCounter-j][0]):
            continue
        x = int(centroids[frameCounter-j][0])
        y = int(centroids[frameCounter-j][1])
        cv.circle(frame,
                  center = (x,y),
                  radius=2,
                  color= (0,0,255),
                  thickness=-1,
                  lineType=cv.LINE_AA)


def save_pred_video(video_path,
                    output_path,
                    detector,
                    start_frame=0,
                    num_frames=None,
                    frame_rate=None,
                    windowSize=1,
                    arrowWindow=20,
                    visAngle=True,
                    dots=False):
    print("saving to: ", output_path)
    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width = int(vcap.get(3))
    height = int(vcap.get(4))
    detector.set_input_size(width, height)

    print(f'width: {width}, height: {height}')
    if frame_rate is None:
        frame_rate = vcap.get(cv.CAP_PROP_FPS)

    videowriter = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'),
                                 frame_rate, (width, height))

    frameCounter = 0
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    

    # while vcap.isOpened(): # while the stream is open
    #inference_time = np.zeros(num_frames)
    centroids = np.empty((num_frames,3))
    centroids[:] = np.nan

    ###############################
    times = dict()
    for key in ['Read','Rsz_inf','Write']:
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
        detections = detector.detect_image(frame)
        times['Rsz_inf'][frameCounter] = time.time() - start_time ##
               
        if detections is not None:
            detection = update_centroids(detections,centroids,frameCounter)
            draw_bounding_boxes(frame, detections)
        
        if not dots:
            draw_k_arrows(frame,frameCounter,centroids,arrowWindow,visAngle,windowSize,scale=5)
        else:
            draw_k_centroids(frame,frameCounter,centroids,arrowWindow)

        start_time = time.time()##
        videowriter.write(frame)
        times['Write'][frameCounter] = time.time() - start_time ##
    
    #######################################
    for key in times.keys():
        print(f'{key}: {times[key].mean()*1000:2.2f} ms')
    ########################################
    vcap.release()
    videowriter.release()

    return times, centroids


def dots_overlay(overlay, centroids):
    """
    overlay function for overlay_video().
    draws a circle for each centroid, fading over time.
    """
    fade = 0.99
    overlay[:, :, 3] = (overlay[:, :, 3] * fade).astype(np.uint8)

    if np.isnan(centroids[-1, 0]):
        return

    x = int(centroids[-1, 0])
    y = int(centroids[-1, 1])

    cv.circle(overlay,
              center=(x, y), radius=2, color=(0,0,255,255),
              thickness=-1,
              lineType=cv.LINE_AA)


def arrows_overlay(overlay, centroids):
    """
    overlay function for overlay_video().
    draws an arrow between each centroid pair, fading over time.
    """

    fade = 0.99
    overlay[:, :, 3] = (overlay[:, :, 3] * fade).astype(np.uint8)
    if (centroids.shape[0] < 2):
        return

    if np.isnan(centroids[-1, 0]) or np.isnan(centroids[-2, 0]):
        return

    end_x = int(centroids[-1, 0])
    end_y = int(centroids[-1, 1])
    start_x = int(centroids[-2, 0])
    start_y = int(centroids[-2, 1])

    vec_color = vec_to_bgr([end_x-start_x, end_y-start_y])
    cv.arrowedLine(overlay,
                   (start_x, start_y),
                   (end_x, end_y),
                   color=(vec_color[0], vec_color[1], vec_color[2], 255),
                   thickness=1,
                   tipLength=0.2,
                   line_type=cv.LINE_AA)


def overlay_video(input_path, output_path, detector, overlay_fn, draw_bbox=True,
                  start_frame=0, num_frames=None, frame_rate=None):
    """
    Output a video file containing the video at input_path including an overlay generated according to 
    detections.
    """
    vcap = cv.VideoCapture(input_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    if frame_rate is None:
        frame_rate = vcap.get(cv.CAP_PROP_FPS)

    width = int(vcap.get(3))
    height = int(vcap.get(4))
    detector.set_input_size(width, height)

    videowriter = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'),
                                 frame_rate, (width, height))

    centroids = np.empty((num_frames, 2))
    centroids[:] = np.nan
    
    overlay = np.zeros((height, width, 4), np.uint8)

    for frame_num in tqdm(range(num_frames)):
        ret, frame = vcap.read()

        if not ret:
            print("error reading frame")
            break
                
        detections = detector.detect_image(frame)
               
        if detections is not None:
            detection = update_centroids(detections, centroids, frame_num)
            if draw_bbox:
                draw_bounding_boxes(frame, detections)

        overlay_fn(overlay, centroids[:frame_num+1, :])
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[:, :, c] = (alpha_s * overlay[:, :, c] +
                              alpha_l * frame[:, :, c])

        videowriter.write(frame)
    
    vcap.release()
    videowriter.release()
 
    
def plot_with_figure(input_name,
                     output_name,
                     centroids,
                     num_frames,
                     width=1440,
                     height=1080,
                     draw_window=240,
                     frame_rate=60):
    
    FIG_WID_EXT = 0
    
    vcap = cv.VideoCapture(input_name)
    
    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))

    width = int(vcap.get(3))+FIG_WID_EXT
    height = int(vcap.get(4))

    print(f'width: {width}, height: {height}')
    if frame_rate is None:
        frame_rate = vcap.get(cv.CAP_PROP_FPS)

    videowriter = cv.VideoWriter(output_name, cv.VideoWriter_fourcc(*'mp4v'),
                                 frame_rate, (width, height))
    

    velocities_mag = compute_velocity(centroids,to_norm=True)
    confies = centroids[:,2]
    
    write_frame = 255 * np.ones((height,width,3)).astype('uint8')
    for frameCounter in tqdm(range(num_frames)):
        ret, frame = vcap.read()

        if not ret:
            print("error reading frame")
            break
        
        draw_k_centroids(frame,frameCounter,centroids,draw_window)
        
        draw_figure_on_frame(write_frame,frame,frameCounter,velocities_mag,confies,num_frames)
        
        videowriter.write(write_frame)
    
    videowriter.release()


def draw_figure_beside_frame(write_frame,
                         vid_frame,
                         frameCounter,
                         velocities,
                         confidences,
                         total_frames,
                         width_inch = 10,
                         height_inch = 5,
                         dpi = 96,
                         marker_size = 50,
                         plot_back = 180):
    """
    draw updating data on video, in the upper left corner
    TODO: combine to a generic function, selecting if on frame or beside frame
    """
    
    fig,axes = plt.subplots(1,2,figsize=(width_inch,height_inch),dpi=dpi)      
    
    start_range = max(frameCounter-plot_back,0)
    end_range = frameCounter
    
    axes[0].scatter(np.arange(start_range,end_range),
                confidences[start_range:end_range],s=marker_size,c='r')
    axes[1].scatter(np.arange(start_range,end_range),
                velocities[start_range:end_range,0],s=marker_size,c='b')
    axes[0].set_xlim(start_range-10,end_range+10)
    axes[0].set_ylim(0,1)
    axes[1].set_xlim(start_range-10,end_range+10)
    plt.rcParams.update({'font.size':10})
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    
    fig_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.clf()
    plt.close('all')
    """
    print(f'fig_image shape: {fig_image.shape}')
    print(f'vid_frame shape: {vid_frame.shape}')
    print(f'frame shape: {write_frame.shape}')
    """

    write_frame[:,:vid_frame.shape[1],:] = vid_frame
    write_frame[(height_inch*dpi)//2:(height_inch*dpi)//2+height_inch*dpi,vid_frame.shape[1]:,:] = fig_image
    
    

def draw_figure_on_frame(write_frame,
                         vid_frame,
                         frameCounter,
                         velocities,
                         confidences,
                         total_frames,
                         width_inch = 10,
                         height_inch = 5,
                         dpi = 96,
                         marker_size = 50,
                         plot_back = 180):
    """
    draw updating data on video, in the upper left corner
    !!!!!!! Subplots insted of one frame somehow halves running time
    """
    
    if frameCounter==0:
        return
    
    fig,axes = plt.subplots(1,2,figsize=(width_inch,height_inch),dpi=dpi)      
    
    start_range = max(frameCounter-plot_back,0)
    end_range = frameCounter
    
    axes[0].scatter(np.arange(start_range,end_range),
                confidences[start_range:end_range],s=marker_size,c='r')
    axes[1].scatter(np.arange(start_range,end_range),
                velocities[start_range:end_range,0],s=marker_size,c='b')
    axes[1].plot(np.arange(start_range,end_range),
                velocities[start_range:end_range,1],c='r')
    axes[0].set_xlim(start_range-10,end_range+10)
    axes[0].set_ylim(0,1)
    axes[1].set_xlim(start_range-10,end_range+10)
    axes[1].set_ylim(0,1)
    plt.rcParams.update({'font.size':10})
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    
    fig_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.clf()
    plt.close('all')
    """
    print(f'fig_image shape: {fig_image.shape}')
    print(f'vid_frame shape: {vid_frame.shape}')
    print(f'frame shape: {write_frame.shape}')
    """
    
    write_frame[:,:vid_frame.shape[1],:] = vid_frame
    write_frame[:fig_image.shape[0],:fig_image.shape[1],:] = fig_image


def compute_velocity(centroids,to_norm=False,mov_avg=2):
    """
    computes  magnitude of velocity
    """
    veloc = np.diff(centroids,axis=0)
    veloc = np.apply_along_axis(np.linalg.norm,1,veloc)
    
    if to_norm:
        norm_speed = np.percentile(veloc[~np.isnan(veloc)],99) # reject top 1% outliers
        veloc[veloc > norm_speed] = np.nan
        veloc = veloc / norm_speed
    
    veloc_ma = pd.Series(veloc).rolling(window=mov_avg).mean().to_numpy()

    return np.stack([veloc,veloc_ma],axis=1)


def save_missed_frames(video_path,
                       output_dir,
                       detector,
                       save_thresh=0.8,
                       start_frame=0,
                       num_frames=None,
                       save_max=200,
                       above=False):
    print("saving to: ",output_dir)
    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    width  = int(vcap.get(3))
    height = int(vcap.get(4))
    frame_rate = vcap.get(cv.CAP_PROP_FPS)
    
    missed_counter = 0
    saved_counter = 0
    vid_time = re.search('-(\d+)',video_path).group(1)
    yyyymmdd = re.search('output/(\d+)-',video_path).group(1)
    
    if not above:
        save_func = lambda detec,save_thres: (detec is None) or (detec[0][4]<save_thres)
    else:
        save_func = lambda detec,save_thres: (detec is not None) and (detec[0][4]>save_thres)
    
    
    for frameCounter in tqdm(range(num_frames)):
        ret, frame = vcap.read()
        if not ret:
            print("error reading frame")
            vcap.release()
            break
                
        detections = detector.detect_image(frame)

        if save_func(detections,save_thresh):
            missed_counter+=1
            if detections is not None:
                prob = str(round(detections[0][4],3))
            else:
                prob='0'
            if missed_counter%(num_frames//save_max)==0:
                save_path = os.path.join(output_dir,yyyymmdd+'_'+vid_time+'_'+prob+'_'+str(saved_counter)+'.jpg')
                
                cv.imwrite(save_path,frame)
                saved_counter+=1
        
    
    print(f'missed saved: {saved_counter}')
    vcap.release()

    return


def cf(num1,num2):
    """
    Return all integers that divide both arguments.
    """
    from math import gcd
    n=[]
    g=gcd(num1, num2)
    for i in range(1, g+1):
        if g%i==0:
            n.append(i)
    return n


def ablation_heatmap(img, detector, color=0):
    """
    Performs ablation (occlusion) test: slides a WHITE (0) square window over the image,
    and run through the detector. Write the confidence returned by the detector
    in the center of the window
    
    Sqaure color - white squares worked the best for some reason
    Possible to change the square size, and the grid size and area being occluded
    are different. Not sure how it would work with respect to the
    
    Grid cell size of 45, 60, 72 or 90 are working best with arena images
    """
    
    # compute possible cell sizes for ablation grid
    poss_cells = cf(img.shape[1], img.shape[0])
    print(f'Possible cell sizes are {poss_cells}')
    print('Please choose cell size from the list:')
    cell = input()
    if (cell == '') or (int(cell) not in poss_cells):
        print('error, cell not available for dimensions')
        return None, None
    cell = int(cell)
    cells_x = img.shape[1] // cell
    cells_y = img.shape[0] // cell
    print(f'cells x: {cells_x}, cells y: {cells_y}, total of {cells_x*cells_y} iterations. Continue [y/n]?')
    inp = input()
    if inp != 'y':
        print('exiting')
        return None, None
    
    # initialize confidence map with image shape
    conf_map = np.ones((img.shape[0],img.shape[1]))
        
    # for each cell, get preditcion and write confidence in conf_map
    for i in tqdm(range(cells_y),desc='Rows'):
        for j in range(cells_x):
            ablated_img = np.copy(img)
            try:
                ind_i = i*cell
                ind_j = j*cell
                end_i = min(ind_i + cell, img.shape[0])
                end_j = min(ind_j + cell, img.shape[1])
                ablated_img[ind_i:end_i, ind_j:end_j, :] = color
                
            except Exception as e:
                print(f'indexing problem at i:{i},j:{j}\n',e)
                print("exiting")
                return
            detection = detector.detect_image(ablated_img)
            if detection is not None:
                conf = detection[0][4]
            else:
                conf = 0.0
            
            conf_map[ind_i:end_i, ind_j:end_j] = conf
            
    detection = detector.detect_image(img)
    if detection is None:
        base_conf = 0
    else:
        base_conf = detection[0][4]
    
    return conf_map, base_conf


def visualize_ablation_heatmap(img,
                               detector,
                               alpha = .5,
                               cmap = 'PiYG',
                               zero_cmap = 'Greens',
                               ones_cmap = 'Reds_r',
                               enhance_fact = 3,
                               epsi = 1e-1,
                               base_thresh=0.05,
                               color=0):
    """
    Get ablation confidence map, and visualize with the original image.
    
    """
    # compute ablation heatmap
    old_conf_thresh = detector.conf_thres
    detector.set_conf_and_nms(new_conf_thres=0.01)
    conf_map, base_conf = ablation_heatmap(img,detector)
    detector.set_conf_and_nms(new_conf_thres=old_conf_thresh)
    if conf_map is None:
        return
    
    # subtract confidence of prediction without any ablation
    conf_map-=base_conf
    
    # replace close to zero values with zero
    conf_map[np.abs(conf_map)<epsi] = 0.0
    
    # enhance contrast of image so it will be more clear with overlay
    img = np.array(ImageEnhance.Contrast(Image.fromarray(img))\
                   .enhance(enhance_fact))
    
    fig, axes = plt.subplots(1,1,figsize=(19,10))
    
    # plot heatmap
    if base_conf<base_thresh or 1-base_conf<base_thresh:
        # if base prediction is very high or low, set 1-way color scale
        if base_conf<base_thresh:
            cmap = zero_cmap
            vmin = 0
            vmax = 1
        else:
            cmap = ones_cmap
            vmin = -1
            vmax = 0
        axes.imshow(conf_map,alpha=1-alpha,cmap=cmap)
    else:
        # if there are both cells that increase or decrease, set 2-way color sclae
        axes.imshow(conf_map,alpha=1-alpha,cmap=cmap,norm=TwoSlopeNorm(0))
        vmin = -1
        vmax = 1
    
    # plot enhanced image
    axes.imshow(img,alpha=alpha)
    
    # set colorbar values
    scal_map = cm.ScalarMappable()
    scal_map.set_cmap(cmap)
    #scal_map.set_clim(vmin=np.min(conf_map),vmax=np.max(conf_map))
    scal_map.set_clim(vmin=vmin,vmax=vmax)
    cbar = plt.colorbar(mappable=scal_map,ax=axes)
    cbar.set_label('$\Delta$ Conf.',rotation=0,labelpad=25,fontsize=18)

    def f_title(x):
        return 'None' if x==0 else f'{x:0.2f}'

    axes.set_title(f'Confidence without occlusion: {f_title(base_conf)}',
                   fontsize=18)
    axes.axis('off')


def plot_image(detections, img, detector, output_path=None):
    """
    Run the detector on img and plot the image with bounding boxes.
    """

    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)

    if detections is not None:
        # browse detections and draw bounding boxes
        for x1, y1, box_w, box_h, conf in detections:
            color = (0,0,1,1)
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                 linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=str(round(conf,2)), 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    plt.axis('off')
    # save image
    # plt.savefig(img_path.replace(".jpg", "-det.jpg"),        
    #                   bbox_inches='tight', pad_inches=0.0)
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()
