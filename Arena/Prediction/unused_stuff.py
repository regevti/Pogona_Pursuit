"""
Archive containing:
Old code - tested code that we replaced with better solutions
Discontinued code - functions and scripts we didn't continue using. Probably won't work with new code and objects.
But we spent some time writing it, and it might contain some useful code or ideas to be used in other code
"""

"""
-----------------------  Utility functions ------------------------------
"""


def hsv_to_rgb(H, S, V):
    """
    transform angle to BGR color as 3-tuple
    """
    C = S * V
    X = C * (1 - np.abs((H / 60) % 2 - 1))
    m = V - C

    if H >= 0 and H < 60:
        r, g, b = C, X, 0
    elif H >= 60 and H < 120:
        r, g, b = X, C, 0
    elif H >= 120 and H < 180:
        r, g, b = 0, C, X
    elif H >= 180 and H < 240:
        r, g, b = 0, X, C
    elif H >= 240 and H < 300:
        r, g, b = X, 0, C
    else:
        r, g, b = C, 0, X

    def roun(x):
        return int(round(x))

    # return b,g,r
    return roun((b + m) * 255), roun((g + m) * 255), roun((r + m) * 255)


def vec_to_bgr(vec):
    """
    input: 2D vector
    return: 3-tuple, specifying BGR color selected using HSV formula.
    """

    # transform to [-pi,pi] and then to degrees
    angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi

    # transform to [0,360]
    angle = (angle + 360) % 360
    return hsv_to_rgb(angle, 1, 1)


def time_to_bgr(k, arrowWindow):  # DOES NOT WORK - TODO
    # map the relative position of the frame to [0,360] angle and then to hue
    rel = (arrowWindow - k) / arrowWindow
    return hsv_to_rgb(0, 1, rel)


"""
-----------------------  Less important video functions ------------------------------
"""


def draw_arrow(
        frame,
        frameCounter,
        centroids,
        arrowWindow,
        k,
        vis_angle=True,
        windowSize=1,
        scale=2.5,
):
    """
    draws the direction of the velocity vector from (arrowWindow) frames back
    directions based on the first discrete derivative of the 2D coordinates of
    windowSize consecutive centroids of the detecions, if both exist
    """

    # initial arrow
    if frameCounter < windowSize:
        return

    # if no prediction at t - windowSize, bo drawing
    if np.isnan(centroids[frameCounter - windowSize, 0]) or np.isnan(
            centroids[frameCounter, 0]
    ):
        return

    arrowBase = tuple(centroids[frameCounter - windowSize].astype(int))
    arrowHead = tuple(centroids[frameCounter].astype(int))

    # scale head for better visibility
    extend_x = scale * (arrowHead[0] - arrowBase[0])
    extend_y = scale * (arrowHead[1] - arrowBase[1])

    new_x = arrowHead[0] + extend_x
    new_y = arrowHead[1] + extend_y

    if new_x < 0:
        new_x = 0
    if new_x > frame.shape[1]:
        new_x = frame.shape[1]

    if new_y < 0:
        new_y = 0
    if new_y > frame.shape[0]:
        new_y = frame.shape[0]

    arrowHead = (new_x, new_y)

    # compute color based on angle or time
    if vis_angle:
        vec_color = vec_to_bgr(
            [arrowHead[0] - arrowBase[0], arrowHead[1] - arrowBase[1]]
        )
    else:
        vec_color = time_to_bgr(k, arrowWindow)

    cv.arrowedLine(
        frame,
        arrowBase,
        arrowHead,
        color=vec_color,
        thickness=2,
        tipLength=0.2,
        line_type=cv.LINE_AA,
    )


def draw_bounding_boxes(frame,
                        detections,
                        color=(0, 0, 255),
                        is_xyxy=True,
                        ):
    """
    frame - a numpy array representing the image.
    detections - [(x, y, x, y, conf)...] bounding boxes array.
    if is_xyxy==False, then assumes detections is xywh

    draws bounding boxes on frame (in place).
    """

    font = cv.FONT_HERSHEY_COMPLEX
    scale = 0.4
    thickness = cv.FILLED
    margin = 4

    if detections is not None:
        for x1, y1, c, d, conf in detections:
            x1 = int(x1)
            y1 = int(y1)

            x2, y2, box_w, box_h = None, None, None, None
            if is_xyxy:
                x2 = int(c)
                y2 = int(d)
            else:
                box_w = int(c)
                box_h = int(d)

            text = str(round(conf, 2))
            txt_size = cv.getTextSize(text, font, scale, thickness)
            end_x = int(x1 + txt_size[0][0] + margin)
            end_y = int(y1 + txt_size[0][1] + margin)

            cv.rectangle(frame, (x1, y1), (end_x, end_y), color, thickness)

            if is_xyxy:
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
            cv.putText(
                frame,
                text,
                (x1, end_y - margin),
                font,
                scale,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )


def draw_k_arrows(
        frame, frameCounter, centroids, arrowWindow, visAngle, windowSize, scale=5
):
    for k in range(arrowWindow):
        draw_arrow(
            frame,
            frameCounter - k,
            centroids,
            arrowWindow,
            k,
            visAngle,
            windowSize,
            scale,
        )


def draw_k_centroids(frame, frameCounter, centroids, k, color=(0, 0, 255)):
    if k > frameCounter:
        k = frameCounter - 1

    for j in range(k):
        if np.isnan(centroids[frameCounter - j][0]):
            continue
        x = int(centroids[frameCounter - j][0])
        y = int(centroids[frameCounter - j][1])
        cv.circle(
            frame,
            center=(x, y),
            radius=2,
            color=color,
            thickness=-1,
            lineType=cv.LINE_AA,
        )


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
                         width_inch=10,
                         height_inch=5,
                         dpi=96,
                         marker_size=50,
                         plot_back=180):
    """
    draw updating data on video, in the upper left corner
    !!!!!!! Subplots insted of one frame somehow halves running time
    """
    
    if frameCounter == 0:
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
        axes.imshow(conf_map,alpha=1-alpha,cmap=cmap,norm=matplotlib.colors.TwoSlopeNorm(0))
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
            if frame_num > 1:
                prev = centroids[frame_num - 1][:2]
                detection = nearest_detection(detections, prev)
                centroid = xywh_to_centroid(detection)
                centroids[frame_num][0] = centroid[0]
                centroids[frame_num][1] = centroid[1]
                centroids[frame_num][2] = detection[4]

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
                     with_figure=True,
                     filtered_centroids=None,
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
        
        if filtered_centroids is not None:
            draw_k_centroids(frame,frameCounter,filtered_centroids,draw_window,color=(255,0,0))
        
        if with_figure:
            draw_figure_on_frame(write_frame,frame,frameCounter,velocities_mag,confies,num_frames)
            videowriter.write(write_frame)
        else:
            videowriter.write(frame)
        
    videowriter.release()


    
    
# analyze timings
XLIM=40
YLIM=40

TOTAL_XLIM = 80
TOTAL_YLIM = 80

centros = centroids[:,:2]
speed = visualize.compute_velocity(centros)

confs = centroids[:,2]
for k in times.keys():
    print(k,": ",round(1/times[k].mean()))
times['Total'] = np.sum(np.stack([times[k] for k in times.keys() if k!='Total'],axis=1),axis=1)
print("Average FPS: ",round(1/times['Total'].mean()))



phases = ['Rsz_inf','Read','Write'] # sorted order
fig,axs = plt.subplots(len(phases)+1,2,figsize=(20,30))
k='Total'
axs[0][0].set_title(k+' Histogram')
axs[0][1].set_title(k+' Time Plot')
axs[0][0].hist(times[k]*1000,label=k,bins=100)
axs[0][0].set_xlim(0,XLIM)
axs[0][0].set_xlabel('Time (ms)')
axs[0][1].set_xlabel('Frame number')
axs[0][0].set_ylabel('Freq')
axs[0][1].set_ylabel('Time (ms)')
axs[0][1].scatter(np.arange(times[k].shape[0]),times[k]*1000)
axs[0][1].set_ylim(0,YLIM)
axs[0][0].plot(np.ones(5)*16.6,np.linspace(1,1000,5),color='r')
axs[0][1].plot(np.linspace(1,3000,5),np.ones(5)*16.6,color='r')
for i,k in enumerate(phases):
    i+=1
    axs[i][0].set_title(k+' Histogram')
    axs[i][1].set_title(k+' Time Plot')
    axs[i][0].hist(times[k]*1000,label=k,bins=100)
    axs[i][0].set_xlim(0,XLIM)
    axs[i][0].set_xlabel('Time (ms)')
    axs[i][1].set_xlabel('Frame number')
    axs[i][0].set_ylabel('Freq')
    axs[i][1].set_ylabel('Time (ms)')
    axs[i][1].scatter(np.arange(times[k].shape[0]),times[k]*1000)
    axs[i][1].set_ylim(0,YLIM)
#plt.savefig('timings.jpg')




"""
All path trajectory from predictor eval notebook. Each ########.. signals new cell.
"""
##################################
#trial_data = val[1]
trial_data = lots_o_touches2
bboxes = all_df.loc[trial_data][['x1','y1','x2','y2']].values

FIRST_FRAME = 0

bboxes = bboxes[FIRST_FRAME:]
# TODO - slicing bbox array creates a discrepency between the forecasts and the dataframe?
# might
#bboxes[60:70] = np.nan
eval_results, forecasts = train_eval.eval_trajectory_predictor(traj_predictor, bboxes)

##################################

################################## Plot entire trial trajectory
def gen_forecasts_path(forecasts, n=0):
    # first_forecast - the first forecast which doesn't contain any np.nan
    first_forecast = next(i for i, fc in enumerate(forecasts) if fc is not None and not np.any(np.isnan(fc)))
    path = np.empty((len(forecasts) - first_forecast - 1, 4))
    # create the entire path from the forecasts
    for i in range(path.shape[0]):
        path[i] = forecasts[first_forecast + i][n]
    return path, first_forecast
##################################
X1, Y1, X2, Y2 = 0, 1, 2, 3

# TODO - maybe function
n = 10
fpath, first_f = gen_forecasts_path(forecasts, n=n)
fpath = np.roll(fpath, n+1, axis=0)
fpath[:n+1] = np.nan
rpath = bboxes #[first_f+1:]

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')

# change these parameters to inspect only a single path and make the others less visible
# can also use zorder paramater to control the oreder of plots
real_alpha = 0.5
fore_alpha = 0.0
diff_alpha = 0.0

# plot top left, bottom right coordinates of recorded path
ax.plot(rpath[:, X2], rpath[:, Y2], color="r" ,label = 'real_x2y2', alpha=real_alpha)
ax.plot(rpath[:, X1], rpath[:, Y1], color="m", label = 'real_x1y1', alpha=real_alpha)

# plot top left, bottom right coordinates of predicted path
ax.plot(fpath[:, X2], fpath[:, Y2], color="c",label = 'forecast_x2y2', alpha=fore_alpha)
ax.plot(fpath[:, X1], fpath[:, Y1], color="b",label = 'forecast_x1y1', alpha=fore_alpha)

print(len(fpath), len(rpath), first_f, len(forecasts), len(bboxes))
lines = np.stack([fpath[:, 0:2], rpath[first_f+1:, 0:2]], axis=1)
ax.add_collection(LineCollection(lines, colors=['r'],alpha=diff_alpha, label = "diff_x1y1"))

lines2 = np.stack([fpath[:, 2:4], rpath[first_f+1:, 2:4]], axis=1)
ax.add_collection(LineCollection(lines2, colors=['orange'], alpha=diff_alpha, label = "diff_x2y2"))
#ax.set_xlim([0,1920])
#ax.set_ylim([-200,3500])
ax.plot(np.linspace(0,1920,num=5),np.zeros(5),linestyle='--', color='r')

ax.legend()
print(trial_data)
##################################Plot all preictions per timestep
def predictions_per_step(forecasts):
    first_forecast = next(i for i, fc in enumerate(forecasts) if fc is not None)
    forecast_len = forecasts[first_forecast].shape[0]
    preds = np.empty((len(forecasts) - first_forecast, forecast_len, 4))
    preds[:] = np.nan

    for i in range(preds.shape[0]):
        for j in range(max(0, i - forecast_len + 1), i + 1):
            f = forecasts[j]
            if f is not None:
                preds[i, i - j] = f[i - j]

    return preds
##################################
real_alpha = 0.5
fore_alpha = 0.5

preds = predictions_per_step(forecasts)
avgs = np.mean(preds, axis=1)
rpath = bboxes[first_f+1:]
npath = preds[:, 0]
plt.figure(figsize=(10,10))
plt.plot(rpath[:, X2], rpath[:, Y2], color='b', alpha=real_alpha, label='real')
plt.plot(avgs[:, X2], avgs[:, Y2], color='g',alpha=fore_alpha, label='forecast')
#plt.plot(npath[:, 0], npath[:, 1], color='r')
plt.axis('equal')
plt.legend()
plt.plot(np.linspace(0,1920,num=5),np.zeros(5),linestyle='--', color='r')
##################################

##################################

##################################

##################################

# Unused stuff from seq2seq_predict.py

class LSTMdense(nn.Module):
    def __init__(
            self,
            output_seq_size,
            embedding_size=None,
            hidden_size=128,
            LSTM_layers=2,
            dropout=0.0,
    ):
        """
        Implementation of RED predictor - LSTM for encoding and Linear layer for decoding.
        "RED: A simple but effective Baseline Predictor for the TrajNet Benchmark"
        From pedestrian trajectory prediction literature

        :param output_seq_size: forecast length
        :param embedding_size: output dimension of linear embedding layer of input (default: None)
        :param hidden_size: dimension of hidden vector in each LSTM layer
        :param LSTM_layers: number of LSTM layers
        :param dropout: dropout normalization probability (default: 0)
        """
        super(LSTMdense, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        self.output_seq_size = output_seq_size
        self.output_dim = 4 * output_seq_size

        if embedding_size is not None:
            self.embedding_encoder = nn.Linear(in_features=4, out_features=embedding_size)
        else:
            embedding_size = 4
            self.embedding_encoder = None

        self.dropout = nn.Dropout(dropout)

        self.LSTM = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=LSTM_layers,
                            dropout=dropout, )  # TODO: add batch first and remove transpose from forward?

        self.out_dense = nn.Linear(in_features=hidden_size, out_features=self.output_dim)

    def forward(self, input_seq):
        # take the last coordinates from the input sequence and tile them to the output length shape, switch dims 0,1
        offset = input_seq[:, -1].repeat(self.output_seq_size, 1, 1).transpose(0, 1)

        # compute diffs
        diffs = input_seq[:, 1:] - input_seq[:, :-1]

        if self.embedding_encoder is not None:
            diffs = self.embedding_encoder(diffs)

        inp = self.dropout(diffs)

        # ignores output (0) and cell (1,1)
        _, (h_out, _) = self.LSTM(inp.transpose(0, 1))

        output = self.out_dense(h_out[-1])  # take hidden state of last layer
        output_mat = output.view(-1, self.output_seq_size, 4)

        # add the offset to the deltas output
        return offset + output_mat


class GRUEncDec(nn.Module):
    def __init__(
            self,
            output_seq_size=20,
            hidden_size=64,
            GRU_layers=1,
            dropout=0.0,
            tie_enc_dec=False,
            use_gru_cell=False
    ):
        """
        Encoder-decoder architechture with GRU cells as encoder and decoder
        :param output_seq_size: forecast length (defualt: 20)
        :param hidden_size: dimension of hidden state in GRU cell (defualt: 64)
        :param GRU_layers: number of GRU layers (defualt: 1)
        :param dropout: probablity of dropout of input (default: 0)
        :param tie_enc_dec: Boolean, whether to use the same parameters in the encoder and decoder (default: False)
        :param use_gru_cell: Boolean, whether to use the nn.GRUCell class instead of nn.GRU (default: False)
        """
        super(GRUEncDec, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.tie_enc_dec = tie_enc_dec
        self.use_gru_cell = use_gru_cell

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(input_size=4,
                                 hidden_size=hidden_size,
                                 num_layers=GRU_layers,
                                 batch_first=True,
                                 )

        if not tie_enc_dec:
            if use_gru_cell:
                self.decoderGRU = nn.GRUCell(input_size=4,
                                             hidden_size=hidden_size,
                                             )
            else:
                self.decoderGRU = nn.GRU(input_size=4,
                                         hidden_size=hidden_size,
                                         num_layers=GRU_layers,
                                         batch_first=True,
                                         )
        else:
            self.decoderGRU = self.encoderGRU

        self.linear = nn.Linear(in_features=hidden_size, out_features=4)

    def forward(self, input_seq):
        offset = input_seq[:, -1][:, None, :]
        input_vels = input_seq[:, 1:] - input_seq[:, :-1]

        input_vels = self.dropout_layer(input_vels)

        _, hn = self.encoderGRU(input_vels)
        out_list = []

        # prev_x = input_seq[:, -1]
        vel = input_vels[:, -1]
        # prev_x = torch.zeros(diffs[:, -1].size()).to(self.device) # doesn't seem to make a difference...

        if self.use_gru_cell:
            hn = hn[0]

        for i in range(self.output_seq_size):
            vel = self.dropout_layer(vel)
            if self.use_gru_cell:
                hn = self.decoderGRU(vel, hn)
                vel = self.linear(hn)
            else:
                _, hn = self.decoderGRU(vel.unsqueeze(1), hn)
                vel = self.linear(hn[-1])

            # x = vel + prev_x
            out_list.append(vel.unsqueeze(1))

        out = torch.cat(out_list, dim=1)
        # add the deltas to the last location
        # cumsum marginally improves generalization
        return out.cumsum(dim=1) + offset


class GRUEncDecSched(nn.Module):
    def __init__(
            self,
            output_seq_size=20,
            hidden_size=64,
            GRU_layers=1,
            dropout=0.0,
            tie_enc_dec=False,
            use_gru_cell=False
    ):
        """
        Encoder-decoder architechture with GRU cells as encoder and decoder
        :param output_seq_size: forecast length (defualt: 20)
        :param hidden_size: dimension of hidden state in GRU cell (defualt: 64)
        :param GRU_layers: number of GRU layers (defualt: 1)
        :param dropout: probablity of dropout of input (default: 0)
        :param tie_enc_dec: Boolean, whether to use the same parameters in the encoder and decoder (default: False)
        :param use_gru_cell: Boolean, whether to use the nn.GRUCell class instead of nn.GRU (default: False)
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.tie_enc_dec = tie_enc_dec
        self.use_gru_cell = use_gru_cell

        self.epsi = 0
        self.target = None

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(input_size=4,
                                 hidden_size=hidden_size,
                                 num_layers=GRU_layers,
                                 batch_first=True,
                                 )

        if not tie_enc_dec:
            if use_gru_cell:
                self.decoderGRU = nn.GRUCell(input_size=4,
                                             hidden_size=hidden_size,
                                             )
            else:
                self.decoderGRU = nn.GRU(input_size=4,
                                         hidden_size=hidden_size,
                                         num_layers=GRU_layers,
                                         batch_first=True,
                                         )
        else:
            self.decoderGRU = self.encoderGRU

        self.linear = nn.Linear(in_features=hidden_size, out_features=4)

    def forward(self, input_seq):
        offset = input_seq[:, -1].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        input_vels = input_seq[:, 1:] - input_seq[:, :-1]

        input_vels = self.dropout_layer(input_vels)
        if self.epsi:
            vel_target = self.target[:, 1:] - self.target[:, :-1]

        _, hn = self.encoderGRU(input_vels)
        out_list = []

        # prev_x = input_seq[:, -1]
        vel = input_vels[:, -1]
        # prev_x = torch.zeros(diffs[:, -1].size()).to(self.device) # doesn't seem to make a difference...

        if self.use_gru_cell:
            hn = hn[0]

        for i in range(self.output_seq_size):

            vel = self.dropout_layer(vel)

            if self.epsi and i > 0:
                coins = torch.rand(input_seq.shape[0])
                take_true = (coins < self.epsi).unsqueeze(1).to(self.device)
                truths = take_true * vel_target[:, i - 1]
                model_preds = (~take_true) * vel
                vel = truths + model_preds

            if self.use_gru_cell:
                hn = self.decoderGRU(vel, hn)
                vel = self.linear(hn)
            else:
                _, hn = self.decoderGRU(vel.unsqueeze(1), hn)
                vel = self.linear(hn[-1])

            # x = vel + prev_x
            out_list.append(vel.unsqueeze(1))

        out = torch.cat(out_list, dim=1)
        # add the deltas to the last location
        # cumsum marginally improves generalization
        return out.cumsum(dim=1) + offset


class GRUEncDecPosVel(nn.Module):
    """
    Encoder-decoder architechture with GRU cells as encoder and decoder
    """

    def __init__(
            self,
            output_seq_size=20,
            hidden_size=64,
            GRU_layers=1,
            dropout=0.0,
            tie_enc_dec=False,
            use_gru_cell=False
    ):
        """
        :param output_seq_size: forecast length (defualt: 20)
        :param hidden_size: dimension of hidden state in GRU cell (defualt: 64)
        :param GRU_layers: number of GRU layers (defualt: 1)
        :param dropout: probablity of dropout of input (default: 0)
        :param tie_enc_dec: Boolean, whether to use the same parameters in the encoder and decoder (default: False)
        :param use_gru_cell: Boolean, whether to use the nn.GRUCell class instead of nn.GRU (default: False)
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size
        self.tie_enc_dec = tie_enc_dec
        self.use_gru_cell = use_gru_cell

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(input_size=8,
                                 hidden_size=hidden_size,
                                 num_layers=GRU_layers,
                                 batch_first=True,
                                 )

        if not tie_enc_dec:
            if use_gru_cell:
                self.decoderGRU = nn.GRUCell(input_size=8,
                                             hidden_size=hidden_size,
                                             )
            else:
                self.decoderGRU = nn.GRU(input_size=8,
                                         hidden_size=hidden_size,
                                         num_layers=GRU_layers,
                                         batch_first=True,
                                         )
        else:
            self.decoderGRU = self.encoderGRU

        self.linear = nn.Linear(in_features=hidden_size, out_features=4)

    def forward(self, input_seq):
        input_vel = input_seq[:, 1:] - input_seq[:, :-1]
        input_en = torch.cat((input_seq[:, :-1], input_vel), dim=-1)

        input_en = self.dropout_layer(input_en)

        _, hn = self.encoderGRU(input_en)
        out_list = []

        if self.use_gru_cell:
            hn = hn[0]

        x = input_seq[:, -1]
        vel = input_vel[:, -1]

        for i in range(self.output_seq_size):
            vel = self.dropout_layer(vel)
            input_dec = torch.cat((x, vel), dim=-1)

            if self.use_gru_cell:
                hn = self.decoderGRU(input_dec, hn)
                vel = self.linear(hn)
            else:
                _, hn = self.decoderGRU(input_dec.unsqueeze(1), hn)
                vel = self.linear(hn[-1])

            # x = x + vel
            out_list.append(vel.unsqueeze(1))

        out = torch.cat(out_list, dim=1)

        return out.cumsum(dim=1) + input_seq[:, -1][:, None, :]


class ConvEncoder(nn.Module):
    """
    Convolutional encoder to map cropped head image to a low dimensional vector. Trained jointly with
    the the rest of the network. Constant architechture with 2 conv layers.
    """

    def __init__(self, in_width, in_height, out_size, conv1_out_chan, conv2_out_chan):
        """
        :param in_width: image width
        :param in_height: image height
        :param out_size: 1d dimension of the embedding vector
        :param conv1_out_chan: number of kernels in conv1
        :param conv2_out_chan: number of kernels in conv2
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=conv1_out_chan, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=conv1_out_chan, out_channels=conv2_out_chan, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2)

        inp = torch.rand(1, 1, in_width, in_height)
        inp = self.conv1(inp)
        inp = self.pool(inp)
        inp = self.conv2(inp)
        inp = self.pool(inp)

        self.fc = torch.nn.Linear(inp.flatten().size(0), out_size)

        self.out_size = out_size
        self.in_width = in_width
        self.in_height = in_height

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        return self.fc(x.reshape(x.size(0), -1))


class GRUEncDecWithHead(nn.Module):
    def __init__(
            self,
            output_seq_size=20,
            hidden_size=64,
            GRU_layers=1,
            dropout=0.0,
            head_embedder=ConvEncoder(32, 32, 5, 4, 10),
    ):
        """
        Encoder-decoder architechture with GRU cells as encoder and decoder
        :param output_seq_size: forecast length (defualt: 20)
        :param hidden_size: dimension of hidden state in GRU cell (defualt: 64)
        :param GRU_layers: number of GRU layers (defualt: 1)
        :param dropout: probability of dropout of input (default: 0)
        :param tie_enc_dec: Boolean, whether to use the same parameters in the encoder and decoder (default: False)
        :param use_gru_cell: Boolean, whether to use the nn.GRUCell class instead of nn.GRU (default: False)
        :param head_embedder: initialized torch.nn module to map cropped head image to a low dimensional vector
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_seq_size = output_seq_size
        self.hidden_size = hidden_size

        self.dropout_layer = torch.nn.Dropout(dropout)

        self.encoderGRU = nn.GRU(input_size=4 + head_embedder.out_size,
                                 hidden_size=hidden_size,
                                 num_layers=GRU_layers,
                                 batch_first=True,
                                 )

        self.decoderGRU = nn.GRUCell(input_size=4,
                                     hidden_size=hidden_size)

        self.linear = nn.Linear(in_features=hidden_size, out_features=4)

        self.head_embedder = head_embedder

    def forward(self, bbox_seq, head_seq):
        offset = bbox_seq[:, -1].repeat(self.output_seq_size, 1, 1).transpose(0, 1)
        diffs = bbox_seq[:, 1:] - bbox_seq[:, :-1]

        diffs = self.dropout_layer(diffs)

        # drop last head image, unnecessary for diffs
        head_seq = head_seq[:, :-1]

        # reshape tensor for batch inference of images
        head_input = head_seq.unsqueeze(2).reshape(-1, 1, head_seq.size(-2), head_seq.size(-1))
        head_embedding = self.head_embedder(head_input)

        # reshape tensor back to (batch, sequence) dimensions.
        head_embedding = head_embedding.reshape(diffs.size(0), diffs.size(1), -1)

        _, hn = self.encoderGRU(torch.cat([diffs, head_embedding], dim=2))
        out_list = []

        # prev_x = input_seq[:, -1]
        prev_x = diffs[:, -1]
        # prev_x = torch.zeros(diffs[:, -1].size()).to(self.device) # doesn't seem to make a difference...

        hn = hn[0]

        for i in range(self.output_seq_size):
            hn = self.decoderGRU(prev_x, hn)
            lin = self.linear(hn)

            x = lin + prev_x
            out_list.append(x.unsqueeze(1))
            prev_x = x

        out = torch.cat(out_list, dim=1)
        # add the deltas to the last location
        # cumsum marginally improves generalization
        return out.cumsum(dim=1) + offset


class VelLinear(nn.Module):
    """
    A baseline linear fully-connected model with one hidden layer and RELU activation.
    """

    def __init__(
            self,
            input_size=4,
            output_size=4,
            input_seq_size=20,
            output_seq_size=20,
            hidden_size=64,
            dropout=0.0,
    ):
        super(VelLinear, self).__init__()

        self.output_size = output_size
        self.output_seq_size = output_seq_size
        self.input_seq_size = output_seq_size

        self.dropout_layer = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Linear(
            in_features=input_size * (input_seq_size - 1), out_features=hidden_size
        )
        self.decoder = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size * output_seq_size
        )

    def forward(self, input_seq):
        diffs = input_seq[:, 1:] - input_seq[:, :-1]

        x = self.dropout_layer(diffs)
        x = self.encoder(x.view(x.shape[0], -1))
        x = torch.nn.functional.relu(x)
        x = self.decoder(x)

        out = x.view(x.shape[0], self.input_seq_size, self.output_size)

        return out + input_seq[:, -1][:, None, :]


# Old functions for finding calibration homography using black shapes. Funding the shapes was not robust and
# consistent, with many FP and FN, so we replaced them with the Aruco markers. Only old vidoes have these markers,
#and their homographies are already computed.
# code taken from:
#    https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/

# TODO: old black squares function
def get_points(polygons, max_y=True):
    """
    Return the rightmost and leftmost upper points (closest to the screen) of two squares.
    """

    if any((polygons[0] - polygons[1])[:, 0] < 0):
        right, left = polygons[1], polygons[0]
    else:
        right, left = polygons[0], polygons[1]

    min_xs = np.argsort(right[:, 0])
    if max_y:
        p_right_ind = np.argmax(right[min_xs][:2, 1])
    else:
        p_right_ind = np.argmin(right[min_xs][:2, 1])

    max_xs = np.argsort(left[:, 0])[::-1]
    if max_y:
        p_left_ind = np.argmax(left[max_xs][:2, 1])
    else:
        p_left_ind = np.argmin(left[max_xs][:2, 1])

    return right[min_xs][p_right_ind], left[max_xs][p_left_ind]


# TODO: old black squares function
def thresh_dist(poly, min_thresh, max_thresh):
    """
    Return True if the distance between each pair of points is larger than
    min_thresh and smaller than max_thresh.
    """
    for i, p1 in enumerate(poly):
        for j, p2 in enumerate(poly[i + 1:]):
            norm = np.linalg.norm(p1 - p2)
            if norm < min_thresh or norm > max_thresh:
                return False
    return True


# TODO: old black squares function
def polygons_min_distance(polygons, min_dist=300):
    """
    Checks if the polygons centers are mutually far away from another, in case some reflections
    are detected by mistake.
    :param polygons: a list of polygons, numpy arrays each with 4 edges
    :param min_dist: minimal L2 distance between polygons
    :return: True if polygons are too close, else False
    """

    centroids = np.empty((len(polygons), 2))
    for i, poly in enumerate(polygons):
        centroids[i] = poly.mean(axis=0)

    for i, cent in enumerate(centroids):
        for j, cent2 in enumerate(centroids[i + 1:]):
            dist = np.linalg.norm(cent - cent2)
            if dist < min_dist:
                return True
    return False


# TODO: Old function, operates on black squares
def find_arena_homography_black_squares(
        cal_img,
        screen_x_res=1920,
        contrast=2.4,
        brightness=0,
        min_near_edge_size=30,
        min_far_edge_size=10,
        max_near_edge_size=100,
        max_far_edge_size=100,
        near_far_y_split=700,
        min_dist=300,
):
    """
    Calculate the homography matrix to map from camera coordinates to
    a coordinate system relative to the touch screen.

    Assumes cal_img contains 4 visually clear black squares marking the screen edges
    and the rear end of the arena.
    Assumes the image is corrected for lense distortion.
    Finds the innermost right and left points that are closest to the screen and
    returns the transformation and an image with the features highlighted.
    :return: homography H, labelled image, screen length in image pixels
    """

    img = cv.cvtColor(cal_img.copy(), cv.COLOR_BGR2GRAY)
    img = cv.convertScaleAbs(img, alpha=contrast, beta=brightness)

    _, threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    near_polygons = []
    far_polygons = []

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue

        approx = approx.squeeze()

        if all(approx[:, 1] > near_far_y_split):
            if thresh_dist(approx, min_near_edge_size, max_near_edge_size):
                near_polygons.append(approx)
                cv.drawContours(img, [approx], 0, (255, 255, 0), 5)
            else:
                cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
        else:
            if thresh_dist(approx, min_far_edge_size, max_far_edge_size):
                far_polygons.append(approx)
                cv.drawContours(img, [approx], 0, (0, 0, 255), 5)
            else:
                cv.drawContours(img, [approx], 0, (255, 0, 0), 5)

    if len(near_polygons) != 2 or len(far_polygons) != 2:
        return None, img, "Could not find 2 far and 2 near square marks in the image."

    if polygons_min_distance(near_polygons, min_dist):
        return None, img, "Some of the near polygons are too close to each other."

    if polygons_min_distance(far_polygons, min_dist):
        return None, img, "Some of the far polygons are too close to each other."

    p_bottom_r, p_bottom_l = get_points(near_polygons)
    p_top_r, p_top_l = get_points(far_polygons, max_y=False)

    # Draw the screen line.
    cv.line(img, pt1=tuple(p_bottom_r), pt2=tuple(p_bottom_l), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_bottom_r), pt2=tuple(p_top_r), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_bottom_l), pt2=tuple(p_top_l), color=(0, 255, 0), thickness=10)
    cv.line(img, pt1=tuple(p_top_r), pt2=tuple(p_top_l), color=(0, 255, 0), thickness=10)

    arena_h_pixels = screen_x_res * (ARENA_H_CM / ARENA_W_CM)
    dst_p = np.array([[0, 0],
                      [screen_x_res, 0],
                      [0, arena_h_pixels],
                      [screen_x_res, arena_h_pixels]])
    src_p = np.vstack([p_bottom_r, p_bottom_l, p_top_r, p_top_l]).astype(np.float64)

    homography, _ = cv.findHomography(src_p, dst_p)

    return homography, img, None

# another function from dataset.py

# Old function, works with black squares and not Aruco markers.
def get_homography_from_video_black_squares(
    video_path, undist_alpha=0, max_test_frames=50, **homography_args
):
    """
    An offline function to get the homography transformation from a video. Grid searching values of brightness and
    contrast until the correct number of polygons is found
    :param video_path: path to video
    :param undist_alpha: alpha paramater to the lense undistortion function
    :param max_test_frames: maximal number of random frames to try
    :param homography_args: arguments for find_arena_homography function
    :return: homography, or None if no homography found
    """
    vcap = cv.VideoCapture(video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))

    (mapx, mapy), roi, _ = calib.get_undistort_mapping(
        width, height, calib.MTX, calib.DIST, alpha=undist_alpha
    )

    error = None

    # Try random video frames until one works.
    for _ in range(max_test_frames):
        ret, frame = vcap.read()
        if not ret:
            raise DatasetException("Error reading frame.")

        undistorted_img = calib.undistort_image(frame, (mapx, mapy))

        # Try a few contrast and brightness values until one works.
        for contrast in np.arange(0, 3.1, 0.1):
            for brightness in np.arange(-40, 20, 5):
                (
                    homography,
                    homography_im,
                    error,
                ) = calib.find_arena_homography_black_squares(
                    undistorted_img,
                    contrast=contrast,
                    brightness=brightness,
                    **homography_args,
                )
                if homography is not None:
                    print(
                        f"Found homography with brightness: {brightness}, contrast: {contrast}"
                    )
                    break

            if homography is not None:
                break

        if homography is not None:
            break
        else:
            random_frame = random.randint(0, num_frames)
            vcap.set(cv.CAP_PROP_POS_FRAMES, random_frame)

    if error is not None:
        print("Could not find homography:", error)

    vcap.release()

    return homography, homography_im

# Maybe delete, not necessary. If not, create similiar Aruco function
def save_homography_data(path, undist_alpha=0, homography_args={}):
    """
    TODO can
    :param path:
    :param undist_alpha:
    :param homography_args:
    :return:
    """
    vid_path = os.path.join(path, REALTIME_ID + ".avi")
    rt_data_path = os.path.join(path, RT_DATA_FOLDER)
    json_fn = os.path.join(rt_data_path, VID_STATS_FN)
    homography_im_fn = os.path.join(rt_data_path, HOMOGRAPHY_IM_FN)

    homography, homography_im = get_homography_from_video_black_squares(
        vid_path, undist_alpha=undist_alpha, **homography_args
    )

    if not os.path.exists(rt_data_path):
        os.mkdir(rt_data_path)

    if homography is not None:
        homography = homography.tolist()

    if homography_im is not None:
        cv.imwrite(homography_im_fn, homography_im)

    vcap = cv.VideoCapture(vid_path)
    vid_width = int(vcap.get(3))
    vid_height = int(vcap.get(4))
    vcap.release()

    with open(json_fn, "w") as fp:
        vid_stats = {"width": vid_width, "height": vid_height, "homography": homography}
        if undist_alpha != 0:
            vid_stats["undist_alpha"] = undist_alpha

        json.dump(vid_stats, fp)


""" ###############  Head images data functions from the dataset module ###############  """

def get_cropped_dict(vid_dims, first_date, all_path=EXPERIMENTS_ROOT):
    heads_dict = dict()
    for exper in glob.glob(all_path + "*"):

        if not os.path.isdir(exper):
            continue

        exper_date = ret_date(exper)
        if exper_date < first_date:
            continue

        # ignore words
        if any([dont in exper for dont in EXP_DONT]):
            print(f"skipped {exper}, ignored word")
            continue

        exper_log = parse_exper_log(exper)
        exper_name = os.path.split(exper)[-1]
        heads_dict[exper_name] = dict()

        for k in range(1, exper_log["num_trials"] + 1):
            try:
                rt_data_path = os.path.join(exper, f"trial{k}", RT_DATA_FOLDER)

                json_fn = os.path.join(rt_data_path, VID_STATS_FN)
                with open(json_fn, "r") as fp:
                    vid_stat = json.load(fp)

                if vid_stat["width"] != vid_dims[0]:
                    print(
                        f'ignored {exper} trial{k}, {vid_stat["width"]} != {vid_dims[0]}'
                    )
                    continue

                head_crops_fn = os.path.join(rt_data_path, HEAD_CROPS_FN)

                with open(head_crops_fn, "rb") as fp:
                    heads_dict[exper_name][k] = pickle.load(fp)
            except FileNotFoundError:
                continue

    for key in list(heads_dict.keys()):
        if len(heads_dict[key].keys()) < 1:
            heads_dict.pop(key)

    return heads_dict


def heads_list2mat(l, resize):
    flt_imgs = np.empty((len(l), resize ** 2))
    flt_imgs[:] = np.nan

    for i, img in enumerate(l):
        if img is not None:
            resized_img = cv.resize(img, (resize, resize))
            flt_imgs[i, :] = resized_img.flatten()
    return flt_imgs.astype(
        "uint8"
    )  # TODO returning as uint8 converts np.nan to zero (0)


def get_unified_heads_mat(vid_dims, first_date, resize=32, all_path=EXPERIMENTS_ROOT):
    """
    Generating the full matrix from 25~ trials is 1GB, should not use or maybe restrict sizes
    or number of trials
    """
    heads_dict = get_cropped_dict(vid_dims, first_date, all_path)
    mat_list = []

    for key in heads_dict.keys():
        for trial in heads_dict[key].keys():
            mat_list.append(heads_list2mat(heads_dict[key][trial], resize))
    return np.concatenate(mat_list).astype("uint8")

""" ---------- From train_eval ------------- """


def grid_input_output(
    model_name,
    df,
    input_seqs,
    output_seqs,
    input_labels,
    output_labels,
    path,
    num_epochs=5000,
):
    """
    Perform 2D grid search over cartesian product of input and output sequence lengths with labels
    assumes input_seqs and output_seqs are iterables
    :return: pandas df with ADE scores
    """

    scores = pd.DataFrame(index=input_seqs, columns=output_seqs)
    train, val, test = create_train_val_test_splits(df.index.unique(), [0.7, 0.2, 0.1])
    count = 0
    num_trains = len(input_seqs) * len(output_seqs)

    for inp_seq in input_seqs:
        for out_seq in output_seqs:
            count += 1
            print("================================================")
            print(
                f"{count}/{num_trains} Training with input_seq_len={inp_seq}, output_seq_len={out_seq}"
            )

            train_dl, val_dl, test_dl = create_train_val_test_dataloaders(
                df, train, val, test, input_labels, output_labels, inp_seq, out_seq
            )

            net = model_name(len(input_labels), len(output_labels), out_seq)

            _, best_ADE = train_trajectory_model(
                net,
                train_dl,
                val_dl,
                num_epochs,
                path,
                eval_freq=100,
                model_name=f"model_{inp_seq}_{out_seq}",
            )

            scores.loc[inp_seq, out_seq] = best_ADE

    return scores