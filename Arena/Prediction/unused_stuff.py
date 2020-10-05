# unused functions that we wanted to keep hanging around

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
All path trajectory from predictor eval notebook
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

