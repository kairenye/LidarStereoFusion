'''
This script is a slightly more sophisticated way of naively fusing lidar and
stereo disparities. However, the performance is slower.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import glob
from scipy import io as sio

from PIL import Image, ImageDraw
from collections import defaultdict
from math import ceil, floor

# fn = "data/2011_09_26_2/2011_09_26_drive_0005_sync/velodyne_points/data/0000000010.bin"
# velo = np.fromfile(fn, dtype=np.float32).reshape(-1,4)

# R = np.array([7.533745e-03,-9.999714e-01,-6.166020e-04,1.480249e-02 ,7.280733e-04 ,-9.998902e-01 ,9.998621e-01 ,7.523790e-03 ,1.480755e-02])
# R = R.reshape(3,3)
# T = np.array([-4.069766e-03 ,-7.631618e-02 ,-2.717806e-01])
# T = T.reshape(3,1)
# tmp = np.array([0,0,0,1])
# TR_velo_to_cam = np.vstack((np.hstack((R,T)), tmp))
# R_0 = np.array([1.000000e+00 ,0.000000e+00 ,0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,0.000000e+00 ,0.000000e+00 ,0.000000e+00 ,1.000000e+00])
# R_0 = R_0.reshape(3,3)
# R_0 = np.hstack((R_0,np.zeros((3,1))))
# R_0 = np.vstack((R_0,np.zeros((1,4))))
# R_0[3,3] = 1

# P_rect_00 = np.array([7.215377e+02 ,0.000000e+00 ,6.095593e+02 ,0.000000e+00 ,0.000000e+00 ,7.215377e+02 ,1.728540e+02 ,0.000000e+00 ,0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,0.000000e+00])
# P_rect_00 = P_rect_00.reshape(3,4)


#fig = plt.figure()
#ax = plt.axes(projection='3d')

#ax.scatter(velo[:,0], velo[:,1], velo[:,2], marker='.', s = 0.1)

points = 0.2
points_step = int(1. / points)
point_size = 0.01 * (1. / points)
# velo_range = range(0, velo.shape[0], points_step)
# velo_frame = velo[velo_range, :]


axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]

axes_str = ['X', 'Y', 'Z']

'''
Draws 3D point cloud data using matplotlib.plt
'''
def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):

        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)



#f2 = plt.figure(figsize=(15, 8))
#ax2 = f2.add_subplot(111, projection='3d')
#draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))



#f, ax3 = plt.subplots(2, 1, figsize=(15, 25))
# draw_point_cloud(
#     ax3[0],
#     'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right',
#     axes=[0, 2] # X and Z axes
# )
# draw_point_cloud(
#     ax3[1],
#     'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right',
#     axes=[0, 1] # X and Y axes
#)
# draw_point_cloud(
#     ax3[0],
#     'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane',
#     axes=[1, 2] # Y and Z axes
# )
#plt.show()

'''
Replaces the reflectance value by 1, and tranposes the array, so
points can be directly multiplied by the camera projection matrix
Point with zero reflectance values are filtered out.
'''
def prepare_velo_points(pts3d_raw):


    pts3d = pts3d_raw
    # Reflectance > 0
    pts3d = pts3d[pts3d[:, 3] > 0 ,:]
    pts3d[:,3] = 1 #homogenous coordinate
    return pts3d.transpose()

'''
Project 3D points into 2D image. Expects pts3d as a 4xN
numpy array. Returns the 2D projection of the points that
are in front of the camera only and the corresponding 3D points.
'''
def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):

    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts3d_cam = pts3d_cam[:,idx]
    dist = np.sqrt(pts3d_cam[0,:]**2+pts3d_cam[1,:]**2+pts3d_cam[2,:]**2)
    pts2d_cam = Prect.dot(pts3d_cam)

    return dist, pts3d[:, idx], pts2d_cam/pts2d_cam[2,:]

# pts3d_t = prepare_velo_points(velo)
# dist, _, pts = project_velo_points_in_img(pts3d_t, TR_velo_to_cam, R_0, P_rect_00)
# pts = pts[0:2, :].T


# c = (dist - np.min(dist)) / (np.max(dist)-np.min(dist)) * 255

'''
An exploratory function for drawing lidar points on image using ImageDraw.
'''
def draw_pts_on_image():
    img = Image.open(data_folder/"2011_09_26/2011_09_26_drive_0005_sync/image_00/data/0000000010.png").convert("RGB")
    #print(img.size)
    img_draw = ImageDraw.Draw(img)
    for i, point in enumerate(pts):
      img_draw.point(point,fill="rgb({},{},{})".format(int(c[i]),0,0))

    img.show()
# for obj in data:
#     plt.text(obj[1],obj[0],obj[2])      # change x,y as there is no view() in mpl

'''
Plots lidar points projected onto 2D image plane using pyplot.
'''
def plot_2d_pyplot():
    plt.figure()
    plt.axis([0, 1242, 0, 375])
    plt.grid(False)                         # set the grid
    cm = plt.get_cmap("hot")
    plt.scatter(pts[:,0],pts[:,1],s=0.1,marker='o',c=c, cmap=cm)
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
        #ax.yaxis.set_ticks(np.arange(0, 375, 1)) # set y-ticks
    ax.yaxis.tick_left()

'''
Plots 3D lidar points in 3D space.
'''
def plot_3d_camera_colormap():
    ax = plt.axes(projection='3d')
    cm = plt.get_cmap("cool")
    ax.scatter(pts[:,0], dist, (375-pts[:,1]), marker='.', s = 0.1, c=c, cmap=cm)
    ax.set_xlim3d([0,1242])
    ax.set_ylim3d([0,60])
    ax.set_zlim3d([0,375])

'''
Plots projected 2D lidar points onto image plane along with the original image.
The original image is shown with a reduced alpha so that the points can better
be seen. This function is useful for verifying that the projection is working
correctly.
'''
def draw_pts_as_cmp_on_image_pyplot(img, pts, dist):
    #img=mpimg.imread("data/2011_09_26_2/2011_09_26_drive_0005_sync/image_00/data/0000000010.png")
    #print(img.shape)
    c = (dist - np.min(dist)) / (np.max(dist)-np.min(dist)) * 255
    imgplot = plt.imshow(img,cmap='gray',alpha=0.8)
    plt.axis([0, 1242, 0, 375])
    plt.grid(False)                         # set the grid
    cm = plt.get_cmap("hot")
    plt.scatter(pts[:,0],pts[:,1],s=0.1,marker='o',c=c, cmap=cm)
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
        #ax.yaxis.set_ticks(np.arange(0, 375, 1)) # set y-ticks
    ax.yaxis.tick_left()
    plt.show()

'''
This function filters the points and only returns the ones that are in the
image window.
'''
def filter_pts_to_image_window(points, dist, width, height):
    mask = points[:,0] <= width
    points = points[mask, :]
    dist = dist[mask]
    mask = points[:,1] <= height
    points = points[mask, :]
    dist = dist[mask]
    return points, dist

'''
This function computes the stereo disparity using OpenCV's StereoSGBM. All the
parameters have been tuned for the KITTI dataset.
'''
def compute_stereo_disparity(imgL, imgR):
    stereo = cv.StereoSGBM_create(minDisparity = 1,
            numDisparities = 128,
            blockSize = 11,
            uniquenessRatio = 5,
            speckleRange = 1,
            speckleWindowSize = 100,
            disp12MaxDiff = -200,
            P1 = 1000, #1000
            P2 = 3000 #3000

            # SADWindowSize = 6,
            # uniquenessRatio = 10,
            # speckleWindowSize = 100,
            # speckleRange = 32,
            # disp12MaxDiff = 1,
            # P1 = 8*3*3**2,
            # P2 = 32*3*3**2,
            # fullDP = False
        )

    #imgL = cv.imread('data/2011_09_26_2/2011_09_26_drive_0005_sync/image_02/data/0000000010.png')
    #imgR = cv.imread('data/2011_09_26_2/2011_09_26_drive_0005_sync/image_03/data/0000000010.png')
    #disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disp = stereo.compute(imgL, imgR) * 16

    return disp

    #cv.imshow("disparity", (disp)/(128))


    #baseline*focal/disparity = 0.54*721/disp

    #cv.waitKey()

    # mask = disp > 1e-8
    # depth = np.zeros(disp.shape)
    # depth[mask] = 0.54*721/disp[mask]


    # ax = plt.axes(projection='3d')
    # cm = plt.get_cmap("cool")

    # for i in range(1242):
    #     for j in range(375):
    #         ax.scatter(i,j,depth[j,i],marker='.',s=0.1,cmap=cm)
    # ax.set_xlim3d([0,1242])
    # ax.set_ylim3d([0,60])
    # ax.set_zlim3d([0,375])
    #print(np.unravel_index(np.argmax(depth),depth.shape))
    # plt.figure()
    # plt.imshow(depth)

    # plt.show()

baseDir = "/Users/kairenye/Desktop/lidarstereo_data/"
lidarList = glob.glob(baseDir + "lidarstereonet_test/*.bin")
imgLList = glob.glob(baseDir + "lidarstereonet_test/*_l.png")
imgRList = glob.glob(baseDir + "lidarstereonet_test/*_r.png")
calibList = glob.glob(baseDir + "lidarstereonet_test/*_calib.mat")
gtList = glob.glob(baseDir + "lidarstereonet_test/*_gt.png")
lidarList.sort()
imgLList.sort()
imgRList.sort()
calibList.sort()
gtList.sort()

numIters = 0

cumErr = np.array([0.0, 0.0, 0.0])
ScumErr = np.array([0.0, 0.0, 0.0])
cumRel_err = 0.0
ScumRel_err = 0.0
cumDelta = 0.0
ScumDelta = 0.0
cumDensity = 0.0
ScumDensity = 0.0

tau0 = [2.0, 3.0, 5.0]
tau1 = 0.05

for i in range(0,len(lidarList)):
    velo = np.fromfile(lidarList[i], dtype=np.float32).reshape(-1,4)
    mat = sio.loadmat(calibList[i])
    TR_velo_to_cam = mat["TR_velo_to_cam"]
    R = mat["R"]
    P = mat["P"]
    pts3d_t = prepare_velo_points(velo)
    dist, _, pts = project_velo_points_in_img(pts3d_t, TR_velo_to_cam, R, P)
    pts = pts[0:2, :].T
    pts, dist = filter_pts_to_image_window(pts, dist, 1242, 375)
    imgL = cv.imread(imgLList[i])
    imgR = cv.imread(imgRList[i])
    disp = compute_stereo_disparity(imgL, imgR)
    #cv.imwrite("../test_stereo_disp.png", disp.astype(np.uint16))

    # draw_pts_as_cmp_on_image_pyplot(imgL, pts, dist)
    # input("")
    lidar_disp = 0.54*721/dist
    lidar_disp = np.round(256*lidar_disp)
    # print(len(lidarList))
    # print(disp.shape)
    # print(pts.shape)
    # print(lidar_disp.shape)

    lidarDict = defaultdict(list)
    for k in range(0, pts.shape[0]):
        pt_x, pt_y = pts[k]
        lidarDict[(ceil(pt_x), ceil(pt_y))].append(lidar_disp[k])
        lidarDict[(ceil(pt_x), floor(pt_y))].append(lidar_disp[k])
        lidarDict[(floor(pt_x), ceil(pt_y))].append(lidar_disp[k])
        lidarDict[(floor(pt_x), floor(pt_y))].append(lidar_disp[k])
        #print(lidarDict)
        #input("")

    dispMap = np.zeros((disp.shape[0], disp.shape[1]))
    for m in range(disp.shape[0]):
        for n in range(disp.shape[1]):
            if lidarDict[(n,m)] != []:
                dispMap[m,n] = np.median(lidarDict[(n,m)])

    #cv.imwrite("../test_lidar_disp.png", dispMap.astype(np.uint16))

    lidarMask = dispMap > 0
    #print(lidarMask)

    #METHOD for fusion: use Lidar measurements as the true disparity for all
    #pixels which have a corresponding Lidar point. For pixels that don't have
    #a Lidar measurement, take a 6px by 6px window around that pixel and average
    #the median Lidar measurements of all valid pixels in the window with the
    #stereo disparity calculation of that pixel. If neither is available, set
    #disparity to 0.
    for m in range(disp.shape[0]):
        for n in range(disp.shape[1]):
            if lidarMask[m,n] == 0:
                windowLidar = []
                for v in range(-3, 4):
                    if m+v < 0 or m+v >= disp.shape[0]:
                        continue
                    for w in range(-3,4):
                        if n+w < 0 or n+w >= disp.shape[1]:
                            continue
                        if lidarMask[m+v, n+w] != 0:
                            windowLidar.append(dispMap[m+v, n+w])
                if windowLidar != []:
                    #print(windowLidar, disp[m,n])
                    if m >= disp.shape[0] or n >= disp.shape[1]:
                        dispMap[m,n] = np.median(windowLidar)
                    elif disp[m,n] == 0:
                         dispMap[m,n] = np.median(windowLidar)
                    else:
                        dispMap[m,n] = (disp[m,n] + np.median(windowLidar))/2.0
                else:
                    if m >= disp.shape[0] or n >= disp.shape[1]:
                        dispMap[m,n] = 0
                    else:
                        dispMap[m,n] = disp[m,n]

    #print(dispMap)
    #.imwrite("../test_fused_disp.png", dispMap.astype(np.uint16))

    estDisp = dispMap.astype(float)/256.0;
    estDisp[dispMap==0] = -1

    I = cv.imread(gtList[i], cv.IMREAD_UNCHANGED) #cv.IMREAD_GRAYSCALE)
    gtDisp = I.astype(np.float)/256.0
    gtDisp[I==0] = -1

    n_total = np.count_nonzero(gtDisp > 0)
    numIters += 1

    print("Example {}".format(numIters))

    # FUSION metrics
    E = np.abs(gtDisp - estDisp)
    d_err = np.array([0.0,0.0,0.0])
    for t in range(3):
        d_err[t] = np.count_nonzero((gtDisp > 0) * (E>tau0[t]) * ((E/gtDisp) > tau1))/n_total * 100.0
    rel_err = np.sum((gtDisp > 0) * (E/gtDisp))/n_total * 100.0
    delta1 = np.count_nonzero(((gtDisp > 0) * (estDisp > 0) * (np.maximum(estDisp/gtDisp, gtDisp/estDisp) < 1.25)))/np.count_nonzero( (gtDisp > 0) * (estDisp > 0)) * 100.0
    density = np.count_nonzero(estDisp > 0)/(disp.shape[0] * disp.shape[1]) * 100.0
    cumErr += d_err
    cumRel_err += rel_err
    cumDelta += delta1
    cumDensity += density
    print("FUSION 2px: {}, 3px: {}, 5px: {}, abs rel: {}%, delta: {}, density: {}, cum2px: {}, cum3px: {}, cum5px: {}, cumRel_Err: {}, cumDelta: {}, cumDensity: {}".format(d_err[0], d_err[1], d_err[2], rel_err, delta1, density, cumErr[0], cumErr[1], cumErr[2], cumRel_err, cumDelta, cumDensity))

    # STEREO only metrics
    estDisp = disp.astype(float)/256.0;
    estDisp[disp==0] = -1
    E = np.abs(gtDisp - estDisp)
    d_err = np.array([0.0,0.0,0.0])
    for t in range(3):
        d_err[t] = np.count_nonzero((gtDisp > 0) * (E>tau0[t]) * ((E/gtDisp) > tau1))/n_total * 100.0
    rel_err = np.sum((gtDisp > 0) * (E/gtDisp))/n_total * 100.0
    delta1 = np.count_nonzero(((gtDisp > 0) * (estDisp > 0) * (np.maximum(estDisp/gtDisp, gtDisp/estDisp) < 1.25)))/np.count_nonzero( (gtDisp > 0) * (estDisp > 0)) * 100.0
    density = np.count_nonzero(estDisp > 0)/(disp.shape[0] * disp.shape[1]) * 100.0
    ScumErr += d_err
    ScumRel_err += rel_err
    ScumDelta += delta1
    ScumDensity += density
    print("STEREO 2px: {}, 3px: {}, 5px: {}, abs rel: {}%, delta: {}, density: {}, cum2px: {}, cum3px: {}, cum5px: {}, cumRel_Err: {}, cumDelta: {}, cumDensity: {}".format(d_err[0], d_err[1], d_err[2], rel_err, delta1, density, ScumErr[0], ScumErr[1], ScumErr[2], ScumRel_err, ScumDelta, ScumDensity))

    # input("")
    #cv.imshow("disparity", (disp.astype(np.float32))/(256)/128)
    #cv.waitKey()

print(numIters)
print("FUSION")
print('Final disp error: ', cumErr/float(numIters))
print('Final relative absolute error: ', cumRel_err/float(numIters))
print('Final delta: ', cumDelta/float(numIters))
print('Final density: ', cumDensity/float(numIters))

print("STEREO")
print('Final disp error: ', ScumErr/float(numIters))
print('Final relative absolute error: ', ScumRel_err/float(numIters))
print('Final delta: ', ScumDelta/float(numIters))
print('Final delta: ', ScumDensity/float(numIters))

#print(img.shape)
#plot_3d_camera_colormap()
#compute_stereo_disparity()
#plt.scatter(img[0,:],img[1,:], s=0.01)

#plt.xlim(-300,300)
#plt.ylim(0, 1341)
#plt.show()
