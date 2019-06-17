"""
This script projects the training lidar points to the left and right camera's
image planes (and at the same time discretizes the lidar points to the
granularity of the image pixels).
"""

import numpy as np
import glob
#import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import png

# def draw_pts_as_cmp_on_image_pyplot(img,pts, dist):
#     #img=mpimg.imread("data/2011_09_26_2/2011_09_26_drive_0005_sync/image_00/data/0000000010.png")
#     #print(img.shape)
#     c = (dist - np.min(dist)) / (np.max(dist)-np.min(dist)) * 255
#     imgplot = plt.imshow(img,cmap='gray',alpha=0.8)
#     plt.axis([0, 1242, 0, 375])
#     plt.grid(False)                         # set the grid
#     cm = plt.get_cmap("hot")
#     plt.scatter(pts[:,0],pts[:,1],s=0.1,marker='o',c=c, cmap=cm)
#     ax=plt.gca()                            # get the axis
#     ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
#     ax.xaxis.tick_top()                     # and move the X-Axis
#         #ax.yaxis.set_ticks(np.arange(0, 375, 1)) # set y-ticks
#     ax.yaxis.tick_left()
#     plt.show()

def filter_pts_to_image_window(points, dist, width, height):
    mask = (points[:,0] <= width) & (points[:,0] >= 0)
    points = points[mask, :]
    dist = dist[mask]
    mask = (points[:,1] <= height) & (points[:,1] >= 0)
    points = points[mask, :]
    dist = dist[mask]
    return points, dist

def getCalibDict(calibPath):

    calib = {}

    with open(calibPath, 'r') as f:
        for line in f.readlines():
            if ':' in line and 'calib_time' not in line:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])

    tmp = np.array([0,0,0,1])
    calib['P0'] = calib['P0'].reshape(3,4)
    calib['P1'] = calib['P1'].reshape(3,4)
    calib['P2'] = calib['P2'].reshape(3,4)
    calib['P3'] = calib['P3'].reshape(3,4)
    calib['Tr'] = np.vstack((calib['Tr'].reshape(3,4), tmp))

    return calib


def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
       points can be directly multiplied by the camera projection matrix'''

    pts3d = pts3d_raw
    # Reflectance > 0
    pts3d = pts3d[pts3d[:, 3] > 0 ,:]
    pts3d[:,3] = 1 #homogenous coordinate
    return pts3d.transpose()

def project_velo_points_in_img(pts3d, Tr, P):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''

    # 3D points in camera reference frame.
    pts3d_cam = Tr.dot(pts3d)

    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts3d_cam = pts3d_cam[:,idx]
    #dist = np.sqrt(pts3d_cam[0,:]**2+pts3d_cam[1,:]**2+pts3d_cam[2,:]**2)
    dist = pts3d_cam[2,:]
    pts2d_cam = P.dot(pts3d_cam)

    return dist, pts2d_cam/pts2d_cam[2,:]


baseDir = "/Volumes/EXT_DRIVE/lidarstereo_data/"
lidarList = glob.glob(baseDir + "vo/data_odometry_velodyne/**/**/**/**/*.bin")

totalItems = len(lidarList)

for i, lidarPath in enumerate(lidarList):
    print('{} / {}'.format(i+1, totalItems))
    #print(lidarPath)
    velo = np.fromfile(lidarPath, dtype=np.float32).reshape(-1,4)
    calibPath = '/'.join(lidarPath.split('/')[:-2]).replace('data_odometry_velodyne', 'data_odometry_calib')
    calibPath = calibPath + '/calib.txt'
    calibDict = getCalibDict(calibPath)
    imgPath = '/'.join(lidarPath.split('/')[:-2]).replace('data_odometry_velodyne', 'data_odometry_color')

    #beginning of code for projecting onto left image plane
    imgPathL = imgPath + '/image_2/' + lidarPath.split('/')[-1].replace('bin', 'png')
    imgL = cv.imread(imgPathL)
    height = imgL.shape[0]
    width = imgL.shape[1]
    pts3d_t = prepare_velo_points(velo)
    dist_L, pts2d_L = project_velo_points_in_img(pts3d_t, calibDict['Tr'], calibDict['P2'])
    pts2d_L, dist_L = filter_pts_to_image_window(pts2d_L.T, dist_L, width-1, height-1)
    f = round(calibDict['P2'][0,0])
    lidar_disp_L = 0.54*f/dist_L
    lidar_disp_L = np.round(256*lidar_disp_L)
    lidar_png_L = np.zeros((height,width)).astype(np.uint16)

    for i in range(len(lidar_disp_L)):
        if lidar_disp_L[i] > lidar_png_L[int(round(pts2d_L[i,1])), int(round(pts2d_L[i,0]))]:
            lidar_png_L[int(round(pts2d_L[i,1])), int(round(pts2d_L[i,0]))] = int(lidar_disp_L[i])

    lidarPngLPath = imgPathL.replace('data_odometry_color', 'data_odometry_projlidar')

    with open(lidarPngLPath, 'wb') as f:
        writer = png.Writer(width=lidar_png_L.shape[1], height=lidar_png_L.shape[0], bitdepth=16, greyscale=True)
        lidar_png_L2list = lidar_png_L.tolist()
        writer.write(f, lidar_png_L2list)

    #beginning of code for projecting onto right image plane
    imgPathR = imgPath + '/image_3/' + lidarPath.split('/')[-1].replace('bin', 'png')
    imgR = cv.imread(imgPathR)
    heightR = imgR.shape[0]
    widthR = imgR.shape[1]
    dist_R, pts2d_R = project_velo_points_in_img(pts3d_t, calibDict['Tr'], calibDict['P3'])
    pts2d_R, dist_R = filter_pts_to_image_window(pts2d_R.T, dist_R, widthR-1, heightR-1)
    f = round(calibDict['P3'][0,0])
    lidar_disp_R = 0.54*f/dist_R
    lidar_disp_R = np.round(256*lidar_disp_R)
    lidar_png_R = np.zeros((heightR,widthR)).astype(np.uint16)

    for i in range(len(lidar_disp_R)):
        if lidar_disp_R[i] > lidar_png_R[int(round(pts2d_R[i,1])), int(round(pts2d_R[i,0]))]:
            lidar_png_R[int(round(pts2d_R[i,1])), int(round(pts2d_R[i,0]))] = int(lidar_disp_R[i])

    lidarPngRPath = imgPathR.replace('data_odometry_color', 'data_odometry_projlidar')

    with open(lidarPngRPath, 'wb') as f:
        writer = png.Writer(width=lidar_png_R.shape[1], height=lidar_png_R.shape[0], bitdepth=16, greyscale=True)
        lidar_png_R2list = lidar_png_R.tolist()
        writer.write(f, lidar_png_R2list)

    #draw_pts_as_cmp_on_image_pyplot(imgR, pts2d_R[:,0:2], dist_R)








