'''
This script is a work-in-progress for exploring fast naive stereo and lidar
fusion strategies. It substitutes all the stereo disparities with lidar
disparities. For pixels that don't have lidar disparities, the averaged stereo
disparities of the surrounding 4 pixels are used.
'''

import numpy as np
import cv2 as cv
import glob
from scipy import io as sio


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
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
       numpy array. Returns the 2D projection of the points that
       are in front of the camera only an the corresponding 3D points.'''

    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts3d_cam = pts3d_cam[:,idx]
    dist = np.sqrt(pts3d_cam[0,:]**2+pts3d_cam[1,:]**2+pts3d_cam[2,:]**2)
    pts2d_cam = Prect.dot(pts3d_cam)

    return dist, pts3d[:, idx], pts2d_cam/pts2d_cam[2,:]

'''
This function filters the points and only returns the ones that are in the
image window.
'''
def filter_pts_to_image_window(points, dist, width, height):
    mask = (points[:,0] <= width) & (points[:,0] >= 0)
    points = points[mask, :]
    dist = dist[mask]
    mask = (points[:,1] <= height) & (points[:,1] >= 0)
    points = points[mask, :]
    dist = dist[mask]
    return points, dist

'''
This function computes the stereo disparity using OpenCV's StereoSGBM. All the
parameters have been tuned for the KITTI dataset.
'''
def compute_stereo_disparity(imgL, imgR):
    stereo = cv.StereoSGBM_create(minDisparity = 1,
            numDisparities = 64,
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
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    #disp = stereo.compute(imgL, imgR)

    return disp

def compute_stereo_disparity_fast(imgL, imgR):
    stereo = cv.StereoBM_create(numDisparities=128, blockSize=19)
    disp = stereo.compute(imgL, imgR)
    return disp

l_list = glob.glob("/Volumes/EXT_DRIVE/lidarstereo_data/lidarstereonet_test/*_l.png")
r_list = glob.glob("/Volumes/EXT_DRIVE/lidarstereo_data/lidarstereonet_test/*_r.png")
calibList = glob.glob("/Volumes/EXT_DRIVE/lidarstereo_data/lidarstereonet_test/*_calib.mat")
lidarList = glob.glob("/Volumes/EXT_DRIVE/lidarstereo_data/lidarstereonet_test/*.bin")
l_list.sort()
r_list.sort()
calibList.sort()
lidarList.sort()

for idx, l in enumerate(l_list):
    print('{} / {}'.format(idx+1, len(l_list)))
    imgL = cv.imread(l_list[idx])
    imgR = cv.imread(r_list[idx])
    disp = compute_stereo_disparity(imgL, imgR)
    # cv.imshow('test', disp)
    # cv.waitKey(1)


    lidarPath = lidarList[idx]
    velo = np.fromfile(lidarPath, dtype=np.float32).reshape(-1,4)
    calibPath = calibList[idx]
    calibMat = sio.loadmat(calibPath)
    height = imgL.shape[0]
    width = imgL.shape[1]
    pts3d_t = prepare_velo_points(velo)
    dist_L, _, pts2d_L = project_velo_points_in_img(pts3d_t, calibMat['TR_velo_to_cam'], calibMat['R'], calibMat['P'])
    pts2d_L, dist_L = filter_pts_to_image_window(pts2d_L.T, dist_L, width-1, height-1)
    f = round(calibMat['P'][0,0])
    lidar_disp_L = 0.54*f/dist_L
    #lidar_disp_L = np.round(lidar_disp_L)
    lidar_png_L = np.zeros((height,width)).astype(np.float32)

    for i in range(len(lidar_disp_L)):
        if lidar_disp_L[i] > lidar_png_L[int(round(pts2d_L[i,1])), int(round(pts2d_L[i,0]))]:
            lidar_png_L[int(round(pts2d_L[i,1])), int(round(pts2d_L[i,0]))] = int(lidar_disp_L[i])

    mask = lidar_png_L > 1e-5

    # print(disp[317,910])
    # #print(np.argmax(lidar_png_L))
    # #print(lidar_png_L.shape)
    # print(lidar_png_L[317,910])
    disp[mask] = lidar_png_L[mask]

    mask = disp < 1e-5

    #utlize numpy vectorized operations for better performance
    disp[mask] = (disp[np.roll(mask, 1, axis=0)]+disp[np.roll(mask, -1, axis=0)] + disp[np.roll(mask, 1, axis=1)] + disp[np.roll(mask,-1, axis=1)])/4

    #cv.imshow('test', disp.astype(np.uint16)*256)
    # cv.imshow('test2', lidar_png_L.astype(np.uint16)*256)
    #cv.imshow('test1', mask.astype(np.uint8)*56)
    cv.imshow('test', disp.astype(np.uint16)*256)
    cv.waitKey(0)

    #cv.imshow('test', lidar_png_L*256)
    #cv.waitKey(0)

    # lidarPngLPath = imgPathL.replace('data_odometry_color', 'data_odometry_projlidar')

    # with open(lidarPngLPath, 'wb') as f:
    #     writer = png.Writer(width=lidar_png_L.shape[1], height=lidar_png_L.shape[0], bitdepth=16, greyscale=True)
    #     lidar_png_L2list = lidar_png_L.tolist()
    #     writer.write(f, lidar_png_L2list)