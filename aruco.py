import numpy as np
from PIL import Image
import logging
import pybullet as pb
import pybullet_data
# from pybullet_object_models import ycb_objects
import os
from camera import Camera
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import time

log = logging.getLogger(__name__)

def exp_map_3d(w):
    """
    w: 3D vector (angular velocity)
    Returns the exponential map of the 3D vector to a rotation matrix.
    """
    skew_symmetric = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])  # 创建一个skew对称矩阵
    return expm(skew_symmetric)

camera_width  = 512
camera_height = 384
# camera_matrix = np.array([[207.84608841,0.,   160.],[0.,  207.84608841, 120.],[0,  0,   1.]])
camera_matrix = np.array([[332.55374908,0.,   256.],[0.,  332.55374908, 192.],[0,  0,   1.]])
camera_dist = np.array([0., 0., 0., 0., 0.] )
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        # # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corner = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (255, 0, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 2)
            cv2.line(image, bottomLeft, topLeft, (255, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            # cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            # cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            # print("[Inference] ArUco marker ID: {}".format(markerID))    
                  
        # 坐标系  
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, camera_matrix, camera_dist)
        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(image, camera_matrix, camera_dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            # cv2.aruco.drawDetectedMarkers(image, corners)            
    else:
        cv2.putText(image, "No Ids", (0,64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    return image # show the output image

def draw_des(image, s_des):
    (topLeft, topRight, bottomRight, bottomLeft) = s_des
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))

    cv2.line(image, topLeft, topRight, (255, 0, 0), 2)
    cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
    cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 2)
    cv2.line(image, bottomLeft, topLeft, (255, 255, 0), 2)
    return image
    
def norm_point(p):# 4*2
    p_tmp = np.empty_like(p)
    p_tmp[:,0] = (p[:,0] - camera_matrix[0,2]) / camera_matrix[0,0]
    p_tmp[:,1] = (p[:,1] - camera_matrix[1,2]) / camera_matrix[1,1]
    # p[:,1] -= self.camera_matrix[1,2] # 不能这么写
    return p_tmp
        
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
# parameters =  cv2.aruco.DetectorParameters_create()

def detect_aruco(img):
    # 4.7.x修改了API
    # dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    # parameters =  cv.aruco.DetectorParameters()
    # detector = cv.aruco.ArucoDetector(dictionary, parameters)
    # frame = cv.imread(...)
    # markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
    # cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    # print(f'markerCorners: {markerCorners}')
    img = aruco_display(markerCorners, markerIds, rejectedCandidates, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))   
    s = None
    if markerIds is not None:   
        for (markerCorner, markerID) in zip(markerCorners, markerIds):
            if markerID == 0:
                s = markerCorners[0][0]  # list np 1 4 2
    else:
        s = None
    return s, img
    
        
def load_environment(client):
    pb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), 
        physicsClientId=client
    )
    pb.setGravity(0, 0, -9.81)
    pb.setRealTimeSimulation(0)
    ground_id = pb.loadURDF("plane.urdf", 
                           (0, 0, 0), 
                           useFixedBase=True, 
                           physicsClientId=client)
    
    cube_id = pb.loadURDF("cube_aruco.urdf", 
                          (0.0, 0.0, 0.1),
                          (0.0, 0.0, 0.0, 1.0),
                        #   useFixedBase=True, 
                        #   flags=pb.URDF_USE_INERTIA_FROM_FILE
                        )
    bodies = {
        "ground": ground_id,
        "cube": cube_id,
    }
    
    camera = Camera.from_eye_fov(
        camera_position=(0.3, 0.3, 0.3),
        target_position=(0.6, 0.3, 0.3),
        up_vector=(0, 0, 1),
        near=0.1,
        far=5,                
        width=camera_width,
        height=camera_height,
    )
    camera.set_pose((0.0, 0.0, 0.1), (90.0, 90.0, 180.0))

    print(camera.proj_matrix)
    # print(camera.view_matrix)
    print(camera.intrinsics)
    return bodies, camera


def get_next_velocity(s, s_des):
    if s is not None:
        s_norm = norm_point(s)
        s_des_norm = norm_point(s_des)
        
        z = 0.2
        zi = 1/z
        L = []
        for p in s_norm:
            x = p[0]
            y = p[1]
            L.append(np.array([
                [-zi, 0, x * zi, x * y, -(1 + x ** 2), y],
                [0, -zi, y * zi, 1 + y ** 2, -x * y, -x],
            ]))
        L = np.array(L).reshape(-1,6)
        # print(f'L shape: {L.shape}')
        # print(f'L: {L}')
        
        assert L.shape == (8, 6)
        
        error = s_norm.flatten() - s_des_norm.flatten() # 一行一行展开
        # print(f'L: {L}')
        # print(f's: {s_norm.flatten()}')
        # print(f'sd: {s_des_norm.flatten()}')
        # print(f'error: {error}')
        # print(f'np.linalg.pinv(L): {np.linalg.pinv(L)}')

        V = -1 * np.linalg.pinv(L) @ error
        
    else:
        V = np.zeros((6))
        
    return V

def main():
    client = pb.connect(pb.GUI)
    # control_dt = 1. / 60.
    # pb.setTimestep = control_dt
    bodies, camera = load_environment(client)
    # K = camera.intrinsics
    # fx = K[0,0]
    # fy = K[1,1]
    # cx = K[0,2]
    # cy = K[1,2]

    axis = ['x', 'y', 'z']
    # start_value = np.array([0.0, 0.1, 0.1])
    start_value = np.array([-0.15, 0.05, 0.4])
    p = start_value
    pos_params_ids = [pb.addUserDebugParameter(
        paramName=axis[i],
        rangeMin=-2,
        rangeMax=2,
        startValue=start_value[i]
    ) for i in range(3)]
    # start_value = np.array([np.pi/2, np.pi/2, np.pi])
    start_value = np.array([-np.pi/2, 0.4, np.pi])
    r = start_value
    ang_params_ids = [pb.addUserDebugParameter(
        paramName="ang" + axis[i],
        rangeMin=-np.pi,
        rangeMax=np.pi,
        startValue=start_value[i]
    ) for i in range(3)]
    # linvel_params_ids = [p.addUserDebugParameter(
    #     paramName="v" + axis[i],
    #     rangeMin=-1,
    #     rangeMax=1,
    #     startValue=0
    # ) for i in range(3)]
    # angvel_params_ids = [p.addUserDebugParameter(
    #     paramName="w" + axis[i],
    #     rangeMin=-1,
    #     rangeMax=1,
    #     startValue=0
    # ) for i in range(3)]
    
    # r = np.array([np.pi/2, np.pi/2, np.pi])
    
    # p = np.array([0.0, 0.0, 0.0])
    # r = np.array([0, 0, 0])
    camera.set_pose(p, r)
    
    size = 50.
    w = camera_width
    h = camera_height
    s_des = np.array([[w/2-size,h/2-size],[w/2+size,h/2-size],[w/2+size,h/2+size],[w/2-size,h/2+size]])
    
    idx = 0
    while 1:
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        # pos = [pb.readUserDebugParameter(param_id) for param_id in pos_params_ids]
        # ang = [pb.readUserDebugParameter(param_id) for param_id in ang_params_ids]
        # camera.set_pose(pos, ang)
        
        rgb, depth, seg = camera.get_frame()
        s, img_aruco = detect_aruco(rgb)
        img_aruco=draw_des(img_aruco,s_des)
        # print(rgb.shape)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # disp = np.concatenate((rgb, ctrl.des_img, rgb-ctrl.des_img), axis=1)
        
        cv2.imshow("Demo", img_aruco)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("save.png", bgr)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
            
        
        V = get_next_velocity(s, s_des)
        # print(f"V:{V}")
        # V = np.zeros((6))
        # print(f's: {s}')
        # print(f'sd: {s_des}')
        
        r_now = R.from_euler('zyx', r, degrees=False)
        # rotm = R.from_euler('zyx', r, degrees=False).as_matrix()
        # print(f"r_now: {r_now.as_matrix()}")
        v = r_now.as_matrix() @ V[:3]
        # print(f"v:{v}")
        w = 1 * r_now.as_matrix() @ V[3:]
        # print(f"w:{w}")

        dt = 1/240
        # print(f"p:{p}")
        p += (np.array(v) * dt)
        # print(f"p:{p}")
        # r_new = r_now * R.from_matrix(exp_map_3d(np.array(w) * dt))
        r_new = R.from_matrix(exp_map_3d(np.array(w) * dt)) * r_now
        # rot = r_now.as_matrix() * exp_map_3d(np.array(w) * dt)   # TODO:雅克比矩阵还没有乘
        # print(f"r_new: {r_new.as_matrix()}")
        # r = R.from_matrix(rot).as_euler('zyx', degrees=False)
        # print(f"r eul:{r}")
        # print(f"r_now eul:{r_now.as_euler('zyx', degrees=False)}")
        # print(f"r_new eul:{r_new.as_euler('zyx', degrees=False)}")
        r = r_new.as_euler('zyx', degrees=False)
        camera.set_pose(p, r)
        idx += 1
        # log.debug(f"velocity: {V}\nvel: {v}\npose: {p},{r}")
        # log.info(f"mse: {ctrl.mse(rgb,ctrl.des_img)}")
        
        pb.stepSimulation()
        time.sleep(1./240.)
    
if __name__ == '__main__':
    main()