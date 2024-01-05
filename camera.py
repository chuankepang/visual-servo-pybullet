"""Provides a utility Camera class for PyBullet."""
import numpy as np
import pybullet as pyb
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

class Camera:
    """A PyBullet camera."""

    def __init__(
            self,
            view_matrix,
            proj_matrix,
            near=0.1,
            far=1000.0,
            width=512,
            height=384,
    ):
        self.near = near
        self.far = far
        self.width = width
        self.height = height

        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix  # proj推出内参比较麻烦,只有内参无法推proj,必须有near
        self.intrinsics = self.proj_matrix2intrinsics(proj_matrix,width,height)
        
        
    @classmethod
    def from_pose_intrinsics(
            cls,
            pos,
            eul,
            intrinsics,
            near=0.1,
            far=1000.0,
            width=512,
            height=384,
    ):
        view_matrix = cls.pose2view_matrix(pos,eul)
        proj_matrix = cls.intrinsics2proj_matrix(intrinsics,width,height,near,far)
        return cls(
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            near=near,
            far=far,
            width=width,
            height=height,
        )
        
    @classmethod    
    def from_pose_fov(cls,
            pos,
            eul,
            fov=60,
            near=0.1,
            far=1000.0,
            width=512,
            height=384,
    ):
        """FOV用于主点在中心的，相机左右
        """
        view_matrix = cls.pose2view_matrix(pos,eul)
        proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height, nearVal=near, farVal=far
        )
        return cls(
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            near=near,
            far=far,
            width=width,
            height=height,
        )
        
    @classmethod
    def from_eye_fov(
            cls,
            target_position,
            camera_position,
            up_vector,
            fov=60.0,
            near=0.1,
            far=1000.0,            
            width=512,
            height=384,
    ):
        """Construct a new camera from target and camera positions.

        Parameters:
            target_position: The position of the camera's target point
            camera_position: The position of the camera in the world
            up_vector: Camera Up Vector
            near: Near value
            far: Far value
            fov: Field of view
            width: Width of the image
            height: Height of the image
        """
        view_matrix = pyb.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector,
        )
        proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height, nearVal=near, farVal=far
        )
        return cls(
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            near=near,
            far=far,
            width=width,
            height=height,
        )

    @classmethod
    def from_distance_rpy(
            cls,
            target_position,
            distance,
            roll=0,
            pitch=0,
            yaw=0,
            fov=60.0,
            near=0.1,
            far=1000.0,            
            width=512,
            height=384,
    ):
        """Construct a new camera from target position, distance, and roll,
        pitch, yaw angles.

        Parameters:
            target_position: The position of the camera's target point
            distance: Distance of camera from target.
            roll: Roll of the camera.
            pitch: Pitch of the camera.
            yaw: Yaw of the camera.
            near: Near value
            far: Far value
            fov: Field of view
            width: Width of the image
            height: Height of the image
        """
        view_matrix = pyb.computeViewMatrixFromYawPitchRoll(
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            cameraTargetPosition=target_position,
            upAxisIndex=2,
        )
        proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height, nearVal=near, farVal=far
        )
        return cls(
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            near=near,
            far=far,
            width=width,
            height=height,
        )
        
    
    @staticmethod    
    def proj_matrix2intrinsics(P, w, h):
        fx = P[0] * w / 2
        fy = P[5] * h / 2
        cx = w * (1-P[2])/2
        cy = h * (1-P[6])/2
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        return K
    
    @staticmethod    
    def intrinsics2proj_matrix(K, w, h, near, far):
        """
        cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
        and ROS to the projection matrix used in openGL and Pybullet.

        :param K:  OpenCV 3x3 camera intrinsic matrix
        :param w:  Image width
        :param h:  Image height
        :near:     The nearest objects to be included in the render
        :far:      The furthest objects to be included in the render
        :return:   4x4 projection matrix as used in openGL and pybullet
        """ 
        f_x = K[0,0]
        f_y = K[1,1]
        c_x = K[0,2]
        c_y = K[1,2]
        A = (near + far)/(near - far)
        B = 2 * near * far / (near - far)

        projection_matrix = [
                            [2/w * f_x,  0,          (w - 2*c_x)/w,  0],
                            [0,          2/h * f_y,  (2*c_y - h)/h,  0],
                            [0,          0,          A,              B],
                            [0,          0,          -1,             0]]
        #The transpose is needed for respecting the array structure of the OpenGL
        return np.array(projection_matrix).T.reshape(16).tolist()
   
    
    @staticmethod
    def fov2intrinsics(fov, width, height):
        fov = fov / 180 * np.pi
        focal_length = (height / 2) / np.tan(fov / 2)
        intrinsics = np.array([[focal_length, 0.0, width / 2],
                              [0.0, focal_length, height / 2],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
        return intrinsics

    @staticmethod
    def compute_pose(eye_position, target_position, up_vector):
        eye_position = np.asarray(eye_position)
        target_position = np.asarray(target_position)
        up_vector = np.asarray(up_vector)
        y = -up_vector / np.linalg.norm(up_vector)
        z = target_position - eye_position
        z = z / np.linalg.norm(z)
        x = np.cross(y, z)
        rotation_matrix = np.array([x, y, z]).T
        # camera to world
        pose = np.eye(4, dtype=np.float32)
        pose[0:3, 0:3] = rotation_matrix
        pose[0:3, 3] = eye_position
        return pose

    def set_eye_up(self, eye_position, target_position, up_vector):
        eye_position = np.asarray(eye_position)
        target_position = np.asarray(target_position)
        up_vector = np.asarray(up_vector)
        self.view_matrix = pyb.computeViewMatrix(cameraEyePosition=eye_position,
                                                 cameraTargetPosition=target_position,
                                                 cameraUpVector=up_vector)

    def set_pose(self, pos, eul, index='zyx'):
        view_matrix = self.pose2view_matrix(pos, eul, index)
        self.view_matrix = view_matrix.flatten(order='F')
    
    @staticmethod
    def pose2view_matrix(pos, eul, index='zyx'):
        pos = np.asarray(pos)
        eul = np.asarray(eul)
        # print(eul)
        # 相机的指向，z朝前
        rotation_matrix = R.from_euler(index, eul, degrees=False)
        # R.from_euler('zyx', [90, 90, 180], degrees=True).as_matrix()
        # R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix()
        # 变换到和view一样的方位，z朝后，绕x转180
        r = R.from_euler('x', np.pi, degrees=False)
        rot_matrix = np.eye(4)
        rot_matrix[0:3, 0:3] = (rotation_matrix * r).as_matrix().T

        trans_matrix = np.eye(4)
        trans_matrix[0, 3] = -pos[0]
        trans_matrix[1, 3] = -pos[1]
        trans_matrix[2, 3] = -pos[2]

        view_matrix = np.dot(rot_matrix, trans_matrix)  # 世界 -》摄像机。 先平移，再旋转
        # print(view_matrix.flatten(order='F'))
        return view_matrix
    
    @staticmethod
    def poseq2view_matrix(t,q):
        """
        cvPose2BulletView gets orientation and position as used 
        in ROS-TF and opencv and coverts it to the view matrix used 
        in openGL and pyBullet.
        
        :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
        :param t: ROS postion expressed as [tx, ty, tz]
        :return:  4x4 view matrix as used in pybullet and openGL
        
        """
        q = Quaternion([q[3], q[0], q[1], q[2]])
        R = q.rotation_matrix

        T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
                                np.array([0, 0, 0, 1])])
        # Convert opencv convention to python convention
        # By a 180 degrees rotation along X
        Tc = np.array([[1,   0,    0,  0],
                    [0,  -1,    0,  0],
                    [0,   0,   -1,  0],
                    [0,   0,    0,  1]]).reshape(4,4)
        
        # pybullet pse is the inverse of the pose from the ROS-TF
        T=Tc@np.linalg.inv(T)
        # The transpose is needed for respecting the array structure of the OpenGL
        viewMatrix = T.T.reshape(16)
        return viewMatrix
    
    @staticmethod
    def view_matrix2pose(view_matrix):
        pass
    
    @staticmethod
    def eye_target_up2pose(eye_position, target_position, up_vector):
        # 获取摄像机坐标系的3个基向量
        eye_position = np.asarray(eye_position)
        target_position = np.asarray(target_position)
        up_vector = np.asarray(up_vector)
        z = eye_position - target_position
        z = z / np.linalg.norm(z)
        x = np.cross(up_vector, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        rotation_matrix = np.array([x, y, z])
        rot_matrix = np.eye(4)
        rot_matrix[0:3, 0:3] = rotation_matrix

        # 计算摄像机相对于世界坐标系的位移
        trans_matrix = np.eye(4)
        trans_matrix[0, 3] = -eye_position[0]
        trans_matrix[1, 3] = -eye_position[1]
        trans_matrix[2, 3] = -eye_position[2]
        return np.dot(rot_matrix, trans_matrix)  # 世界 -》摄像机。 先平移，再旋转

    def get_frame(self):
        """Get a frame from the camera.

        Returns:
            rgba: The RGBA colour data
            depth: Depth buffer
            seg: Segmentation mask
        """
        _, _, rgba, depth, seg = pyb.getCameraImage(
            width=self.width,
            height=self.height,
            shadow=1,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pyb.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgba[:, :, :3], depth, seg

    def save_frame(self, filename, rgba=None):
        """Save a frame to a file.

        Parameters:
            filename: The name of the image file
            rgba: Optional, RGBA data provided by `Camera.get_frame()`. If not
                provided, `self.get_frame()` is called to retrieve this data.
        """
        if rgba is None:
            rgba, _, _ = self.get_frame()
        img = Image.fromarray(
            np.reshape(rgba, (self.height, self.width, 4)), "RGBA"
        )
        img.save(filename)

    def linearize_depth(self, depth=None):
        """Convert depth map to actual distance from camera plane.

        See <https://stackoverflow.com/a/6657284>.

        Parameters:
            depth: Optional, depth buffer provided by `Camera.get_frame()`. If
                not provided, `self.get_frame()` is called to retrieve this
                data.

        Returns: linearized depth buffer: actual depth values from the camera
            plane
        """
        if depth is None:
            _, depth, _ = self.get_frame()
        depth_ndc = 2 * depth - 1  # normalized device coordinates
        depth_linear = (
                2.0
                * self.near
                * self.far
                / (self.far + self.near - depth_ndc * (self.far - self.near))
        )
        return depth_linear

    def set_camera_pose(self, position, target):
        """Change position and target of the camera."""
        self.position = position
        self.target = target
        self.view_matrix = pyb.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
        )

    def get_point_cloud(self, depth=None):
        """Convert depth buffer to 3D point cloud in world coordinates.

        See <https://stackoverflow.com/a/62247245> for the main source of this
        code.

        Parameters:
            depth: Optional, depth buffer provided by `Camera.get_frame()`. If
                not provided, `self.get_frame()` is called to retrieve this
                data.

        Returns: A (width, height, 3)-dimensional array of points seen by the
            camera.
        """
        if depth is None:
            _, depth, _ = self.get_frame()

        # view matrix maps world coordinates to camera coordinates (extrinsics)
        V = np.array(self.view_matrix).reshape((4, 4), order="F")

        # camera projection matrix: map camera coordinates to clip coordinates
        # (intrinsics)
        P = np.array(self.proj_matrix).reshape((4, 4), order="F")

        PV_inv = np.linalg.inv(P @ V)

        # depth is stored (height * width) (i.e., transpose of what one might
        # expect on the numpy side)
        points = np.zeros((self.width, self.height, 3))
        for h in range(self.height):
            for w in range(self.width):
                # convert to normalized device coordinates
                # notice that the y-transform is negative---we actually have a
                # left-handed coordinate frame here (x = right, y = down, z =
                # out of the screen)
                x = (2 * w - self.width) / self.width
                y = -(2 * h - self.height) / self.height

                # depth buffer is already in range [0, 1]
                z = 2 * depth[h, w] - 1

                # back out to world coordinates by applying inverted projection
                # and view matrices
                r_ndc = np.array([x, y, z, 1])
                r_world_unnormalized = PV_inv @ r_ndc

                # normalize homogenous coordinates to get rid of perspective
                # divide
                points[w, h, :] = (
                        r_world_unnormalized[:3] / r_world_unnormalized[3]
                )
        return points


class VideoRecorder:
    """Recorder for a video of a PyBullet simulation."""

    def __init__(self, filename, camera, fps, codec="mp4v"):
        """Initialize the VideoRecorder.

        Parameters:
            filename: The file to write the video to.
            camera: Camera object to use for rendering frames.
            fps: Frames per second. Each `fps` frames will be played over one
                second of video.
        """
        self.camera = camera
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(filename),
            fourcc,
            fps,
            (camera.width, camera.height),
        )

    def capture_frame(self, rgba=None):
        """Capture a frame and write it to the video.

        Parameters:
            rgba: If provided, write this data to the video (this can be used
            to avoid multiple renderings with the camera). Otherwise, get the
            frame data from the camera.
        """
        if rgba is None:
            rgba, _, _ = self.camera.get_frame()

        # OpenCV uses BGR instead of RGB
        bgr = rgba[..., [2, 1, 0]]
        self.writer.write(bgr)
