import os

from tqdm import tqdm
import cv2
import numpy as np



# 图像去畸变的类
class CUndistortImage:
    def __init__(self, intrinsic, distortion_coeff):
        self.intrinsic = intrinsic
        self.distortion_coeff = distortion_coeff
        self.map1 = None
        self.map2 = None
        self.w = 0
        self.h = 0

    def remap(self, image):
        h, w = image.shape[:2]
        if w != self.w or h != self.h:
            self.w = w
            self.h = h
            self.map1, self.map2 = cv2.initUndistortRectifyMap(self.intrinsic, self.distortion_coeff, None, None,
                                                               (self.w, self.h), cv2.CV_32FC1)

        img = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
        return img


# 实现了相机的标定和去畸变操作
def calibration():
    folder = r"E:\sight\20230911_RGBD_camera\Ofilm_camera_tool_v1\Release\SaveData\2023_11_27_11_01_14\IR"
    save_folder = r"E:\sight\20230911_RGBD_camera\Ofilm_camera_tool_v1\Release\SaveData\2023_11_27_11_01_14\dst"
    save_remap_folder = r"E:\sight\20230911_RGBD_camera\Ofilm_camera_tool_v1\Release\SaveData\2023_11_27_11_01_14\remap"

    if os.path.exists(save_folder) is False:
        os.mkdir(save_folder)

    if os.path.exists(save_remap_folder) is False:
        os.mkdir(save_remap_folder)

    temp = os.listdir(folder)
    images_path = [p for p in temp if os.path.isfile(os.path.join(folder, p))]

    CHECKERBOARD = (10, 7)
    square_size = 30
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001

    # 世界坐标中的3D角点，z恒为0
    objpoints = []
    # 像素坐标中的2D点
    imgpoints = []

    # 利用棋盘定义世界坐标系中的角点
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    print("1. -->>>>> find cornors:")
    gray = None
    for i, element in enumerate(tqdm(images_path)):
        fname = images_path[i]
        name = fname.split("/")[-1]
        img = cv2.imread(os.path.join(folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 查找棋盘角点
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        """
        使用cornerSubPix优化探测到的角点
        """
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # 显示角点
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 1920, 1280)
            cv2.imshow('findCorners', img)
            cv2.waitKey(10)
            cv2.destroyAllWindows()
            new_img = img.copy()
            cv2.imwrite(os.path.join(save_folder, name), new_img)

    print("2. -->>>>> calibrateCamera:")
    # 标定
    ret, intrinsic, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                         None)
    print("重投影误差:\n")
    print(ret)
    print("内参 : \n")
    print(intrinsic)
    print("畸变 : \n")
    print(distortion_coeff)
    np.save("OMS_1920_1280_IR_intrinsic.npy", intrinsic)
    np.save("OMS_1920_1280_distortion_coeff.npy", distortion_coeff)

    map1, map2 = cv2.initUndistortRectifyMap(intrinsic, distortion_coeff, None, None,
                                             (gray.shape[1], gray.shape[0]), cv2.CV_32FC1)

    print("3. -->>>>> save remap images:")
    for i, element in enumerate(tqdm(images_path)):
        fname = images_path[i]
        name = fname.split("/")[-1]
        img = cv2.imread(os.path.join(folder, fname))
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_remap_folder, name), img)

    print("4. -->>>>> Done!")

    print("旋转向量 : \n")
    print(rvecs)
    print("平移向量 : \n")
    print(tvecs)


# 加载相机内参数和畸变系数，并对输入图像进行去畸变操作，将去畸变后的图像保存到指定文件夹中
def undistort_image():
    intrinsic=np.load("OMS_1920_1280_IR_intrinsic.npy")
    distortion_coeff=np.load("OMS_1920_1280_distortion_coeff.npy")
    undisort=CUndistortImage(intrinsic,distortion_coeff)
    print(intrinsic)
    print(distortion_coeff)

    folder=r"E:\tbx\QT_sutdy\CamUSB\Release\move_IR_IR"
    save_folder=r"E:\tbx\QT_sutdy\CamUSB\Release\move_IR_IR\deal"
    if os.path.exists(save_folder) is False:
        os.mkdir(save_folder)

    print(" -->>>>> save remap images:")
    temp=os.listdir(folder)
    images_path = [p for p in temp if os.path.isfile(os.path.join(folder, p))]
    for i,ele in enumerate(tqdm(images_path)):
        img=cv2.imread(os.path.join(folder,ele))
        img=undisort.remap(img)
        cv2.imwrite(os.path.join(save_folder, ele), img)


if __name__ == '__main__':
    calibration()

    # undistort_image()


