from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from butterworth import Butter
from header.utils import preprocessing, correct_image, detect_object
from header.classify import classify, display
from header.histogram import make_masks, calc_histo, draw_ellipse

import sys
import cv2
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('source/pyqt/demo.ui', self)
        self.setWindowIcon(QtGui.QIcon("source/pyqt/python-icon.png"))

        self.image = None

        # 파일
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionPrint.triggered.connect(self.createPrintDialog)
        self.actionQuit.triggered.connect(self.QuestionMessage)

        # 각도변환
        self.actionRotation.triggered.connect(self.rotation)
        self.actionAffine.triggered.connect(self.shearing)

        # 보정
        self.actioAnhXam.triggered.connect(self.anh_Xam)
        self.actionNegative.triggered.connect(self.anh_Negative)
        self.actionLog.triggered.connect(self.Log)
        self.actionHistogram.triggered.connect(self.histogram_Equalization)
        self.actionGamma.triggered.connect(self.gamma)

        # 스무딩
        self.actionBlur.triggered.connect(self.blur)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBilateral_Filter.triggered.connect(self.bilateral_filter)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)

        # 필터
        self.actionMedian_threshold_2.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_4.triggered.connect(self.directional_filtering3)
        self.action_Butterworth.triggered.connect(self.butter_filter)
        self.action_Notch_filter.triggered.connect(self.notch_filter)

        # 특수효과
        self.actionCartoon.triggered.connect(self.cartoon)
        self.actionEmbossing.triggered.connect(self.embossing)
        self.actionSketchContrast.triggered.connect(self.sketchContrast)
        self.actionSketchColor.triggered.connect(self.sketchColor)
        self.actionOil.triggered.connect(self.oil)

        # 성별분류
        self.actionSex.triggered.connect(self.detectSex)

        #보기
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)

        # 화면 우측 커스텀 셋업
        self.dial.valueChanged.connect(self.rotation2)
        self.horizontalSlider.valueChanged.connect(self.Gamma_)
        self.gaussian_QSlider.valueChanged.connect(self.gaussian_filter2)
        self.erosion.valueChanged.connect(self.erode)
        self.Qlog.valueChanged.connect(self.Log)
        self.size_Img.valueChanged.connect(self.SIZE)
        self.canny.stateChanged.connect(self.Canny)
        self.canny_min.valueChanged.connect(self.Canny)
        self.canny_max.valueChanged.connect(self.Canny)
        self.pushButton.clicked.connect(self.reset)

        # ## 안쓰는 거##
        # # 각도변환
        # self.actionTranslation.triggered.connect(self.translation)
        # # made by
        # self.actionQt.triggered.connect(self.AboutMessage)
        # self.actionAuthor.triggered.connect(self.AboutMessage2)
        # # Image Restoration 1
        # self.actionAdaptive_Wiener_Filtering.triggered.connect(self.weiner_filter)
        # self.actionMedian_Filtering.triggered.connect(self.median_filtering)
        # self.actionAdaptive_Median_Filtering.triggered.connect(self.adaptive_median_filtering)
        # # Image Restoration 2
        # self.actionInverse_Filter.triggered.connect(self.inv_filter)
        # # Simple Edge Detection
        # self.actionSHT.triggered.connect(self.simple_edge_detection)
        # # Chapter 3
        # self.actionGaussian.triggered.connect(self.gaussian_noise)
        # self.actionRayleigh.triggered.connect(self.rayleigh_noise)
        # self.actionErlang.triggered.connect(self.erlang_noise)
        # self.actionUniform.triggered.connect(self.uniform_noise)
        # self.actionImpluse.triggered.connect(self.impulse_noise)
        # self.actionHistogram_PDF.triggered.connect(self.hist)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] is the number of pixels in the dimension Y's.
        # image.shape[1] is the number of pixels in the X direction.
        # image.shape[2] store the number of channels displayed per pixel
        img = img.rgbSwapped() # effectively convert an RGB image into a BGR image.
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# adjust the appearance of the figure on the compass
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'source/pyqt', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'source/pyqt', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image) # 저장
        else:    
            print("Save Error")

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imgLabel2.print_(printer)

    def big_Img(self):
        self.image = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def SIZE(self , c):
        self.image = self.tmp
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def reset(self):
        self.image = self.tmp
        self.displayImage(2)

    def AboutMessage(self):
        QMessageBox.about(self, "About Qt - Qt Designer",
            "Qt is a multiplatform C + + GUI toolkit created and maintained byTrolltech.It provides application developers with all the functionality needed to build applications with state-of-the-art graphical user interfaces.\n"
            "Qt is fully object-oriented, easily extensible, and allows true component programming.Read the Whitepaper for a comprehensive technical overview.\n\n"

            "Since its commercial introduction in early 1996, Qt has formed the basis of many thousands of successful applications worldwide.Qt is also the basis of the popular KDE Linux desktop environment, a standard component of all major Linux distributions.See our Customer Success Stories for some examples of commercial Qt development.\n\n"

            "Qt is supported on the following platforms:\n\n"

                "\tMS / Windows - - 95, 98, NT\n"
                "\t4.0, ME, 2000, and XP\n"
                "\tUnix / X11 - - Linux, Sun\n"
                "\tSolaris, HP - UX, Compaq Tru64 UNIX, IBM AIX, SGI IRIX and a wide range of others\n"
                "\tMacintosh - - Mac OS X\n"
                "\tEmbedded - - Linux platforms with framebuffer support.\n\n"
                          
            "Qt is released in different editions:\n\n"
            
                "\tThe Qt Enterprise Edition and the Qt Professional Edition provide for commercial software development.They permit traditional commercial software distribution and include free upgrades and Technical Support.For the latest prices, see the Trolltech web site, Pricing and Availability page, or contact sales @ trolltech.com.The Enterprise Edition offers additional modules compared to the Professional Edition.\n\n"
                "\tThe Qt Open Source Edition is available for Unix / X11, Macintosh and Embedded Linux.The Open Source Edition is for the development of Free and Open Source software only.It is provided free of charge under the terms of both the Q Public License and the GNU General Public License."
        )
    def AboutMessage2(self):
        QMessageBox.about(self, "About Author", "Instructor: Ngo Country Vietnameseto put in one's oar" 
                                                "The man who made it." 
                                                    "\tPhan Hoang Viet - 42.01.104.189"
                          )

    def QuestionMessage(self):
        message = QMessageBox.question(self, "Exit", "정말로 나가시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

################################ 각도 변환 ##############################################################################
    def rotation(self):
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1) #change the direction of an image
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def rotation2(self, angle):
        self.image = self.tmp
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def shearing(self):
        self.image = self.tmp
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))

        self.displayImage(2)

    def translation(self):
        self.image = self.tmp
        num_rows, num_cols = self.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
        self.image = img_translation
        self.displayImage(2)

    def erode(self , iter):
        self.image = self.tmp
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
        self.displayImage(2)

    def Canny(self):
        self.image = self.tmp
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)

################################ Chapter 3 ##############################################################################
    def anh_Xam(self):
        self.image = self.tmp
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)

    def anh_Xam2(self):
        self.image = self.tmp
        if self.gray.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.displayImage(2)

    def anh_Negative(self):
        self.image = self.tmp
        self.image = ~self.image
        self.displayImage(2)

    def histogram_Equalization(self):
        self.image = self.tmp
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        self.displayImage(2)

    def Log(self):
        self.image = self.tmp
        
        # float32로 변환하여 정밀도 유지
        float_img = self.image.astype(np.float32)
        
        # 0값 처리를 위해 1을 더함
        float_img = float_img + 1.0
        
        # 각 채널별로 로그 변환 적용
        for i in range(3):
            # 로그 변환 수행
            float_img[:,:,i] = np.log(float_img[:,:,i])
            
            # 0-255 범위로 정규화
            channel_min = float_img[:,:,i].min()
            channel_max = float_img[:,:,i].max()
            float_img[:,:,i] = ((float_img[:,:,i] - channel_min) * 255 / 
                            (channel_max - channel_min))
        
        # uint8로 변환
        self.image = float_img.astype(np.uint8)
        self.displayImage(2)


    def Gamma_(self, gamma):
        self.image = self.tmp
        gamma = gamma*0.1
        invGamma = 1.0 /gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

    def gamma(self):
        self.image = self.tmp
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

####################################### 각도 변환 ################################################################
    def gaussian_noise(self):
        self.image = self.tmp
        row, col, ch = self.image.shape
        mean = 0
        var = 0.1
        sigma = var * 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        self.image = self.image + gauss
        self.displayImage(2)
    def erlang_noise(self):
        self.image = self.tmp
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        cv2.randu(table, 1, 1)
        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)
    def rayleigh_noise(self):
        self.image = self.tmp
        r = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        self.image = cv2.randu(r, 1, 1)
        self.displayImage(2)
    def uniform_noise(self):
        self.image = self.tmp
        uniform_noise = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        cv2.randu(uniform_noise, 0, 255)
        self.image = (uniform_noise * 0.5).astype(np.uint8)
        self.displayImage(2)
    def impulse_noise(self):
        self.image = self.tmp
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.image.shape]
        out[coords] = 0
        self.image = out
        self.displayImage(2)

    def hist(self):
        self.image = self.tmp
        histg = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        self.image = histg
        plt.plot(self.image)
        plt.show()
        self.displayImage(2)

####################################Image Restoration 1#################################################################
    def median_filtering(self):
        self.image = self.tmp
        self.image = cv2.medianBlur(self.image, 5)
        self.displayImage(2)

    def adaptive_median_filtering(self):
        self.image = self.tmp
        temp = []
        filter_size = 5
        indexer = filter_size // 2
        for i in range(len(self.image)):

            for j in range(len(self.image[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(self.image) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(self.image[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(self.image[i + z - indexer][j + k - indexer])

                temp.sort()
                self.image[i][j] = temp[len(temp) // 2]
                temp = []
        self.displayImage(2)

    def weiner_filter(self):
        self.image = self.tmp
        M = 256  # length of Wiener filter
        Om0 = 0.1 * np.pi  # frequency of original signal
        N0 = 0.1  # PSD of additive white noise

        # generate original signal
        s = np.cos(Om0 * np.ndarray(self.image))
        # generate observed signal
        g = 1 / 20 * np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        n = np.random.normal(size=self.image, scale=np.sqrt(N0))
        x = np.convolve(s, g, mode='same') + n
        # estimate (cross) PSDs using Welch technique
        f, Pxx = sig.csd(x, x, nperseg=M)
        f, Psx = sig.csd(s, x, nperseg=M)
        # compute Wiener filter
        H = Psx / Pxx
        H = H * np.exp(-1j * 2 * np.pi / len(H) * np.arange(len(H)) * (len(H) // 2))  # shift for causal filter
        h = np.fft.irfft(H)
        # apply Wiener filter to observation
        self.image = np.convolve(x, h, mode='same')
        self.displayImage(2)

####################################Image Restoration 2#################################################################
    def inv_filter(self):
        self.image = self.tmp
        for i in range(0, 3):
            g = self.image[:, :, i]
            G = (np.fft.fft2(g))

            # h = cv2.imread(self.image, 0)
            h_padded = np.zeros(g.shape)
            h_padded[:self.image.shape[0], :self.image.shape[1]] = np.copy(self.image)
            H = (np.fft.fft2(h_padded))

            # normalize to [0,1]
            H_norm = H / abs(H.max())
            G_norm = G / abs(G.max())
            F_temp = G_norm / H_norm
            F_norm = F_temp / abs(F_temp.max())

            # rescale to original scale
            F_hat = F_norm * abs(G.max())

            # 3. apply Inverse Filter and compute IFFT
            self.image = np.fft.ifft2(F_hat)
            self.image[:, :, i] = abs(self.image)
        self.displayImage(2)

##################################Simple Edge Detection#################################################################
    def simple_edge_detection(self):
        # self.image = self.tmp
        # # Convert the img to grayscale
        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # # Apply edge detection method on the image
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # # This returns an array of r and theta values
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        # # The below for loop runs till r and theta values
        # # are in the range of the 2d array
        # for r, theta in lines[0]:
        #     # Stores the value of cos(theta) in a
        #     a = np.cos(theta)
        #     # Stores the value of sin(theta) in b
        #     b = np.sin(theta)
        #     # x0 stores the value rcos(theta)
        #     x0 = a * r
        #     # y0 stores the value rsin(theta)
        #     y0 = b * r
        #     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        #     x1 = int(x0 + 1000 * (-b))
        #     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        #     y1 = int(y0 + 1000 * (a))
        #     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        #     x2 = int(x0 - 1000 * (-b))
        #     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        #     y2 = int(y0 - 1000 * (a))
        #     # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        #     # (0,0,255) denotes the colour of the line to be
        #     # drawn. In this case, it is red.
        #     cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self.image = self.tmp
        
        # 이미지 크기 고정
        height = self.image.shape[0]
        width = self.image.shape[1]
        
        # ROI 설정 (도로 부분)
        roi_vertices = np.array([
            [(50, height),
            (width/2 - 45, height/2 + 60),
            (width/2 + 45, height/2 + 60),
            (width - 50, height)]
        ], dtype=np.int32)
        
        # ROI 마스크 생성
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, roi_vertices, (255,255,255))
        masked_image = cv2.bitwise_and(self.image, mask)
        
        # 차선 색상 필터링
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선 범위
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # 노란색 차선 범위
        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # 마스크 합치기
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 엣지 검출 (점선 차선을 위해 파라미터 조정)
        edges = cv2.Canny(combined_mask, 50, 150)
        
        # 허프 변환 파라미터 조정
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,  # 낮은 임계값으로 점선도 검출
            minLineLength=20,  # 최소 선 길이 감소
            maxLineGap=100  # 최대 갭 증가로 점선 연결
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 0:
                    slope = abs((y2 - y1) / (x2 - x1))
                    # 수직/수평에 가까운 선은 제외
                    if 0.3 < slope < 2:
                        cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
        self.displayImage(2)

##################################### 스무딩 ##########################################################################
    def blur(self):
        self.image = self.tmp
        self.image = cv2.blur(self.image, (5, 5))
        self.displayImage(2)
    def box_filter(self):
        self.image = self.tmp
        self.image = cv2.boxFilter(self.image, -1,(20,20))
        self.displayImage(2)
    def median_filter(self):
        self.image = self.tmp
        self.image = cv2.medianBlur(self.image,5)
        self.displayImage(2)
    def bilateral_filter(self):
        self.image = self.tmp
        self.image = cv2.bilateralFilter(self.image,9,75,75)
        self.displayImage(2)
    def gaussian_filter(self):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image,(5,5),0)
        self.displayImage(2)
    def gaussian_filter2(self, g):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image, (5, 5), g)
        self.displayImage(2)
######################################## 필터 ##########################################################################
    def median_threshold(self):
        self.image = self.tmp
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        retval, threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = threshold
        self.displayImage(2)
    def directional_filtering(self):
        self.image = self.tmp
        kernel = np.ones((3, 3), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering2(self):
        self.image = self.tmp
        kernel = np.ones((5, 5), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering3(self):
        self.image = self.tmp
        kernel = np.ones((7, 7), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def butter_filter(self):
        ### 원본 코드 :: 푸리에변환만하고 필터 기능이 다 빠져있음 ###
        # self.image = self.tmp
        # img_float32 = np.float32(self.image)
        # dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        # self.image = np.fft.fftshift(dft)
        # self.image = 20 * np.log(cv2.magnitude(self.image[:, :, 0], self.image[:, :, 1]))
        # self.displayImage(2)


        ### Revised versision #1(vectorized approach) ###
        self.image = self.tmp
        # Convert to grayscale if needed
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        # Convert to float32 and perform DFT
        img_float32 = np.float32(gray)
        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # Create Butterworth filter
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        D0 = 50  # Cutoff frequency
        n = 2    # Order of filter
        # Create meshgrid for filter calculation
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        u, v = np.meshgrid(v, u)
        D = np.sqrt(u**2 + v**2)
        H = 1 / (1 + (D/D0)**(2*n))
        # Apply filter
        H = np.stack([H, H], axis=2)
        filtered_shift = dft_shift * H
        filtered_ishift = np.fft.ifftshift(filtered_shift)
        # Inverse DFT
        filtered_img = cv2.idft(filtered_ishift)
        filtered_img = cv2.magnitude(filtered_img[:,:,0], filtered_img[:,:,1])
        # Normalize for display
        self.image = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.displayImage(2)

    def notch_filter(self):
        self.image = self.tmp

        self.displayImage(2)

######################################## 특수효과 ##########################################################
    def cartoon(self):
        # num_down = 2
        # num_bilateral = 7

        # img_color = self.image
        # for _ in range(num_down):
        #     img_color = cv2.pyrDown(img_color)

        # for _ in range(num_bilateral):
        #     img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

        # for _ in range(num_down):
        #     img_color = cv2.pyrUp(img_color)

        # img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # img_blur = cv2.medianBlur(img_gray, 7)

        # img_edge = cv2.adaptiveThreshold(img_blur, 255,
        #                                  cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                  cv2.THRESH_BINARY,
        #                                  blockSize=9,
        #                                  C=2)
        # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        # self.image = cv2.bitwise_and(img_color, img_edge)

        # self.displayImage(2)
        # 원본 이미지를 복원하여 항상 같은 크기와 타입 유지
        self.image = self.tmp.copy()

        # 다운샘플링 및 양방향 필터 적용
        num_down = 2  # 다운샘플링 횟수
        num_bilateral = 7  # 양방향 필터 적용 횟수

        img_color = self.image.copy()
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)  # 이미지 다운샘플링

        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)  # 양방향 필터 적용

        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)  # 이미지 업샘플링

        # 엣지 검출
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY,
                                        blockSize=9,
                                        C=2)

        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)  # 엣지 이미지를 컬러로 변환

        # 크기 맞추기
        img_edge = cv2.resize(img_edge, (img_color.shape[1], img_color.shape[0]))

        # 데이터 타입 맞추기
        img_edge = np.uint8(img_edge)

        # 엣지와 색상 이미지를 결합
        self.image = cv2.bitwise_and(img_color, img_edge)

        self.displayImage(2)

    def embossing(self):
        self.image = self.tmp
        
        # 엠보싱 필터 생성
        emboss_filter = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 16비트 정수로 변환
        gray16 = np.int16(gray)
        
        # 필터 적용 및 128 더하기
        filtered = cv2.filter2D(gray16, -1, emboss_filter)
        emboss = np.uint8(np.clip(filtered + 128, 0, 255))
        
        # 결과를 BGR 형식으로 변환하여 저장
        self.image = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
        
        self.displayImage(2)

    def sketchContrast(self):
        self.image = self.tmp
        
        # pencilSketch 함수의 gray 스케치 부분 구현
        self.image = cv2.pencilSketch(
            self.image,
            sigma_s=60,
            sigma_r=0.07,
            shade_factor=0.02
        )[0]  # gray 스케치만 반환
        
        # BGR 형식으로 변환
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        self.displayImage(2)

    def sketchColor(self):
        self.image = self.tmp
        
        # pencilSketch 함수의 color 스케치 부분 구현
        self.image = cv2.pencilSketch(
            self.image,
            sigma_s=60,
            sigma_r=0.07,
            shade_factor=0.02
        )[1]  # color 스케치만 반환
        
        self.displayImage(2)

    def oil(self):
        self.image = self.tmp
        
        # 유화 효과 적용
        self.image = cv2.xphoto.oilPainting(
            self.image,
            size=10,
            dynRatio=1,
            code=cv2.COLOR_BGR2Lab
        )
        
        self.displayImage(2)

######################################## 성별검출 ##########################################################
    def detectSex(self):
        self.image = self.tmp.copy()
        
        # 전처리 및 얼굴 검출
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_cascade = cv2.CascadeClassifier("source/pyqt/data/haarcascade_frontalface_alt2.xml")
        eye_cascade = cv2.CascadeClassifier("source/pyqt/data/haarcascade_eye.xml")
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
        if len(faces) == 0:
            return
        
        # 첫 번째 얼굴 영역 처리
        x, y, w, h = faces[0]
        face_center = (x + w//2, y + h//2)
        
        # 눈 검출
        face_image = self.image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))
        if len(eyes) != 2:
            return
        
        # 눈 중심점 계산
        eye_centers = [(x + ex + ew//2, y + ey + eh//2) for ex, ey, ew, eh in eyes]
        
        # 이미지 회전 보정
        self.image, corr_centers = correct_image(self.image, face_center, eye_centers)
        
        # ROI 검출
        rois = detect_object(face_center, faces[0])
        
        # 히스토그램 계산 및 성별 분류
        masks = make_masks(rois, self.image.shape[:2])
        sims = calc_histo(self.image, rois, masks)
        
        # 유사도 출력
        print("유사도 [입술-얼굴: {:.3f} 윗-귀밑머리: {:.3f}]".format(sims[0], sims[1]))
        
        # 얼굴 특징 표시
        cv2.circle(self.image, face_center, 3, (0, 0, 255), 2)  # 얼굴 중심점
        for center in corr_centers:  # 보정된 눈 중심점
            cv2.circle(self.image, tuple(map(int, center)), 10, (0, 255, 0), 2)
        
        # 얼굴과 입술 타원 그리기
        draw_ellipse(self.image, rois[2], 0.35, (0, 0, 255), 2)  # 얼굴
        draw_ellipse(self.image, rois[3], 0.45, (255, 100, 0), 2)  # 입술
        
        # 결과 표시
        text = "Woman" if sims[1] > 0.3 else "Man"
        cv2.putText(self.image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.displayImage(2)

######################################## Moire Pattern ##########################################################


app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())

