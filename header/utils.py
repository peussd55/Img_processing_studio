import numpy as np
import cv2

# 전처리 수행 함수
def preprocessing(no):
    """
    이미지 전처리: 컬러 이미지를 읽고, 명암도 변환 및 히스토그램 평활화 수행
    """
    image = cv2.imread(f'data/face/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None:
        return None, None  # 이미지가 없을 경우 None 반환

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 변환
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    return image, gray  # 원본 이미지와 명암도 이미지 반환

# 이미지 회전 보정 함수
def correct_image(image, face_center, eye_centers):
    """
    얼굴 중심과 눈 좌표를 기준으로 이미지 회전을 보정합니다.
    """
    pt0, pt1 = eye_centers

    if pt0[0] > pt1[0]:  # 좌우 눈 좌표 교환
        pt0, pt1 = pt1, pt0

    dx, dy = np.subtract(pt1, pt0).astype(float)  # 좌표 차이 계산
    angle = cv2.fastAtan2(dy, dx)  # 회전 각도 계산

    face_center = tuple(map(float, face_center))  # 얼굴 중심 좌표를 튜플로 변환
    rot_mat = cv2.getRotationMatrix2D(face_center, angle, 1)  # 회전 행렬 생성

    size = image.shape[1::-1]  # 이미지 크기
    corr_image = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 보정

    eye_centers = np.expand_dims(eye_centers, axis=0)  # 차원 확장
    corr_centers = cv2.transform(eye_centers, rot_mat)  # 눈 좌표 변환
    corr_centers = np.squeeze(corr_centers, axis=0)  # 차원 감소
    return corr_image, corr_centers

# ROI 정의 함수
def define_roi(pt, size):
    """
    2개의 좌표를 결합하여 4원소 튜플로 변환합니다.
    """
    return np.ravel((pt, size)).astype('int')

# 얼굴 내 객체 탐지 함수
def detect_object(center, face):
    """
    얼굴 영역에서 머리카락 및 입술 영역을 검출합니다.
    """
    x, y, w, h = face
    center = np.array(center)

    gap1 = np.multiply((w, h), (0.45, 0.65))  # 좌상단 평행이동 비율
    gap2 = np.multiply((w, h), (0.18, 0.1))   # 우하단 평행이동 비율

    # 머리 영역 정의
    pt1 = center - gap1
    pt2 = center + gap1
    hair = define_roi(pt1, pt2 - pt1)

    # 머리카락 영역 정의
    size = np.multiply(hair[2:4], (1, 0.4))  # 머리카락 영역 비율
    hair1 = define_roi(pt1, size)
    hair2 = define_roi(pt2 - size, size)

    # 입술 영역 정의
    lip_center = center + np.array([0, int(h * 0.3)])  # 입술 중심
    lip1 = lip_center - gap2
    lip2 = lip_center + gap2
    lip = define_roi(lip1, lip2 - lip1)

    return [hair1, hair2, lip, hair]
