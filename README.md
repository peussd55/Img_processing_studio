# OpenCV-PyQt5 이미지 처리 스튜디오

> 📌 **프로젝트 유형**: 데스크톱 애플리케이션 · 컴퓨터 비전 · 인터랙티브 이미지 처리 툴킷

## 1. 프로젝트 개요
PyQt5 기반 GUI와 OpenCV 연산을 결합하여 다양한 영상 보정, 필터링, 효과 적용을 실시간으로 실험할 수 있는 데스크톱 스튜디오입니다. 메뉴와 슬라이더, 체크박스를 통해 이미지를 불러오고, 회전·스케일 조정부터 색상 보정, 주파수 필터, 만화/유화 효과, 얼굴 성별 분류까지 폭넓은 컴퓨터 비전 파이프라인을 한 화면에서 실행할 수 있습니다.

## 2. 주요 기능
- **파일 관리**: 이미지 열기/저장/인쇄 및 원본 복원.
- **기하 변환**: 회전(수동 다이얼 포함), 아핀 변환, 크기 조절.
- **색상/명암 보정**: 그레이스케일, 네거티브, 히스토그램 평활화, 로그·감마 변환.
- **노이즈 및 스무딩**: 박스/가우시안/미디언/양방향 블러, 침식·팽창 슬라이더.
- **주파수/에지 필터**: Canny 에지, 방향성 필터, Butterworth 저역 통과 필터.
- **특수 효과**: 만화, 엠보싱, 연필 스케치(흑백·컬러), 유화 효과.
- **성별 분류**: Haar Cascade 얼굴·눈 검출 후 ROI 히스토그램 유사도 기반 여성/남성 판별.

## 3. 모듈 구성 및 데이터 흐름
```mermaid
flowchart TD
    UI[PyQt5 UI (demo.ui)] -->|신호/슬롯| Main(Pyqt5ui.LoadQt)
    Main -->|파일 선택| ImgIO
    Main -->|OpenCV 연산| Filters
    Filters --> Display[QLabel 미리보기]
    Main -->|성별 분류| Gender
    subgraph "header 패키지"
        Filters[필터 & 효과]
        Gender --> Preproc[utils.preprocessing]
        Gender --> Align[utils.correct_image]
        Gender --> ROI[utils.detect_object]
        Gender --> Histo[histogram.calc_histo]
        Gender --> Classify[classify.classify]
    end
```
1. **Pyqt5ui.LoadQt**: `demo.ui`를 불러와 메뉴/슬라이더 이벤트를 슬롯에 연결하고, OpenCV 연산 결과를 두 개의 QLabel에 렌더링합니다.
2. **header.utils**: 얼굴 전처리, 회전 보정, ROI 산출을 담당합니다.
3. **header.histogram**: 머리/입술/얼굴 영역에 대한 마스크와 3채널 히스토그램을 계산하여 유사도를 반환합니다.
4. **header.classify**: 유사도 지표를 바탕으로 콘솔 및 화면에 성별을 출력하고 주요 특징점에 오버레이를 그립니다.

## 4. 기술 스택
- **언어 & GUI**: Python 3, PyQt5, Qt Designer (`demo.ui`)
- **영상 처리**: OpenCV (opencv-python, opencv-contrib-python), NumPy
- **신호 처리 & 수학**: SciPy, matplotlib (히스토그램 시각화 옵션)

## 5. 디렉터리 구조
```
OpenCV-Pyqt5/
├── Pyqt5ui.py            # 메인 애플리케이션 진입점
├── demo.ui               # Qt Designer에서 생성한 UI 레이아웃
├── header/               # 전처리, 히스토그램, 분류 유틸리티
│   ├── utils.py
│   ├── histogram.py
│   └── classify.py
├── data/                 # Haar cascade 및 샘플 XML 모델
├── *.png                 # 아이콘 및 샘플 이미지 자산
└── README.md
```

## 6. 설치 및 실행
1. 저장소 클론 후 가상환경을 준비합니다.
   ```bash
   git clone <repository-url>
   cd OpenCV-Pyqt5
   python -m venv .venv
   source .venv/bin/activate  # Windows는 .venv\Scripts\activate
   ```
2. 필수 패키지를 설치합니다. `cv2.xphoto.oilPainting`을 사용하므로 `opencv-contrib-python`이 필요합니다.
   ```bash
   pip install PyQt5 opencv-contrib-python numpy scipy matplotlib
   ```
3. Qt UI 및 자산 경로를 확인합니다. 기본 코드는 `source/pyqt/...` 하위 경로를 참조하므로 현재 구조에 맞게 수정하거나 심볼릭 링크/폴더를 구성해야 합니다.
   ```python
   # Pyqt5ui.py 예시 수정
   loadUi('demo.ui', self)
   self.setWindowIcon(QtGui.QIcon('python-icon.png'))
   face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
   ```
4. 애플리케이션을 실행합니다.
   ```bash
   python Pyqt5ui.py
   ```

## 7. UI 사용 가이드
- **상단 메뉴바**
  - *File*: 이미지 열기, 다른 이름으로 저장, 인쇄, 종료.
  - *View*: 확대/축소, 다이얼 기반 회전, 사이즈 조절.
  - *Smoothing & Filter*: 블러 계열, 방향성 필터, Butterworth, Notch(구현 예정) 등.
  - *특수효과 / Cartooning*: 만화, 엠보싱, 스케치, 유화.
  - *Sex Detect*: 얼굴을 포함한 이미지를 선택하면 성별 판별 결과와 ROI 오버레이를 보여줍니다.
- **우측 패널**
  - 각종 슬라이더(QDial, QSlider)로 회전 각도, 감마 값, 가우시안 커널, 침식 반복 횟수, 로그 스케일, Canny 임계값 등을 조절할 수 있습니다.
  - `Reset` 버튼으로 현재 이미지 상태를 원본으로 복원할 수 있습니다.
- **이미지 라벨**
  - 좌측은 원본/중간 결과, 우측은 후처리된 이미지를 표시합니다.

## 8. 성별 분류 파이프라인
1. Haar Cascade로 얼굴과 눈 위치를 검출합니다 (`data/haarcascade_*.xml`).
2. 검출된 눈 좌표를 이용해 얼굴 이미지를 회전 보정합니다 (`utils.correct_image`).
3. 머리카락/입술 영역 ROI를 생성하고 히스토그램 기반 마스크를 적용합니다 (`histogram.make_masks`).
4. 입술-얼굴, 머리카락-머리카락 히스토그램 유사도를 계산하여 기준값 대비 크기를 비교합니다 (`histogram.calc_histo`).
5. `classify.classify`가 최종 텍스트(Man/Woman)를 결정하고 콘솔 및 화면에 출력합니다.

## 9. 개발 및 커스터마이징 팁
- 필터/효과 함수는 `LoadQt` 클래스 메서드로 구현되어 있으므로 새로운 기능을 추가할 때는 메뉴 액션과 슬롯을 연결한 뒤, `self.tmp`(원본)와 `self.image`(현재 상태) 흐름을 유지하세요.
- `header` 모듈은 성별 분류와 관련된 로직에 집중되어 있으며, 독립적인 데이터 전처리 파이프라인으로 재사용 가능합니다.
- SciPy 기반 위너 필터, 주파수 영역 노치 필터 등 확장 포인트가 마련되어 있으나 일부 함수는 샘플/미구현 상태입니다. 필요 시 파라미터 튜닝과 예외 처리를 추가하세요.

## 10. 향후 개선 아이디어
- UI 리소스 경로 통합 및 다국어 지원.
- Butterworth/Notch 필터 파라미터를 사용자 입력으로 설정할 수 있는 대화상자 추가.
- 얼굴 검출 실패 시 사용자 알림 및 다중 얼굴 지원.
- OpenCV DNN 또는 딥러닝 모델을 활용한 고도화된 성별/연령 분류 기능.
- 이미지 처리 결과를 단계별로 비교할 수 있는 히스토리/레이어 기능.
