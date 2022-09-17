import numpy as np
from PIL import ImageGrab, Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget, QFileDialog
from matplotlib import pyplot as plt
import tensorflow as tf

from myUI import Ui_Form
from data_util import handle_imagePath, \
    Alphabet_Lower_Mapping_List, \
    Alphabet_Upper_Mapping_List, \
    Alphabet_Mapping_List, \
    ar_to_tf, \
    ar_to_arBin
import model as md


class myFunction(QWidget, Ui_Form):
    def __init__(self):
        # 初始化UI
        super(myFunction, self).__init__()
        self.setupUi(self)
        # 关闭鼠标跟踪，当鼠标按下时窗口才能跟踪
        self.setMouseTracking(False)
        # 设置变量
        self.pen = QPen(Qt.black, 20, Qt.SolidLine)  # 初始化画笔
        self.pen.setCapStyle(Qt.RoundCap)  # 设置笔帽为圆帽
        self.pos_xy = []  # 记录鼠标在画板边界内的坐标
        self.label_range = [0, 0, 0, 0]  # 记录画板边界
        self.combineFlag = True  # 是否混合大小写识别标志
        self.checkPointDir = './model_data/checkpoint/'
        self.modelName = 'ResNet18'
        self.img_tf = None
        # 设置事件
        self.pushButton.clicked.connect(self.btn_openFile)  # 打开文件按钮
        self.pushButton_2.clicked.connect(self.btn_discern)  # 开始识别按钮
        self.pushButton_3.clicked.connect(self.btn_clear)  # 清除画板按钮
        self.pushButton_4.clicked.connect(self.btn_saveFile)  # 保存画板按钮
        self.horizontalSlider.valueChanged.connect(self.lcdNumber.display)  # 画笔大小调节滑动条
        self.horizontalSlider.valueChanged.connect(self.pen.setWidth)
        self.checkBox.stateChanged.connect(self.cbx_change)  # 是否混合识别选择
        self.comboBox.activated[str].connect(self.cbb_onActivated)  # 卷积神经网络模型选择
        # 卷积神经网络模型初始化
        self.model = md.ResNet18([2, 2, 2, 2], num_classes=26)
        self.model.load_weights(self.checkPointDir + self.modelName
                                + ('_combine' if self.combineFlag else '')
                                + '.ckpt')

    # 画笔绘画
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(self.pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                # 判断是否是断点
                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    # 按住鼠标事件
    def mouseMoveEvent(self, event):
        """
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        """

        self.get_limit_range()
        # 中间变量pos_tmp提取当前点
        pos_tmp = self.range_limit(event.pos().x(), event.pos().y())
        # 若鼠标开始在画板中绘画，清除画板显示图片
        if pos_tmp != (-1, -1):
            self.clearPixMap()
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    # 鼠标松开事件
    def mouseReleaseEvent(self, event):
        """
            鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        """
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    # 打开文件按钮事件
    def btn_openFile(self):
        f_name = QFileDialog.getOpenFileName(self, '选择一张图片文件', './predict_img', 'images(*.png *.jpg)')

        # 若有选择图片文件，更新img_tf,清除画板内容，显示选择图片，预测图片内容
        if f_name[0]:
            self.btn_clear()
            self.img_tf = handle_imagePath(f_name[0])
            self.label.setPixmap(QPixmap(f_name[0]))
            self.label.setScaledContents(True)  # 根据label控件尺寸大小缩放
            self.predict(self.img_tf)
            self.update()

    # TODO 保存文件按钮事件
    def btn_saveFile(self):
        f_path = QFileDialog.getSaveFileName(self, '选择一张图片文件', './predict_img', 'images(*.png)')
        img = self.grabLabel()
        img.save(f_path[0])

    # 开始识别按钮事件
    def btn_discern(self):
        # 若画板有笔迹，则截图更新img_tf
        if self.pos_xy:
            img = self.grabLabel()
            im = img.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
            # plt.imshow(im)
            # plt.show()
            im_array = np.array(im).astype(np.float32)
            im_array_bin = ar_to_arBin(im_array, threshold=200)
            # plt.imshow(im_array_bin, cmap="gray")
            # plt.show()
            self.img_tf = ar_to_tf(im_array_bin)
        # print(im)
        # plt.imshow(im)
        # plt.show()
        # recognize_result = self.recognize_img(im)  # 调用识别函数
        # self.label.setText(str(recognize_result))  # 显示识别结果
        # self.update()
        self.predict(self.img_tf)

    # 清除画板内容
    def btn_clear(self):
        self.pos_xy = []
        self.label_4.setText('')
        self.clearPixMap()
        self.img_tf = None
        self.update()

    # 混合识别按钮改变
    def cbx_change(self, state):
        if state == Qt.Checked:
            self.combineFlag = True
        else:
            self.combineFlag = False
        self.model_change()

    # 模型选择按钮改变
    def cbb_onActivated(self, text):
        self.modelName = text
        self.model_change()

    # 清除label显示的图片
    def clearPixMap(self):
        self.label.setPixmap(QPixmap())

    # 过滤超出画板的鼠标坐标
    def range_limit(self, x, y):

        if (x < self.label_range[0]) or (x > self.label_range[2]):
            pos_x = -1
            pos_y = -1

        elif (y < self.label_range[1]) or (y > self.label_range[3]):
            pos_x = -1
            pos_y = -1
        else:
            pos_x = x
            pos_y = y

        return pos_x, pos_y

    # 画板范围限制
    def get_limit_range(self, offset=20):
        self.label_range = [self.label.x() + offset,
                            self.label.y() + offset,
                            self.label.x() + self.label.width() - offset,
                            self.label.y() + self.label.height() - offset]

    # 截取画板图像
    def grabLabel(self):
        self.get_limit_range()
        im_range = [self.label_range[0] + self.x(),
                    self.label_range[1] + self.y() + 20,
                    self.label_range[2] + self.x(),
                    self.label_range[3] + self.y()]
        im = ImageGrab.grab(im_range)  # 截屏，手写数字部分
        # plt.imshow(im)
        # plt.show()
        im = im.convert('L')
        return im

    # 识别并显示结果
    def predict(self, data):
        if data != None:
            result = self.model.predict(data)
            if self.combineFlag:
                label_text = Alphabet_Upper_Mapping_List[np.argmax(result[0])] \
                             + '/' \
                             + Alphabet_Lower_Mapping_List[np.argmax(result[0])]
            else:
                label_text = Alphabet_Mapping_List[np.argmax(result[0])]
            self.label_4.setText(label_text)
            self.update()


    # 模型发生改变
    def model_change(self):
        if self.combineFlag:
            num = 26
        else:
            num = 52

        if self.modelName.upper() == 'ResNet18'.upper():
            self.model = md.ResNet18([2, 2, 2, 2], num_classes=num)
        elif self.modelName.upper() == 'Inception10'.upper():
            self.model = md.Inception10(num_blocks=2, num_classes=num)
        elif self.modelName.upper() == 'VGG16'.upper():
            self.model = md.VGG16(num_classes=num)
        elif self.modelName.upper() == 'AlexNet8'.upper():
            self.model = md.AlexNet8(num_classes=num)
        elif self.modelName.upper() == 'LeNet5'.upper():
            self.model = md.LeNet5(num_classes=num)
        print('--------------------------load-----------------------------------')
        print(self.checkPointDir + self.modelName
                                + ('_combine' if self.combineFlag else '')
                                + '.ckpt')
        self.model.load_weights(self.checkPointDir + self.modelName
                                + ('_combine' if self.combineFlag else '')
                                + '.ckpt')
        self.predict(self.img_tf)

