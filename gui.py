# -*- coding: utf-8 -*-
"""CeraBeads 拼豆图纸生成器 · 图形界面（PyQt6）

运行：
  - 打包版：双击 CeraBeads.exe / 启动.bat
  - 源码版：py -3 gui.py
自检：gui.py --selftest（写 _selftest.txt 到程序目录）

功能：
  · 拖入图片（可多张，列表管理）
  · 滑杆调：格子大小 / 颜色数上限 / 背景容差 / 参考图透明度
  · 点格改色：选色后点格涂色、右键清空、吸管从原图取色、Ctrl+Z 撤销
  · 参考图叠加对照
  · 导出：成品 PNG / 纯净 PNG / JPG / 打印 PDF（按板分页）/ 用量 CSV，可多选一次导出多张
"""
import copy
import os
import sys
import time
import shutil
import traceback

import numpy as np

from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QComboBox, QCheckBox, QSlider,
                             QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
                             QGroupBox, QFormLayout, QProgressBar, QLineEdit, QSplitter,
                             QScrollArea, QFrame, QSizePolicy,
                             QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QToolButton,
                             QScrollArea)

from PyQt6.QtWidgets import QRadioButton, QButtonGroup
import bead_core as B

APP_TITLE = 'CeraBeads 拼豆图纸生成器'
APP_VERSION = 'v1.6.2'
IMG_EXT = B.IMG_EXT      # 与引擎共用：含 HEIC/HEIF/AVIF 等（见 bead_core）


def app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def res_dir():
    return getattr(sys, '_MEIPASS', None) or app_dir()


def icon_path():
    for p in (os.path.join(res_dir(), 'app.ico'), os.path.join(app_dir(), 'app.ico')):
        if os.path.exists(p):
            return p
    return ''


def pil2pixmap(im):
    im = im.convert('RGB')
    data = im.tobytes('raw', 'RGB')
    qimg = QImage(data, im.width, im.height, im.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def refresh_counts(pat):
    """改格之后重算用量清单"""
    import numpy as np
    m = pat['matcher']
    flat = pat['idx'].reshape(-1)
    used = flat[flat >= 0]
    usage = np.bincount(used, minlength=len(m.codes)) if len(used) else np.zeros(len(m.codes), int)
    counts = sorted([(i, int(u)) for i, u in enumerate(usage) if u > 0], key=lambda t: -t[1])
    pat['counts'] = counts
    pat['total'] = int(usage.sum())
    return pat


class GenWorker(QThread):
    done = pyqtSignal(object, float, str)

    def __init__(self, src, opts):
        super().__init__()
        self.src = src
        self.opts = dict(opts)

    def run(self):
        t0 = time.time()
        try:
            pat = B.build_pattern(self.src, **self.opts)
            self.done.emit(pat, time.time() - t0, '')
        except Exception:
            self.done.emit(None, time.time() - t0, traceback.format_exc())


class Preview(QGraphicsView):
    cellHover = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.sc = QGraphicsScene(self)
        self.setScene(self.sc)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(244, 244, 246))
        self.item = None
        self.pix = None
        self.layout = None          # dict(x0,y0,cell,gw,gh)

    def set_image(self, pil_img, layout=None, keep_view=False):
        self.pix = pil2pixmap(pil_img)
        self.layout = layout
        if self.item is None or not keep_view:
            self.sc.clear()
            self.item = self.sc.addPixmap(self.pix)
            self.sc.setSceneRect(QRectF(self.pix.rect()))
            if not keep_view:
                self.fit()
        else:
            self.item.setPixmap(self.pix)
            self.sc.setSceneRect(QRectF(self.pix.rect()))

    def fit(self):
        if self.item:
            self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def actual(self):
        self.resetTransform()

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def _cell_at(self, pos):
        if not (self.layout and self.pix):
            return None
        p = self.mapToScene(pos)
        L = self.layout
        c = int((p.x() - L['x0']) // L['cell'])
        r = int((p.y() - L['y0']) // L['cell'])
        if 0 <= r < L['gh'] and 0 <= c < L['gw']:
            return r, c
        return None

    def mouseMoveEvent(self, e):
        rc = self._cell_at(e.position().toPoint())
        if rc:
            self.cellHover.emit(rc[0], rc[1])
        super().mouseMoveEvent(e)


class ExportDialog(QDialog):
    """一次挑好要出哪几张图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('批量导出')
        v = QVBoxLayout(self)
        v.addWidget(QLabel('勾选要导出的成品（可以多选，会一次全部导出）：'))
        self.ck = {}
        for key, label, default in (
                ('full', '成品图纸 PNG（带色号 + 坐标 + 用量清单）', True),
                ('plain', '纯净图纸 PNG（不带色号，方便对照拼）', True),
                ('multi', '多尺寸 PNG（20 / 32 / 48 像素每格，各一张）', False),
                ('jpg', 'JPG（分享给朋友/发微信，质量 92）', True),
                ('pdf', '打印 PDF（按 29×29 板分页，可直接打印拼装）', True),
                ('csv', '用量 CSV（采购清单）', False)):
            c = QCheckBox(label)
            c.setChecked(default)
            self.ck[key] = c
            v.addWidget(c)
        v.addWidget(QLabel('命名：<标题>_图纸.png / _纯净.png / _图纸_32px.png / _分享.jpg / _打印.pdf / _用量.csv'))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText('选文件夹并导出')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def chosen(self):
        return [k for k, c in self.ck.items() if c.isChecked()]


PREVIEW_MAX_MP = 400.0      # 预览绝对上限（百万像素）；实际按本机可用内存动态算（见 preview_cell_cap）
HUGE_MP = 40.0              # 超过这个百万像素数 → 用流式写 PNG（不拦、不缩，照样出全尺寸）
JPG_MAX_SIDE = 8000         # 分享 JPG 的最大单边（JPEG 编码器限制，仅此一项封顶）


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pattern = None
        self.src_path = ''
        self.worker = None
        self.preview_cell = 32
        self.auto_run = True
        self.cur_cell = 32
        self.setWindowTitle('%s %s' % (APP_TITLE, APP_VERSION))
        self.setMinimumSize(1020, 480)      # 下限放宽：窗口要能跟着内容缩
        self.setFont(QFont('Microsoft YaHei', 9))
        self.setAcceptDrops(True)
        self.build_ui()
        self._install_drops()          # 窗口里任何地方都能接住拖入的图片
        self._fit_window()

    def _fit_window(self):
        """窗口按内容自适应：左栏只留“刚好放得下控件”的宽度，高度跟着当前档位的内容走。
        预览至少留 720px 宽；不越过屏幕 90%；最大化时不改。"""
        try:
            sa = self.findChild(QScrollArea)
            if sa is None or sa.widget() is None or self.isMaximized():
                return
            left = sa.widget()
            for _g in left.findChildren(QGroupBox):   # 先把隐藏/显示的行算完，再量尺寸
                _l = _g.layout()
                if _l is not None:
                    _l.activate()
            if left.layout() is not None:
                left.layout().activate()
            left.adjustSize()
            need_w = left.sizeHint().width() + 24
            need_h = left.sizeHint().height() + 96
            scr = self.screen().availableGeometry() if self.screen() else None
            max_w = int(scr.width() * 0.9) if scr else 2400
            max_h = int(scr.height() * 0.9) if scr else 1300
            win_w = min(max_w, max(1020, need_w + 720))
            win_h = min(max_h, max(500, need_h))
            self.resize(win_w, win_h)
            sp = sa.parentWidget()
            if sp is not None and hasattr(sp, 'sizes'):
                tot = sum(sp.sizes()) or win_w
                sp.setSizes([need_w, max(200, tot - need_w)])
        except Exception:
            pass

    # ------------------------------------------------------------- UI
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        sp = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(sp)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(4)

        # ── 设置档位：普通（小白，只留 4 个旋钮）／高级（全部参数）──
        _mw = QHBoxLayout()
        self.rb_ui_basic = QRadioButton('普通设置（推荐）')
        self.rb_ui_basic.setToolTip('只显示必须调的：优化模式、格数、最大格数上限、色卡。其余按实测最好的默认值来')
        self.rb_ui_adv = QRadioButton('高级设置（全部参数）')
        self.rb_ui_adv.setToolTip('显示全部参数：限色/去背景/取色方式/噪点清理/统一相近色/红线间隔/'
                                  '印色号坐标/参考图叠加/标题/点格改色')
        self.ui_group = QButtonGroup(self)
        self.ui_group.addButton(self.rb_ui_basic)
        self.ui_group.addButton(self.rb_ui_adv)
        self.rb_ui_basic.setChecked(True)
        self.ui_group.buttonClicked.connect(lambda *_: self.apply_mode(True))
        _mw.addWidget(QLabel('设置档位'))
        _mw.addWidget(self.rb_ui_basic)
        _mw.addWidget(self.rb_ui_adv)
        _mw.addStretch(1)
        lv.addLayout(_mw)
        self.lb_ui_hint = QLabel('')
        self.lb_ui_hint.setWordWrap(True)
        self.lb_ui_hint.setStyleSheet('color:#23408e;background:#f2f7ff;border:1px solid #cfe0f7;'
                                      'border-radius:4px;padding:6px')
        self.b_gen_top = QPushButton('▶ 生成图纸')
        self.b_gen_top.setMinimumHeight(34)
        self.b_gen_top.setToolTip('按当前参数出图 / 改了参数后重新计算（Ctrl+Enter 也行）')
        self.b_gen_top.setMinimumWidth(0)
        self.b_gen_top.clicked.connect(lambda: self.generate())
        self.b_batch = QPushButton('📦 批量导出')
        self.b_batch.setMinimumHeight(34)
        self.b_batch.setToolTip('列表里每张图都出图并导出：勾好成品（PNG/纯净/JPG/打印PDF/用量CSV）'
                                '→ 选目录，一次全出')
        self.b_batch.setMinimumWidth(0)
        self.b_batch.clicked.connect(self.batch_export)
        self.b_all = self.b_batch
        _tb = QHBoxLayout()                 # 两个按钮并列在顶部（文字短，窄窗口也不会被裁）
        _tb.setSpacing(6)
        _tb.addWidget(self.b_gen_top, 1)
        _tb.addWidget(self.b_batch, 1)
        lv.addLayout(_tb)
        lv.addWidget(self.lb_ui_hint)
        self.adv_rows = []      # 只有高级档才显示的 (表单, 行号)
        self.bs_rows = []       # 只有普通档才显示的小字提示
        self.adv_w = []         # 只有高级档才显示的控件
        self.basic_w = []

        g1 = QGroupBox('① 图片（可拖入多张）')
        v1 = QVBoxLayout(g1)
        self.lst = QListWidget()
        self.lst.setMaximumHeight(48)
        self.lst.currentRowChanged.connect(lambda i: self.load_image(
            self.lst.item(i).data(Qt.ItemDataRole.UserRole)) if i >= 0 else None)
        row = QHBoxLayout()
        for t, fn in (('添加…', self.pick_images), ('移除', self.remove_sel), ('清空', self.clear_imgs)):
            b = QPushButton(t)
            b.clicked.connect(fn)
            row.addWidget(b)
        v1.addWidget(self.lst)
        v1.addLayout(row)
        lv.addWidget(g1)

        g2 = QGroupBox('② 图纸参数')
        f2 = QFormLayout(g2)
        self.sp_w = QSpinBox()
        self.sp_w.setRange(8, 400)
        self.sp_w.setValue(96)
        self.sl_w = QSlider(Qt.Orientation.Horizontal)
        self.sl_w.setRange(8, 200)
        self.sl_w.setValue(96)
        self.sl_w.valueChanged.connect(lambda v: self.sp_w.setValue(v))
        self.sp_w.valueChanged.connect(lambda v: self.sl_w.setValue(min(self.sl_w.maximum(), v)))
        self.sp_w.valueChanged.connect(self.on_w_changed)
        self.sp_h = QSpinBox()
        self.sp_h.setRange(8, 400)
        self.sp_h.setValue(103)
        self.sl_h = QSlider(Qt.Orientation.Horizontal)
        self.sl_h.setRange(8, 200)
        self.sl_h.setValue(103)
        self.sl_h.valueChanged.connect(lambda v: self.sp_h.setValue(v))
        self.sp_h.valueChanged.connect(lambda v: self.sl_h.setValue(min(self.sl_h.maximum(), v)))
        self.ck_ratio = QCheckBox('按图片比例自动算高')
        self.ck_ratio.setChecked(True)
        r1 = QHBoxLayout()
        r1.addWidget(self.sl_w)
        r1.addWidget(self.sp_w)
        f2.addRow(QLabel('宽（格）'), r1)
        r2 = QHBoxLayout()
        r2.addWidget(self.sl_h)
        r2.addWidget(self.sp_h)
        r2.addWidget(self.ck_ratio)
        f2.addRow(QLabel('高（格）'), r2)
        for _w2 in (self.sl_w, self.sp_w, self.sl_h, self.sp_h, self.ck_ratio):
            _w2.setToolTip('格数＝图纸有多少格，越多越清晰、也拼得越久。'
                           '拖入图片后会自动给一个合适的格数，想更细就把「宽」调大。')
        self.sp_max = QSpinBox()
        self.sp_max.setRange(50, 5000)
        self.sp_max.setValue(104)   # 默认 104 格
        self.sp_max.setToolTip('宽/高的最大格数上限：想画大图就把这里调大（上限越大，预览会自动缩小显示）')
        self.sp_max.valueChanged.connect(self.on_max_changed)
        # 不进 bs_rows / adv_rows：两个档位都显示（普通档想画大图就要调它）
        self.cb_pal = QComboBox()
        self.cb_pal.addItems(B.palette_labels())
        _rp = QHBoxLayout()
        _rp.addWidget(self.cb_pal, 1)
        _rp.addSpacing(8)
        _rp.addWidget(QLabel('最大格数'))
        _rp.addWidget(self.sp_max)
        f2.addRow(QLabel('色卡'), _rp)
        self.ck_limit = QCheckBox('限制颜色数')
        self.ck_limit.setChecked(False)   # 默认关
        self.ck_limit.stateChanged.connect(self.on_limit_changed)
        self.sl_maxc = QSlider(Qt.Orientation.Horizontal)
        self.sl_maxc.setRange(2, 120)
        self.sl_maxc.setValue(43)
        self.sl_maxc.setEnabled(False)   # 配合上面的默认关
        self.lb_maxc = QLabel('43 色')
        self.lb_maxc.setMinimumWidth(48)
        self.sl_maxc.valueChanged.connect(lambda v: self.lb_maxc.setText('%d 色' % v))
        r3 = QHBoxLayout()
        r3.addWidget(self.ck_limit)
        r3.addWidget(self.sl_maxc)
        r3.addWidget(self.lb_maxc)
        f2.addRow(QLabel('颜色数'), r3)
        self.adv_rows.append((f2, f2.rowCount() - 1))
        self.ck_bg = QCheckBox('自动去掉背景')
        self.ck_bg.setChecked(False)     # 默认关（免得把浅色主体当背景抠掉）
        self.sl_bg = QSlider(Qt.Orientation.Horizontal)
        self.sl_bg.setRange(2, 90)
        self.sl_bg.setValue(int(B.BG_TOL))
        self.lb_bg = QLabel('容差 %d' % self.sl_bg.value())
        self.lb_bg.setMinimumWidth(60)
        self.sl_bg.valueChanged.connect(lambda v: self.lb_bg.setText('容差 %d' % v))
        r4 = QHBoxLayout()
        r4.addWidget(self.ck_bg)
        r4.addWidget(self.sl_bg)
        r4.addWidget(self.lb_bg)
        f2.addRow(QLabel('背景'), r4)
        self.adv_rows.append((f2, f2.rowCount() - 1))
        self.cb_sample = QComboBox()
        self.cb_sample.addItems(['跟随优化模式（推荐）', '照片优化（加权平均）', '平均色', '主导色'])
        self.cb_sample.setToolTip('跟随优化模式（推荐）：优化模式选什么就用什么取色\n'
                                  '照片优化：格内以主导色为基准的加权平均（保层次、不出灰边）\n'
                                  '平均色：格内像素线性光平均（最“平”）\n'
                                  '主导色：取格内占比最大的那一种颜色（色块最干净）')
        self.cb_sample.currentIndexChanged.connect(
            lambda *_: getattr(self, 'auto_run', False) and self.generate())
        self.preset_group = QButtonGroup(self)
        self.rb_pre = []
        _pr = QHBoxLayout()            # 横排：通用 / 图片优化 / 动漫优化
        _pr.setSpacing(12)
        _tips = {
            'general': '通用：不预设任何专项处理——整格取平均色，最忠实、最稳；照片/线稿都可用',
            'photo': '图片优化（照片首选）：先做小半径双边滤波去噪（保边缘），再按“格内主导色引导的加权平均”取色——保层次、不出灰边',
            'anime': '动漫优化（线稿/卡通首选）：先用 Sobel 找出抗锯齿过渡像素并从投票里剔除，再按格取主导色——轮廓不会被灰边拉偏',
        }
        for _k, _lab in B.PRESETS:                 # 通用 / 图片优化 / 动漫优化
            rb = QRadioButton(_lab)
            rb.setToolTip(_tips.get(_k, ''))
            self.preset_group.addButton(rb)
            self.rb_pre.append(rb)
            _pr.addWidget(rb)
        _pr.addStretch(1)
        self.rb_pre[0].setChecked(True)            # 默认「通用」：由你选，不替你判断
        self.preset_group.buttonClicked.connect(lambda *_: self.on_preset_changed())
        f2.addRow(QLabel('优化模式'), _pr)
        f2.addRow(QLabel('取色方式'), self.cb_sample)
        self.adv_rows.append((f2, f2.rowCount() - 1))
        r6 = QVBoxLayout()          # 两行：①清理碎点+阈值 ②标出孤岛 ③……见下
        r6.setSpacing(2)
        self.ck_desp = QCheckBox('清理碎点')
        self.ck_up = QCheckBox('小图先放大')
        self.ck_up.setChecked(True)    # 实测：小图放大后再取色，色号更接近原图（照片 84%→92%）
        self.ck_up.setToolTip('每格不足 12 原图像素时，先放大再取色（最多 4×，零依赖）。\n'
                             '实测比不放大更接近原图（照片 84%→92%、线稿 88%→90%+）；'
                             '不需要就取消勾选——关掉后按原图像素直接取色')
        self.ck_isle = QCheckBox('标出孤岛')
        self.ck_isle.setToolTip('把只有 1~2 格的同色小块用洋红线框出来（只在预览里画标记，'
                             '不会写进出图）。\n要不要清掉请用「清理碎点」')
        self.ck_desp.setChecked(False)     # 默认关
        self.ck_desp.setToolTip('把面积太小的孤立色块（量化噪点）并到旁边的颜色，成品更像“能拼的图”')
        self.sp_desp = QSpinBox()
        self.sp_desp.setRange(2, 9)
        self.sp_desp.setValue(3)
        self.sp_desp.setSuffix(' 格以下')
        self.ck_harm = QCheckBox('统一相近色')
        self.ck_harm.setChecked(True)
        self.ck_harm.setToolTip('防伪杂色：量化时加一层 4 邻域一致性，把“色差极小却当成两个色号”的格'
                                '统一到邻居用的那个色号，\n从源头减少伪杂色（买豆子更省、图纸更干净），'
                                '只会合并、绝不新增色号')
        _r6a = QHBoxLayout()        # 第一行：清理碎点 + 阈值 + 标出孤岛
        _r6a.addWidget(self.ck_desp)
        _r6a.addWidget(self.sp_desp)
        _r6a.addWidget(self.ck_isle)
        _r6a.addStretch(1)
        _r6b = QHBoxLayout()        # 第二行：小图先放大 + 统一相近色
        _r6b.addWidget(self.ck_up)
        _r6b.addWidget(self.ck_harm)
        _r6b.addStretch(1)
        r6.addLayout(_r6a)
        r6.addLayout(_r6b)
        f2.addRow(QLabel('噪点/杂色'), r6)
        self.adv_rows.append((f2, f2.rowCount() - 1))
        lv.addWidget(g2)

        self.g3 = g3 = QGroupBox('③ 图纸显示')
        f3 = QFormLayout(g3)
        self.sl_cell = QSlider(Qt.Orientation.Horizontal)
        self.sl_cell.setRange(4, 256)
        self.sl_cell.setValue(32)
        self.lb_cell = QLabel('32 px')
        self.lb_cell.setMinimumWidth(56)
        self.sl_cell.setToolTip('只决定图纸上格子画多大（看/打印），不影响颜色效果；'
                                '想印大一点就调大，导出都按这个值')
        self.sl_cell.valueChanged.connect(self.on_cell_changed)
        self.sp_red = QSpinBox()
        self.sp_red.setRange(4, 50)
        self.sp_red.setValue(10)
        r5 = QHBoxLayout()
        r5.addWidget(self.sl_cell)
        r5.addWidget(self.lb_cell)
        r5.addSpacing(10)
        r5.addWidget(QLabel('红线间隔'))
        r5.addWidget(self.sp_red)
        r5.addStretch(1)
        f3.addRow(QLabel('每格像素'), r5)
        self.adv_rows.append((f3, f3.rowCount() - 1))
        self._cell_timer = QTimer(self)
        self._cell_timer.setSingleShot(True)
        self._cell_timer.timeout.connect(self._cell_apply)
        self.ck_codes = QCheckBox('印色号')
        self.ck_codes.setChecked(True)
        self.ck_coord = QCheckBox('坐标刻度')
        self.ck_coord.setChecked(True)
        self.ck_legend = QCheckBox('用量清单')
        self.ck_legend.setChecked(True)
        r6 = QHBoxLayout()
        r6.setSpacing(14)
        r6.addWidget(self.ck_codes)
        r6.addWidget(self.ck_coord)
        r6.addWidget(self.ck_legend)
        r6.addStretch(1)
        f3.addRow('', r6)
        self.adv_rows.append((f3, f3.rowCount() - 1))
        self.ck_over = QCheckBox('叠加原图对照')
        self.ck_over.setChecked(False)     # 默认关
        self.ck_over.stateChanged.connect(lambda: self.refresh_preview())
        self.sl_over = QSlider(Qt.Orientation.Horizontal)
        self.sl_over.setRange(10, 90)
        self.sl_over.setValue(45)
        self.lb_over = QLabel('45%')
        self.lb_over.setMinimumWidth(42)
        self.sl_over.valueChanged.connect(self.on_over_changed)
        r7 = QHBoxLayout()
        r7.addWidget(self.ck_over)
        r7.addWidget(self.sl_over)
        r7.addWidget(self.lb_over)
        f3.addRow(QLabel('参考图'), r7)
        self.adv_rows.append((f3, f3.rowCount() - 1))
        self.le_title = QLineEdit()
        f3.addRow(QLabel('标题'), self.le_title)
        self.adv_rows.append((f3, f3.rowCount() - 1))
        lv.addWidget(g3)

        self.b_gen = self.b_gen_top     # 生成按钮在顶部（别名，兼容旧脚本/自检）
        self.pb = QProgressBar()        # 进度条（平时隐藏，不占地方）
        self.pb.setRange(0, 0)
        self.pb.setVisible(False)
        lv.addWidget(self.pb)
        _st = self.load_settings()
        if _st.get('ui_mode') == 'advanced':
            self.rb_ui_adv.setChecked(True)
        self.apply_mode()
        for _g in self.findChildren(QGroupBox):     # 分组框内边距/行距收紧（高级档要塞进一屏）
            _l = _g.layout()
            if _l is not None:
                _l.setContentsMargins(6, 3, 6, 3)
                if hasattr(_l, 'setVerticalSpacing'):
                    _l.setVerticalSpacing(2)
                    _l.setHorizontalSpacing(8)
        lv.addStretch(1)
        sa = QScrollArea()              # 只保留一层滚动区（原来套了两层，内层还限宽 360，
        sa.setWidget(left)              # 结果内容 1300px 高却没有滚动条、右边全被裁掉）
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setMinimumWidth(430)
        sa.setMaximumWidth(620)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sp.addWidget(sa)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setSizes([520, 920])

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.lb_info = QLabel('把图片拖进窗口 → 调参数 → 点「生成图纸」')
        self.lb_info.setStyleSheet('color:#555')
        for t, fn in (('适应窗口', lambda: self.pv.fit()), ('100%', lambda: self.pv.actual())):
            b = QToolButton()
            b.setText(t)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.insertWidget(0, self.lb_info, 1)
        rv.addLayout(bar)
        self.pv = Preview()
        self.pv.cellHover.connect(self.on_cell_hover)
        rv.addWidget(self.pv, 1)
        sp.addWidget(right)
        self.statusBar().showMessage('就绪 ｜ 内置 %d 套色卡（默认 %s），离线可用' % (len(B.palette_labels()), B.PALETTE_DEFAULT))

    # ------------------------------------------------------------- 图片列表
    def _settings_path(self):
        return os.path.join(app_dir(), 'cerabeads_settings.json')

    def load_settings(self):
        try:
            import json
            with open(self._settings_path(), encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return {}

    def save_settings(self):
        try:
            import json
            with open(self._settings_path(), 'w', encoding='utf-8') as fh:
                json.dump({'ui_mode': 'advanced' if self.rb_ui_adv.isChecked() else 'basic'},
                          fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def apply_mode(self, save=False):
        """普通档：只留优化模式/格数/色卡/每格像素；高级档：全部参数。"""
        basic = self.rb_ui_basic.isChecked()
        for form, idx in self.adv_rows:
            try:
                form.setRowVisible(idx, not basic)
            except Exception:
                pass
        for form, idx in self.bs_rows:
            try:
                form.setRowVisible(idx, basic)
            except Exception:
                pass
        for w in self.adv_w:
            w.setVisible(not basic)
        for w in self.basic_w:
            w.setVisible(basic)
        self.g3.setVisible(not basic)          # 普通档没有它的项目，整框一起收起来
        # 普通档：顶部给大按钮，底部那排仍可用；状态行不塞诊断信息
        self.b_gen_top.setVisible(True)     # 生成 + 导出都在顶部，两种档位都不用滚
        self.b_batch.setVisible(True)
        try:
            self.centralWidget().layout().activate()
            self.centralWidget().updateGeometry()
        except Exception:
            pass
        self.lb_ui_hint.setVisible(True)
        if basic:
            self.lb_ui_hint.setText(
                '三步：① 拖入图片（拖到窗口任何地方都行）'
                '② 选优化模式（照片→图片优化，动漫/线稿→动漫优化，拿不准→通用）'
                '③ 点「生成图纸」。\n下面只需要调三样：格数（越大越清晰）、最大格数上限（想画大图就调大）和色卡。'
                '想要更多控制（打印格子大小、限色、去背景、噪点清理…）就切到「高级设置」。')
        else:
            self.lb_ui_hint.setVisible(False)      # 高级档不占这行（本来也没多少信息量）
        if save:
            self.save_settings()
        self._fit_window()                     # 换档位后窗口重新按内容自适应
        try:
            self.on_gen_done_visibility_refresh()
        except Exception:
            pass

    def on_gen_done_visibility_refresh(self):
        """档位切换后按当前尺寸重新适应预览（不影响导出）。"""
        if getattr(self, 'pattern', None):
            self.pv.fit()
        if getattr(self, 'pix', None):
            self.pv.fit()

    def _install_drops(self):
        """让窗口里任何地方都能接住拖入的图片（按钮、滑块、预览、列表、空白处都行）。"""
        n = 0
        targets = [self, self.centralWidget()] + list(self.findChildren(QWidget))
        for w in targets:
            if w is None:
                continue
            try:
                w.setAcceptDrops(True)
                w.installEventFilter(self)
                n += 1
            except Exception:
                pass
        self._drop_targets = n
        return n

    @staticmethod
    def _accept_drag(ev):
        """三个拖放事件（DragEnter / DragMove / Drop）都要「接受」，少一个 Qt 就会判定这里不能放。"""
        try:
            if ev.mimeData().hasUrls():
                ev.setDropAction(Qt.DropAction.CopyAction)
                ev.accept()
                return True
        except Exception:
            pass
        return False

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if self._accept_drag(ev):
                return True                      # 拦下来自己处理（否则子控件的默认实现会 ignore）
        elif t == QEvent.Type.Drop:
            if self._accept_drag(ev):
                self._take_urls(ev.mimeData().urls())
                return True
        return super().eventFilter(obj, ev)

    # 下面三个是「窗口自己」的兜底实现（有些事件不走过滤器时也能收）
    def dragEnterEvent(self, e):
        self._accept_drag(e)

    def dragMoveEvent(self, e):
        self._accept_drag(e)

    def dropEvent(self, e):
        if self._accept_drag(e):
            self._take_urls(e.mimeData().urls())

    def _take_urls(self, urls):
        """真正把拖进来的路径收进列表（文件、文件夹都交给 add_image 判；文件夹会自动挑里面的图片）。"""
        before = self.lst.count()
        for u in urls:
            p = u.toLocalFile()
            if p:
                self.add_image(p)
        got = self.lst.count() - before
        if got:
            self.lst.setCurrentRow(self.lst.count() - 1)
            self.statusBar().showMessage('拖入 %d 个（列表共 %d 个）' % (got, self.lst.count()))
        else:
            self.statusBar().showMessage('拖进来的东西里没有能用的图片')

    def pick_images(self):
        fs, _ = QFileDialog.getOpenFileNames(self, '选择图片', '',
                                            '图片 (*.png *.jpg *.jpeg *.jpe *.jfif *.bmp *.gif *.webp '
                                            '*.tif *.tiff *.avif *.heic *.heif *.hif *.ico *.jp2);;所有文件 (*)')
        for p in fs:
            self.add_image(p)
        if fs:
            self.lst.setCurrentRow(self.lst.count() - 1)

    IMG_EXT = B.IMG_EXT

    @staticmethod
    def _can_open(path):
        """真能读出来才算图片（扩展名不认识时也给它一次机会：HEIC/AVIF 之类）"""
        try:
            from PIL import Image as _I
            with _I.open(path) as im:
                im.convert('RGB')
            return True
        except Exception:
            return False

    def add_image(self, path):
        """只收「能打开的图片文件」：目录 / 非图片 / 不存在 都挡在外面并说清原因。"""
        path = os.path.abspath(path)
        if os.path.isdir(path):
            QMessageBox.information(self, '这是文件夹', '「%s」是文件夹，不是图片。\n'
                                                     '拖文件夹进来时我会自动挑里面的图片。' % os.path.basename(path))
            for f in sorted(os.listdir(path)):
                self.add_image(os.path.join(path, f))      # 子文件夹也递归进去
            if self.lst.count() and self.lst.currentRow() < 0:
                self.lst.setCurrentRow(0)
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, '找不到文件', '这个路径不存在：\n%s' % path)
            return
        if os.path.splitext(path)[1].lower() not in self.IMG_EXT and not self._can_open(path):
            QMessageBox.warning(self, '不是图片', '只支持这些格式：\n%s\n\n你给的是：%s'
                                % (' '.join(self.IMG_EXT), os.path.basename(path)))
            return
        for i in range(self.lst.count()):
            if self.lst.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return
        it = QListWidgetItem(os.path.basename(path))
        it.setData(Qt.ItemDataRole.UserRole, path)
        it.setToolTip(path)
        self.lst.addItem(it)

    def remove_sel(self):
        r = self.lst.currentRow()
        if r >= 0:
            self.lst.takeItem(r)
        if self.lst.count() == 0:
            self._reset_canvas()

    def _reset_canvas(self):
        """清空图片列表后，预览/状态/撤销栈一起复位（免得画布还挂着上一张图）。"""
        self.pattern = None
        self.src_path = ''
        self.src_img = None
        self.pix = None
        self.lb_info.setText('把图片拖进上面的框，或点「添加…」')
        try:
            from PIL import Image as _I
            self.pv.set_image(_I.new('RGB', (600, 400), (252, 252, 252)))
        except Exception:
            pass

    def clear_imgs(self):
        self.lst.clear()
        self._reset_canvas()

    def load_image(self, path):
        from PIL import Image
        try:
            im = Image.open(path)
            im.load()
        except Exception as e:
            QMessageBox.warning(self, '打不开', '读不了这个图片：\n%s' % e)
            for i in range(self.lst.count()):          # 打不开就把这条摘掉，别留个点不动的条目
                if self.lst.item(i).data(Qt.ItemDataRole.UserRole) == path:
                    self.lst.takeItem(i)
                    break
            if self.lst.count() == 0:
                self._reset_canvas()
            return
        self.src_img = im
        self.src_path = path
        if not self.le_title.text().strip():
            self.le_title.setText(os.path.splitext(os.path.basename(path))[0])
        # 默认把格数调到「每格约 9 个原图像素」这个最佳区间（不够再手动加；上限受「最大格数上限」约束）
        try:
            cap = int(self.sp_max.value())
            gw0 = max(8, min(cap, int(round(im.width / 9.0))))
            gh0 = max(8, int(round(gw0 * im.height / max(1, im.width))))
            if gh0 > cap:
                gh0 = cap
                gw0 = max(8, min(cap, int(round(gh0 * im.width / max(1, im.height)))))
            if gw0 != self.sp_w.value() or gh0 != self.sp_h.value():
                self.sp_w.setValue(gw0)
                self.sp_h.setValue(gh0)
            self.auto_grid_note = '已按图片尺寸默认 %d×%d 格（每格约 9 个原图像素，可手动改）' % (gw0, gh0)
        except Exception:
            pass
        if self.ck_ratio.isChecked() and im.width:
            self.sp_h.setValue(max(8, min(400, round(self.sp_w.value() * im.height / im.width))))
        self.lb_info.setText('%s ｜ %d×%d 像素' % (os.path.basename(path), im.width, im.height))
        if self.auto_run:
            self.generate()

    # ------------------------------------------------------------- 参数联动
    def on_max_changed(self, v):
        """最大格数上限：调大后宽/高才能填更大的数"""
        for sp in (self.sp_w, self.sp_h):
            sp.setRange(8, v)
            if sp.value() > v:
                sp.setValue(v)
        for sl in (self.sl_w, self.sl_h):
            sl.setRange(8, min(v, 999))
            if sl.value() > sl.maximum():
                sl.setValue(sl.maximum())

    def on_w_changed(self, v):
        if self.ck_ratio.isChecked() and getattr(self, 'src_img', None):
            self.sp_h.setValue(max(8, min(400, round(v * self.src_img.height / self.src_img.width))))

    def on_limit_changed(self):
        on = self.ck_limit.isChecked()
        self.sl_maxc.setEnabled(on)
        self.lb_maxc.setEnabled(on)

    def on_cell_changed(self, v):
        self.lb_cell.setText('%d px' % v)
        self.preview_cell = v                 # 预览也跟着「每格像素」变（原来是滑杆只改标签，所以看着没反应）
        self._cell_timer.start(160)           # 拖滑杆时防抖，松手后再重渲染

    def _cell_apply(self):
        if getattr(self, 'pattern', None):
            self.refresh_preview()

    def on_over_changed(self, v):
        self.lb_over.setText('%d%%' % v)
        if self.ck_over.isChecked():
            self.refresh_preview()

    def sample_mode(self):
        i = self.cb_sample.currentIndex()
        return (None, 'photo', 'mean', 'dominant')[i] if 0 <= i <= 3 else None

    def preset_mode(self):
        for i, rb in enumerate(getattr(self, 'rb_pre', [])):
            if rb.isChecked():
                return B.PRESETS[i][0]
        return 'general'

    def on_preset_changed(self):
        if getattr(self, 'auto_run', False):
            self.generate()

    def despeckle_min(self):
        return (self.sp_desp.value() if self.ck_desp.isChecked() else 0)

    def opts(self):
        return dict(gw=self.sp_w.value(), gh=self.sp_h.value(), palette=self.cb_pal.currentText(),
                    max_colors=(self.sl_maxc.value() if self.ck_limit.isChecked() else None),
                    remove_bg=self.ck_bg.isChecked(), bg_tol=self.sl_bg.value(),
                    sample=self.sample_mode(), despeckle_min=self.despeckle_min(),
                    harmonize_on=self.ck_harm.isChecked(), preset=self.preset_mode(),
                    upscale=self.ck_up.isChecked())

    # ------------------------------------------------------------- 生成
    def generate(self):
        if not self.src_path:
            QMessageBox.information(self, '先选图片', '先把图片拖进来或点「添加…」。')
            return
        self.b_gen.setEnabled(False)
        if getattr(self, 'worker', None) is not None and self.worker.isRunning():
            return                     # 上一轮还没算完，别叠着跑（Qt 里叠跑会 abort）
        self.pb.setVisible(True)
        self.lb_info.setText('正在计算…')
        self.worker = GenWorker(self.src_path, self.opts())
        self.worker.done.connect(self.on_gen_done)
        self.worker.start()

    def on_gen_done(self, pat, sec, err):
        self.pb.setVisible(False)
        self.b_gen.setEnabled(True)
        if err or pat is None:
            QMessageBox.critical(self, '出错了', (err or '未知错误')[-1500:])
            self.lb_info.setText('生成失败')
            return
        self.pattern = pat
        self.refresh_preview()
        self.lb_info.setText('用色 %d 种 ｜ 共 %d 颗 ｜ %d×%d 格 ｜ 取色 %.2fs%s'
                             % (len(pat['counts']), pat['total'], pat['gw'], pat['gh'], sec,
                                '（%d 色已并入最近色）' % pat['dropped'] if pat['dropped'] else ''))
        self._density_hint(pat)
        self._diag_hint(pat)

    def refresh_preview(self):
        if not self.pattern:
            return
        pat = self.pattern
        # 预览默认就是「每格像素」本身；只有本机扛不住时，才按实测内存/核数降（并说明降了多少）
        cell, self.pv_info = B.preview_cell_cap(pat['gw'], pat['gh'], self.preview_cell)
        self.cur_cell = cell
        kw = dict(cell=cell, red_every=self.sp_red.value(),
                  title=self.le_title.text().strip() or None,
                  show_codes=self.ck_codes.isChecked(), show_coords=self.ck_coord.isChecked(),
                  show_legend=self.ck_legend.isChecked())
        if self.ck_isle.isChecked():               # 只画洋红细框，不改色块
            try:
                isl = B.islands(pat['idx'], 2)
                from PIL import ImageDraw as _ID
                _d = _ID.Draw(im)
                _L = B.layout(pat, cell=kw['cell'], red_every=kw['red_every'],
                              show_coords=kw['show_coords'], show_legend=kw['show_legend'])
                for _r in range(isl.shape[0]):
                    for _c in range(isl.shape[1]):
                        if isl[_r, _c]:
                            _x = _L['x0'] + _c * kw['cell']
                            _y = _L['y0'] + _r * kw['cell']
                            _d.rectangle([_x + 1, _y + 1, _x + kw['cell'] - 2, _y + kw['cell'] - 2],
                                         outline=(230, 0, 160), width=max(2, int(kw['cell'] * 0.12)))
                self.n_island = int(isl.sum())
            except Exception:
                self.n_island = -1
        lkw = dict(cell=kw['cell'], red_every=kw['red_every'], title=kw['title'],
                   show_coords=kw['show_coords'], show_legend=kw['show_legend'])
        im = B.render(pat, **kw)
        if self.ck_over.isChecked() and getattr(self, 'src_img', None):
            from PIL import Image
            L = B.layout(pat, **lkw)
            ref = self.src_img.convert('RGB').resize(
                (L['grid_w'] + L['x0'], L['grid_h'] + L['y0']))
            a = self.sl_over.value() / 100.0
            base = im.copy()
            base.paste(Image.blend(base.crop((0, 0, ref.width, ref.height)), ref, a), (0, 0))
            im = base
        L = B.layout(pat, **lkw)          # 点击定位用（与 render 同算法）
        L['gw'], L['gh'] = pat['gw'], pat['gh']
        self.prev_img = im
        self.pv.set_image(im, layout=L)
        _i = getattr(self, 'pv_info', {}) or {}
        if _i.get('capped'):
            self.statusBar().showMessage(
                '本机可用内存 %.1f GB / %d 核 → 预览按 %d px/格 画（%.0f 百万像素；'
                '按你设的 %d px/格 是 %.0f 百万像素，会超出本机预算 %.0f 百万像素）。'
                '导出仍是全分辨率的 %d px/格。%s'
                % (_i['avail_mb'] / 1024.0, _i['cores'], _i['cell'], _i['cells'] * _i['cell'] ** 2 / 1e6,
                   _i['wanted'], _i['need_mp'], _i['budget_mp'], self.preview_cell,
                   '（格数太多才会这样；想看清就减少格数，或换台内存更大的机器）'
                   if _i['cell'] * 2 < _i['wanted'] else ''))
        elif _i:
            self.statusBar().showMessage(
                '本机可用内存 %.1f GB / %d 核 → 预览按你设的 %d px/格 全分辨率渲染'
                % (_i['avail_mb'] / 1024.0, _i['cores'], self.preview_cell))


    def on_cell_hover(self, r, c):
        pat = self.pattern
        if not pat:
            return
        i = int(pat['idx'][r, c])
        m = pat['matcher']
        if i < 0:
            self.statusBar().showMessage('第 %d 行 第 %d 列：空' % (r + 1, c + 1))
        else:
            self.statusBar().showMessage('第 %d 行 第 %d 列：%s ｜ %s' %
                                         (r + 1, c + 1, m.codes[i], m.hexes[i]))

    # ------------------------------------------------------------- 导出
    def closeEvent(self, e):
        """关窗时把还在跑的生成线程收干净，免得 Qt 里线程还活着就被销毁（会 abort）。"""
        wk = getattr(self, 'worker', None)
        if wk is not None and wk.isRunning():
            try:
                wk.requestInterruption()
                if not wk.wait(3000):
                    wk.terminate()
                    wk.wait(1000)
            except Exception:
                pass
        e.accept()

    def base_name(self):
        t = self.le_title.text().strip()
        if t:
            return t
        return os.path.splitext(os.path.basename(self.src_path))[0] or 'CeraBeads图纸'

    def _density_hint(self, pat):
        """显示“每格平均覆盖多少原图像素”：<16 提示分辨率不够（取色会抖）。"""
        ppc = pat.get('px_per_cell') or 0.0
        w0, h0 = pat.get('src_wh') or (0, 0)
        self.lb_info.setText(self.lb_info.text().split('（原图')[0])
        tip = ''
        if ppc and ppc < 16:
            tip = ('（原图 %d×%d，每格只覆盖 %.1f 个像素，颜色不够稳——建议减少格数或先把图片放大）'
                   % (w0, h0, ppc))
            self.lb_info.setStyleSheet('color:#b8860b')
        else:
            self.lb_info.setStyleSheet('color:#555')
            if ppc:
                tip = '（原图 %d×%d，每格 %.0f 像素）' % (w0, h0, ppc)
        if tip:
            self.lb_info.setText(self.lb_info.text() + ' ' + tip)

    def _diag_hint(self, pat):
        """线宽预警 / 色号预算 / 孤岛：一句话说清“现在这套参数会不会丢细节”。
        普通档不显示（小白不需要），切到高级档才报。"""
        if self.rb_ui_basic.isChecked():
            return
        bits = []
        try:
            src = getattr(self, 'src_img', None)
            if src is not None:
                band, q25 = B.line_width_px(np.asarray(src.convert('RGB')))
                ppc = float(src.width) * float(src.height) / float(pat['gw'] * pat['gh'])
                if band:
                    need = band * 3.0
                    if ppc < band:
                        bits.append('⚠️ 最细线过渡带约 %.1f px，每格只有 %.1f 个原图像素 → 细线会被抹掉，'
                                    '建议提升格数/换动漫优化' % (band, ppc))
                    elif ppc < need:
                        bits.append('最细线过渡带约 %.1f px，每格 %.1f 个原图像素（建议 ≥ %.0f 才稳）'
                                    % (band, ppc, need))
                    else:
                        bits.append('线宽 %.1f px ／ 每格 %.1f 个原图像素（够用）' % (band, ppc))
        except Exception:
            pass
        try:
            if pat.get('upscaled', 1) > 1:
                bits.append('已把小图放大 %d× 再取色' % pat['upscaled'])
        except Exception:
            pass
        try:
            k, cov = B.suggest_max_colors(pat['counts'])
            bits.append('色号：现用 %d 种；按 98%% 用量推荐上限 ≈ %d 种' % (len(pat['counts']), k))
        except Exception:
            pass
        try:
            n_isl = getattr(self, 'n_island', 0)
            if n_isl > 0:
                bits.append('孤岛 %d 处（≤2 格同色小块，可勾「标出孤岛」看位置，用「清理碎点」清掉）' % n_isl)
        except Exception:
            pass
        if bits:
            self.lb_info.setText(self.lb_info.text() + ' ｜ ' + ' ｜ '.join(bits))

    def _out_mp(self, cell):
        """整张图大约多少百万像素（内存保护用）。"""
        pat = self.pattern
        if not pat:
            return 0.0
        return pat['gw'] * pat['gh'] * cell * cell / 1e6

    def _out_side(self, cell, extra=0):
        pat = self.pattern
        if not pat:
            return 0
        return max(pat['gw'], pat['gh']) * cell + extra

    def _huge(self, cell):
        """输出是否“超大”：超大不再拦截、也不缩图，改用流式写 PNG（照样全尺寸成功）。"""
        return self._out_mp(cell) > HUGE_MP

    def _pattern_for(self, path):
        """按当前参数给指定图片出图纸（批量导出用，一张一张来，不动预览）"""
        from PIL import Image as _Im
        im = _Im.open(path)
        gw = self.sp_w.value()
        gh = self.sp_h.value() if not self.ck_ratio.isChecked() else max(
            8, min(self.sp_max.value(), round(gw * im.height / im.width)))
        return B.build_pattern(path, gw=gw, gh=gh, palette=self.cb_pal.currentText(),
                               max_colors=(self.sl_maxc.value() if self.ck_limit.isChecked() else None),
                               remove_bg=self.ck_bg.isChecked(), bg_tol=self.sl_bg.value(),
                               sample=self.sample_mode(), despeckle_min=self.despeckle_min(),
                               harmonize_on=self.ck_harm.isChecked(), preset=self.preset_mode(),
                               upscale=self.ck_up.isChecked())

    def _export_one(self, pat, name, title, outdir, want):
        """把一张图纸按勾选的成品导出；返回 (文件列表, 说明列表)"""
        cell = self.sl_cell.value()
        red = self.sp_red.value()
        common = dict(red_every=red, show_codes=self.ck_codes.isChecked(),
                      show_coords=self.ck_coord.isChecked(), show_legend=self.ck_legend.isChecked())
        made, notes = [], []
        if 'full' in want:
            p = os.path.join(outdir, name + '_图纸.png')
            if self._huge(cell):
                B.save_png_stream(pat, p, cell=cell, red_every=red,
                                  show_codes=self.ck_codes.isChecked())
                notes.append('「%s」图纸按 %.0f 百万像素流式写出（全尺寸）' % (name, self._out_mp(cell)))
            else:
                B.render(pat, cell=cell, title=title, **common).save(p)
            made.append(p)
        if 'plain' in want:
            p = os.path.join(outdir, name + '_纯净.png')
            if self._huge(cell):
                B.save_png_stream(pat, p, cell=cell, red_every=red, show_codes=False)
            else:
                B.render(pat, cell=cell, title=title, show_codes=False, show_coords=False,
                         show_legend=False).save(p)
            made.append(p)
        if 'multi' in want:
            for px in (20, 32, 48):
                if self._huge(px):
                    notes.append('「%s」%dpx 版跳过（会到 %.0f 百万像素）'
                                 % (name, px, self._out_mp(px)))
                    continue
                p = os.path.join(outdir, '%s_图纸_%dpx.png' % (name, px))
                B.render(pat, cell=px, title=title, **common).save(p)
                made.append(p)
        if 'jpg' in want:
            p = os.path.join(outdir, name + '_分享.jpg')
            jc = cell
            if self._out_side(cell) > JPG_MAX_SIDE:
                jc = max(4, int(JPG_MAX_SIDE / max(pat['gw'], pat['gh'])))
                notes.append('「%s」分享 JPG 按 %dpx/格 出（JPEG 单边上限 %d 像素）'
                             % (name, jc, JPG_MAX_SIDE))
            B.save_jpg(pat, p, cell=jc, title=title, show_codes=True, show_coords=False,
                       show_legend=True, red_every=red)
            made.append(p)
        if 'pdf' in want:
            p = os.path.join(outdir, name + '_打印.pdf')
            B.save_pdf(pat, p, cell=cell, title=title, tile=(29, 29), **common)
            made.append(p)
        if 'csv' in want:
            p = os.path.join(outdir, name + '_用量.csv')
            B.save_csv(pat, p)
            made.append(p)
        return made, notes

    def batch_export(self):
        """一个按钮干完：列表里每张图都按当前参数出图并导出（只拖了一张时就是导出这一张）。
        先勾要哪些成品，再选输出文件夹。"""
        n = self.lst.count()
        if n == 0 and not self.pattern:
            QMessageBox.information(self, '没有图', '先把图片拖进来，或点「生成图纸」先出一张。')
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        want = dlg.chosen()
        if not want:
            QMessageBox.information(self, '没选', '至少勾一样。')
            return
        outdir = QFileDialog.getExistingDirectory(self, '选一个文件夹放图', app_dir())
        if not outdir:
            return
        if n == 0:                      # 列表空了：退回「只导出预览这张」
            return self._batch_current()
        title = self.le_title.text().strip() or None
        made_all, notes_all, errs = [], [], []
        self.pb.setVisible(True)
        try:
            for k in range(n):
                path = self.lst.item(k).data(Qt.ItemDataRole.UserRole)
                nm = os.path.splitext(os.path.basename(path))[0] or 'CeraBeads图纸'
                self.lb_info.setText('出图中 %d/%d：%s' % (k + 1, n, os.path.basename(path)))
                QApplication.processEvents()
                try:
                    pat = self._pattern_for(path)
                    m1, nt = self._export_one(pat, nm, title or nm, outdir, want)
                    made_all += m1
                    notes_all += nt
                except Exception as e:
                    errs.append('%s：%s' % (os.path.basename(path), e))
        finally:
            self.pb.setVisible(False)
        if not made_all and errs:
            QMessageBox.critical(self, '导出失败', '\n'.join(errs[:10]))
            return
        self.statusBar().showMessage('已导出 %d 个文件到 %s' % (len(made_all), outdir))
        tot = sum(os.path.getsize(p) for p in made_all if os.path.exists(p))
        head = '\n'.join(os.path.basename(p) for p in made_all[:12])
        if len(made_all) > 12:
            head += '\n…还有 %d 个' % (len(made_all) - 12)
        msg = ('已导出 %d 个文件（%.1f MB）\n图片 %d 张，输出到：\n%s\n\n%s'
               % (len(made_all), tot / 1e6, n, outdir, head))
        if notes_all:
            msg += '\n\n说明：\n' + '\n'.join(notes_all[:6])
        if errs:
            msg += '\n\n失败 %d 张：\n%s' % (len(errs), '\n'.join(errs[:5]))
        QMessageBox.information(self, '导出完成', msg)

    def _batch_current(self):
        if not self.pattern:
            QMessageBox.information(self, '先出图', '请先点「生成图纸」。')
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        want = dlg.chosen()
        if not want:
            QMessageBox.information(self, '没选', '至少勾一样。')
            return
        outdir = QFileDialog.getExistingDirectory(self, '选一个文件夹放图', app_dir())
        if not outdir:
            return
        pat = self.pattern
        name = self.base_name()
        title = self.le_title.text().strip() or None
        cell = self.sl_cell.value()
        common = dict(red_every=self.sp_red.value(), show_codes=self.ck_codes.isChecked(),
                      show_coords=self.ck_coord.isChecked(), show_legend=self.ck_legend.isChecked())
        made = []
        try:
            notes = []
            self.statusBar().showMessage('正在导出…')
            QApplication.processEvents()
            if 'full' in want:
                p = os.path.join(outdir, name + '_图纸.png')
                if self._huge(cell):          # 超大：流式写，全尺寸、不缩图
                    B.save_png_stream(pat, p, cell=cell, red_every=self.sp_red.value(),
                                      show_codes=self.ck_codes.isChecked())
                    notes.append('图纸按 %.0f 百万像素流式写出（全尺寸）' % self._out_mp(cell))
                else:
                    B.render(pat, cell=cell, title=title, **common).save(p)
                made.append(p)
            if 'plain' in want:
                p = os.path.join(outdir, name + '_纯净.png')
                if self._huge(cell):
                    B.save_png_stream(pat, p, cell=cell, red_every=self.sp_red.value(),
                                      show_codes=False)
                else:
                    B.render(pat, cell=cell, title=title, show_codes=False, show_coords=False,
                             show_legend=False).save(p)
                made.append(p)
            if 'multi' in want:
                for px in (20, 32, 48):
                    if self._huge(px):
                        notes.append('%dpx 版跳过（会到 %.0f 百万像素）' % (px, self._out_mp(px)))
                        continue
                    p = os.path.join(outdir, '%s_图纸_%dpx.png' % (name, px))
                    B.render(pat, cell=px, title=title, **common).save(p)
                    made.append(p)
            if 'jpg' in want:
                p = os.path.join(outdir, name + '_分享.jpg')
                jc = cell
                if self._out_side(cell) > JPG_MAX_SIDE:      # JPEG 编码器硬上限，只此一项封顶
                    jc = max(4, int(JPG_MAX_SIDE / max(pat['gw'], pat['gh'])))
                    notes.append('分享 JPG 按 %dpx/格 出（JPEG 单边上限 %d 像素）' % (jc, JPG_MAX_SIDE))
                B.save_jpg(pat, p, cell=jc, title=title, show_codes=True,
                           show_coords=False, show_legend=True, red_every=self.sp_red.value())
                made.append(p)
            if 'pdf' in want:
                p = os.path.join(outdir, name + '_打印.pdf')
                B.save_pdf(pat, p, cell=cell, title=title, tile=(29, 29), **common)
                made.append(p)
            if 'csv' in want:
                p = os.path.join(outdir, name + '_用量.csv')
                B.save_csv(pat, p)
                made.append(p)
        except Exception as e:
            QMessageBox.critical(self, '导出失败', '%s\n\n%s' % (e, traceback.format_exc()[-1200:]))
            return
        tot = sum(os.path.getsize(p) for p in made)
        self.statusBar().showMessage('已导出 %d 个文件到 %s' % (len(made), outdir))
        _nt = ('\n\n说明：\n' + '\n'.join(notes)) if notes else ''
        QMessageBox.information(self, '导出完成',
                                '已导出 %d 个文件（共 %.1f MB）：\n\n%s\n\n到：%s%s'
                                % (len(made), tot / 1e6, '\n'.join(os.path.basename(p) for p in made),
                                   outdir, _nt))

    def batch_all_sources(self):
        """兼容旧入口：与「批量导出」是同一个动作"""
        return self.batch_export()
        if self.lst.count() == 0:
            QMessageBox.information(self, '没有图', '先把图片拖进来。')
            return
        outdir = QFileDialog.getExistingDirectory(self, '选一个文件夹放图', app_dir())
        if not outdir:
            return
        self.pb.setVisible(True)
        logs = []
        n = self.lst.count()
        for k in range(n):
            path = self.lst.item(k).data(Qt.ItemDataRole.UserRole)
            try:
                self.lb_info.setText('批量出图中 %d/%d：%s' % (k + 1, n, os.path.basename(path)))
                QApplication.processEvents()
                from PIL import Image
                im = Image.open(path)
                gw = self.sp_w.value()
                gh = self.sp_h.value() if not self.ck_ratio.isChecked() else max(
                    8, min(400, round(gw * im.height / im.width)))
                pat = B.build_pattern(path, gw=gw, gh=gh, palette=self.cb_pal.currentText(),
                                      max_colors=(self.sl_maxc.value() if self.ck_limit.isChecked() else None),
                                      remove_bg=self.ck_bg.isChecked(), bg_tol=self.sl_bg.value(),
                                      sample=self.sample_mode(), despeckle_min=self.despeckle_min(),
                                      harmonize_on=self.ck_harm.isChecked(),
                                      preset=self.preset_mode(),
                                      upscale=self.ck_up.isChecked())
                name = os.path.splitext(os.path.basename(path))[0]
                title = name
                cell = self.sl_cell.value()
                common = dict(red_every=self.sp_red.value(), show_codes=self.ck_codes.isChecked(),
                              show_coords=self.ck_coord.isChecked(), show_legend=self.ck_legend.isChecked())
                _p = os.path.join(outdir, name + '_图纸.png')
                if self._huge(cell):
                    B.save_png_stream(pat, _p, cell=cell, red_every=self.sp_red.value(),
                                      show_codes=self.ck_codes.isChecked())
                else:
                    B.render(pat, cell=cell, title=title, **common).save(_p)
                B.save_pdf(pat, os.path.join(outdir, name + '_打印.pdf'), cell=cell, title=title,
                           tile=(29, 29), **common)
                B.save_csv(pat, os.path.join(outdir, name + '_用量.csv'))
                logs.append('%s：%d 色 / %d 颗' % (name, len(pat['counts']), pat['total']))
            except Exception as e:
                logs.append('%s：失败（%s）' % (os.path.basename(path), e))
        self.pb.setVisible(False)
        self.statusBar().showMessage('批量出图完成，共 %d 张' % n)
        QMessageBox.information(self, '批量出图完成',
                                '共 %d 张，输出到：\n%s\n\n%s' % (n, outdir, '\n'.join(logs[:30])))


# ------------------------------------------------------------------ 自检
def selftest():
    from PIL import Image
    import numpy as np
    import tempfile
    out = ['%s %s' % (APP_TITLE, APP_VERSION),
           'python=%s numpy=%s PIL=%s' % (sys.version.split()[0], np.__version__, Image.__version__),
           B.selftest(None)]
    d = tempfile.mkdtemp(prefix='cerabeads_st_')
    a = np.full((220, 280, 3), 250, dtype=np.uint8)
    a[30:130, 20:130] = (205, 70, 60)
    a[60:180, 150:260] = (40, 90, 190)
    a[140:200, 60:210] = (245, 195, 70)
    ip = os.path.join(d, 'st.png')
    Image.fromarray(a).save(ip)
    app = QApplication.instance() or QApplication(sys.argv)
    # 自检/打包版自检必须无人值守：把模态弹窗变成 no-op，否则一处弹窗就会永久卡住
    try:
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        QMessageBox.question = staticmethod(lambda *a, **k: None)
    except Exception:
        pass
    w = MainWindow()
    DEFAULTS0 = (w.ck_limit.isChecked(), w.ck_bg.isChecked(), w.ck_desp.isChecked(),
                 w.ck_over.isChecked())        # 开机默认（后面的测试会改它们，先存下来）
    DEFAULTS1 = (w.sp_max.value(), w.cb_sample.currentIndex(), w.preset_mode(),
                 len(w.rb_pre))
    _prof = B.machine_profile()
    out.append('本机：可用内存 %.1f GB / 共 %.1f GB / %d 核'
               % (_prof['avail_mb'] / 1024.0, _prof['total_mb'] / 1024.0, _prof['cores']))
    _c1, _i1 = B.preview_cell_cap(96, 103, 32, _prof)
    _c2, _i2 = B.preview_cell_cap(2000, 2000, 256, _prof)
    out.append('预览预算：96×103 格 @32px → %d px/格（%s）｜2000×2000 格 @256px → %d px/格（%s，'
               '预算 %.0f 百万像素）' % (_c1, '不降' if not _i1['capped'] else '降到 %d' % _c1,
                                     _c2, '降' if _i2['capped'] else '不降', _i2['budget_mp']))
    _cap_ok = (_i1['capped'] is False) and (_i2['capped'] is True) and _prof['avail_mb'] > 0
    out.append('本机评估 = %s' % ('OK' if _cap_ok else 'FAIL'))
    out.append('图标 = %s' % (icon_path() or '缺失'))
    w.add_image(ip)
    w.sp_w.setValue(48)
    w.sp_h.setValue(38)
    w.auto_run = False
    w.src_path = ip
    w.generate()
    w.worker.wait(120000)
    app.processEvents()
    pat = w.pattern
    out.append('界面生成 = %s' % ('OK' if pat else 'FAIL'))
    if pat:
        # 参考图叠加
        w.ck_over.setChecked(True)
        app.processEvents()
        out.append('参考图叠加 = OK')
        f = {}
        f['png'] = os.path.join(d, 'a.png')
        B.render(pat, cell=20, title='自检').save(f['png'])
        f['plain'] = os.path.join(d, 'b.png')
        B.render(pat, cell=20, title='自检', show_codes=False, show_legend=False,
                 show_coords=False).save(f['plain'])
        f['jpg'] = os.path.join(d, 'c.jpg')
        B.save_jpg(pat, f['jpg'], cell=20, title='自检')
        f['pdf'] = os.path.join(d, 'd.pdf')
        B.save_pdf(pat, f['pdf'], cell=20, title='自检', tile=(29, 29))
        f['csv'] = os.path.join(d, 'e.csv')
        B.save_csv(pat, f['csv'])
        f['pdf1'] = os.path.join(d, 'f_single.pdf')
        B.save_pdf(pat, f['pdf1'], cell=20, title='自检')
        out.append('用色 %d 种 ｜ 共 %d 颗 ｜ 空 %d 格'
                   % (len(pat['counts']), pat['total'], int(pat['empty'].sum())))
        out.append('导出大小：' + ' ｜ '.join('%s %d 字节' % (k, os.path.getsize(v))
                                            for k, v in sorted(f.items())))
        ok = all(os.path.getsize(f[k]) > 3000 for k in ('png', 'plain', 'jpg', 'pdf', 'pdf1')) \
             and os.path.getsize(f['csv']) > 50
        out.append('导出 = %s' % ('OK' if ok else 'FAIL'))
        out.append('色卡 %d 套 ｜ 预览 %dx%d' % (len(B.palette_labels()), w.pv.pix.width(),
                                               w.pv.pix.height()))
        # 最大格数上限 + 大图预览自动缩小
        w.sp_max.setValue(600)
        app.processEvents()
        cap = (w.sp_w.maximum() == 600 and w.sl_w.maximum() == 600)
        out.append('最大格数上限：设 600 → 宽/高可填到 %d（%s）' % (w.sp_w.maximum(),
                                                          'OK' if cap else 'FAIL'))
        w.sp_w.setValue(320)
        w.sp_h.setValue(300)
        w.auto_run = False
        w.generate()
        w.worker.wait(300000)
        app.processEvents()
        big = w.pattern
        big_ok = bool(big) and big['gw'] == 320 and big['gh'] == 300
        full = (w.cur_cell == w.preview_cell)      # 本机扛得住就不降分辨率
        out.append('大图 320×300 = %s ｜ 预览按 %d px/格（设的就是 %d；%s）｜本机预算 %.0f 百万像素'
                   % ('OK' if big_ok else 'FAIL', w.cur_cell, w.preview_cell,
                      '全分辨率不降' if full else '降到 %d' % w.cur_cell,
                      (getattr(w, 'pv_info', {}) or {}).get('budget_mp', 0)))
        out.append('超大图：320×300 按 200px/格 = %d 像素见方、%.0f 百万像素 → 走流式写（不拦、不缩）'
                   % (w._out_side(200), w._out_mp(200)))
        # 流式写 PNG：拿一张小尺寸验证「流式写出来的尺寸、像素都对」
        try:
            _sp = os.path.join(d, 'stream.png')
            B.save_png_stream(big, _sp, cell=8, red_every=10, show_codes=False)
            from PIL import Image as _I
            _im = _I.open(_sp)
            stream_ok = (_im.size == (big['gw'] * 8, big['gh'] * 8))
            stream_msg = '流式写 PNG：%dx%d（期望 %dx%d）' % (_im.size[0], _im.size[1],
                                                          big['gw'] * 8, big['gh'] * 8)
        except Exception as _e:
            stream_ok = False
            stream_msg = '流式写 PNG 失败：%s' % _e
        out.append('超大图不拦截 = %s ｜ %s' % ('OK' if stream_ok else 'FAIL', stream_msg))
        cap = cap and big_ok and full and stream_ok and w._huge(200)
        # 「每格像素」必须真的影响预览（曾经的 bug：滑杆只改标签）
        w.sl_cell.setValue(8)
        w._cell_apply()
        app.processEvents()
        c8 = (w.preview_cell, w.cur_cell, w.pv.pix.width())
        w.sl_cell.setValue(60)
        w._cell_apply()
        app.processEvents()
        c60 = (w.preview_cell, w.cur_cell, w.pv.pix.width())
        cell_ok = c8[0] == 8 and c60[0] == 60 and c60[2] > c8[2]
        out.append('每格像素生效：设 8 → %s ｜ 设 60 → %s（预览宽变大 = 生效）（%s）'
                   % (c8, c60, 'OK' if cell_ok else 'FAIL'))
        out.append('每格像素滑杆上限 = %d（%s）' % (w.sl_cell.maximum(),
                                                 'OK' if w.sl_cell.maximum() >= 200 else 'FAIL'))
        cap = cap and cell_ok and w.sl_cell.maximum() >= 200
        # 取色方式（平均色 / 主导色）+ 更低的每格像素
        _src = w.src_path
        p_mean = B.build_pattern(_src, gw=72, gh=72, max_colors=16, sample='mean')
        p_dom = B.build_pattern(_src, gw=72, gh=72, max_colors=16, sample='dominant')
        diff = int((p_mean['idx'] != p_dom['idx']).sum())
        out.append('取色方式：平均色 vs 主导色 有 %d 格不同（%s）｜平均色 用色 %d 种、主导色 %d 种'
                   % (diff, 'OK' if diff > 0 else 'FAIL', len(p_mean['counts']),
                      len(p_dom['counts'])))
        w.sl_cell.setValue(4)
        tiny = B.render(p_mean, cell=4)
        out.append('每格像素下限 4px：渲染 %dx%d（%s）' % (tiny.width, tiny.height,
                                                   'OK' if tiny.width > 300 else 'FAIL'))
        cap = cap and diff > 0 and tiny.width > 300
        # 优化模式：默认「通用」；三个模式结果必须各不相同（说明真的换了算法）
        out.append('优化模式默认 = %s（下拉/单选共 %d 项；最大格数上限默认 %d；取色默认第 %d 项=跟随）'
                   % (DEFAULTS1[2], DEFAULTS1[3], DEFAULTS1[0], DEFAULTS1[1]))
        pm = [B.build_pattern(w.src_path, gw=72, gh=72, max_colors=16, preset=k,
                              harmonize_on=False) for k, _ in B.PRESETS]
        d_pg = int((pm[0]['idx'] != pm[1]['idx']).sum())
        d_ga = int((pm[0]['idx'] != pm[2]['idx']).sum())
        pre_ok = d_pg > 0 and d_ga > 0 and DEFAULTS1[2] == 'general' and DEFAULTS1[0] == 104
        out.append('优化模式三档：通用vs图片 %d 格不同 ｜ 通用vs动漫 %d 格不同 ｜ 用色 %s（%s）'
                   % (d_pg, d_ga, [len(p['counts']) for p in pm], 'OK' if pre_ok else 'FAIL'))
        cap = cap and pre_ok
        # 导入图片时按尺寸给默认格数（每格约 9 个原图像素）
        w.load_image(w.src_path)
        g9 = (w.sp_w.value(), w.sp_h.value())
        g9_ok = 20 <= g9[0] <= 40 and 15 <= g9[1] <= 32
        out.append('导入自适应格数：280×220 图 → 默认 %dx%d 格（%s）' % (g9[0], g9[1],
                                                              'OK' if g9_ok else 'FAIL'))
        cap = cap and g9_ok
        # 碎点：造一个 2 格的小色块 + 一个 4 格的正常块，看会不会只清掉小的
        _idx = np.zeros((20, 20), dtype=np.int32)
        _idx[5, 5] = 1
        _idx[5, 6] = 1                      # 2 格（< 3）→ 应被并掉
        _idx[12:14, 12:14] = 2              # 4 格（>= 3）→ 应保留
        _out, _n = B.despeckle(_idx, min_cells=3)
        s_on = {'speck': _n}
        speck_ok = _n == 1 and (_out[_idx == 1] == 0).all() and (_out[_idx == 2] == 2).all()
        out.append('清理碎点：2 格小块→%s（并掉 %d 个）｜4 格块保留=%s（%s）'
                   % ('已并掉' if (_out[5, 5] == 0) else '未并掉', _n,
                      bool((_out[12:14, 12:14] == 2).all()), 'OK' if speck_ok else 'FAIL'))
        cap = cap and speck_ok and len(pm[0]['counts']) > 0
        # 默认开关（用户要求默认关）：限制颜色数 / 自动去掉背景 / 清理碎点 / 参考图对照
        defs = list(DEFAULTS0)
        def_ok = not any(defs)
        out.append('默认开关：限制颜色数=%s 去背景=%s 清碎点=%s 参考图=%s（%s）'
                   % tuple(['开' if d else '关' for d in defs] + ['OK' if def_ok else 'FAIL']))
        cap = cap and def_ok and _cap_ok
        # 照片优化档：下拉有 4 项，且与平均色结果不同（说明真的换了算法）
        p_ph = B.build_pattern(w.src_path, gw=72, gh=72, sample='photo')
        ph_diff = int((p_ph['idx'] != p_mean['idx']).sum())
        out.append('照片优化：下拉 %d 项 ｜ 与平均色有 %d 格不同（%s）｜用色 %d 种'
                   % (w.cb_sample.count(), ph_diff, 'OK' if ph_diff > 0 else 'FAIL',
                      len(p_ph['counts'])))
        cap = cap and w.cb_sample.count() == 4 and ph_diff > 0
        # 相近色统一（ICM）
        h_off = B.build_pattern(w.src_path, gw=72, gh=72, harmonize_on=False)
        h_on = B.build_pattern(w.src_path, gw=72, gh=72, harmonize_on=True)
        out.append('统一相近色：关闭 %d 色 → 开启 %d 色（改了 %d 格，%s）'
                   % (len(h_off['counts']), len(h_on['counts']), h_on['harmonized'],
                      'OK' if len(h_on['counts']) <= len(h_off['counts']) else 'FAIL'))
        cap = cap and len(h_on['counts']) <= len(h_off['counts'])
        # 小图先放大（Lanczos，实测比 AI 超分更接近原图）
        _tiny = os.path.join(d, 'tiny.png')
        Image.fromarray(a[::8, ::8]).save(_tiny)
        pt = B.build_pattern(_tiny, gw=48, gh=48)
        pt2 = B.build_pattern(_tiny, gw=48, gh=48, upscale=False)
        up_ok = pt['upscaled'] > 1 and pt2['upscaled'] == 1
        out.append('小图先放大：35×28 图 → 48×48 格（每格 %.1f px）自动放大 %d×；关掉后 %d×（%s）'
                   % (35 * 28 / 48.0 / 48.0, pt['upscaled'], pt2['upscaled'], 'OK' if up_ok else 'FAIL'))
        cap = cap and up_ok
        # 普通 / 高级 两档
        w.rb_ui_adv.setChecked(True)
        w.apply_mode()
        _adv_ok = (not w.sp_max.isHidden()) and (not w.g3.isHidden()) \
            and (not w.ck_harm.isHidden()) and (not w.ck_isle.isHidden()) \
            and (not w.sp_red.isHidden()) and (not w.ck_codes.isHidden())
        from PyQt6.QtWidgets import QGroupBox as _GB
        for _gg in w.findChildren(_GB):           # 先把各分组布局强制过一遍，否则量到 y=0
            if _gg.layout() is not None:
                _gg.layout().activate()
        _lay = w.centralWidget().layout()
        if _lay is not None:
            _lay.activate()
        app.processEvents()
        app.processEvents()
        _p3 = len({w.ck_codes.y(), w.ck_coord.y(), w.ck_legend.y()}) == 1        # ② 横排
        _f4 = len({w.ck_desp.y(), w.ck_isle.y(), w.ck_up.y(), w.ck_harm.y()}) == 2  # ⑤ 两行
        _bt = len({w.b_gen_top.y(), w.b_batch.y()}) == 1                        # ③ 同排
        _mid = lambda x: x.y() + x.height() / 2.0
        _rd = abs(_mid(w.sp_red) - _mid(w.sl_cell)) <= 4                        # 红线间隔同行
        _nl = ('Lanczos' not in w.ck_up.text()) and ('LANCZOS' not in w.ck_up.text())
        _layout_ok = _p3 and _f4 and _bt and _rd and _nl
        out.append('布局：印色号/坐标/用量横排=%s ｜ 噪点四项两行=%s ｜ 生成+导出同排=%s'
                   ' ｜ 红线间隔同行=%s ｜ 无 Lanczos 字样=%s（y:%s/%s/%s）'
                   % ('OK' if _p3 else 'FAIL', 'OK' if _f4 else 'FAIL',
                      'OK' if _bt else 'FAIL', 'OK' if _rd else 'FAIL',
                      'OK' if _nl else 'FAIL',
                      sorted({w.ck_codes.y(), w.ck_coord.y(), w.ck_legend.y()}),
                      sorted({w.ck_desp.y(), w.ck_isle.y(), w.ck_up.y(), w.ck_harm.y()}),
                      sorted({w.b_gen_top.y(), w.b_batch.y()})))
        cap = cap and _layout_ok
        w.rb_ui_basic.setChecked(True)
        w.apply_mode()
        _bas_ok = (w.g3.isHidden() and w.ck_harm.isHidden()
                   and w.sl_cell.isHidden()                     # 每格像素/图纸显示归高级档
                   and not w.sp_max.isHidden()                  # 但最大格数上限普通档也能调
                   and not w.cb_pal.isHidden() and not w.sp_w.isHidden()
                   and not w.rb_pre[0].isHidden() and not w.b_gen_top.isHidden())
        _pr_h = len({rb.y() for rb in w.rb_pre}) == 1      # 三个优化模式在同一横行
        _one_btn = (w.b_all is w.b_batch)                  # 只剩一个导出按钮
        # 拖放回归：DragEnter + DragMove + Drop 三个事件都要被接受并落到列表里
        # （历史上漏掉 DragMove → Qt 判定"不可放下" → 永远不派发 Drop → 拖放全废）
        from PyQt6.QtCore import QMimeData as _QMD, QPoint as _QP, QPointF as _QPF, QUrl as _QUrl
        from PyQt6.QtGui import (QDragEnterEvent as _DEE, QDragMoveEvent as _DME,
                                 QDropEvent as _DE)
        _png = os.path.join(d, 'dnd.png')
        Image.fromarray(a).save(_png)       # 自检要真的丢一个存在的文件进去
        _md = _QMD()
        _md.setUrls([_QUrl.fromLocalFile(_png)])
        _tgt = w.sl_cell
        _bad3 = []
        for _E, _nm in ((_DEE, 'DragEnter'), (_DME, 'DragMove')):
            _e = _E(_QP(5, 5), Qt.DropAction.CopyAction, _md, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(_tgt, _e)
            if not _e.isAccepted():
                _bad3.append(_nm + ' 未被接受')
        _n0 = w.lst.count()
        _e = _DE(_QPF(5, 5), Qt.DropAction.CopyAction, _md, Qt.MouseButton.LeftButton,
                 Qt.KeyboardModifier.NoModifier)
        w.lst.blockSignals(True)          # 别触发"选中就出图"（自检里没有事件循环）
        try:
            QApplication.sendEvent(_tgt, _e)
        finally:
            w.lst.blockSignals(False)
        _got3 = w.lst.count() - _n0
        if _got3 != 1:
            _bad3.append('Drop 没进列表(%d)' % _got3)
        _drop_ok = not _bad3
        _dnd = (getattr(w, '_drop_targets', 0) > 20 and w.pv.viewport().acceptDrops()
                and _drop_ok)
        out.append('优化模式横排=%s ｜ 导出按钮只剩一个=%s ｜ 到处可拖=%s（%d 个落点）'
                   '｜ 拖放三事件=%s%s'
                   % ('OK' if _pr_h else 'FAIL', 'OK' if _one_btn else 'FAIL',
                      'OK' if _dnd else 'FAIL', getattr(w, '_drop_targets', 0),
                      'OK' if _drop_ok else 'FAIL',
                      '' if _drop_ok else '（' + '、'.join(_bad3) + '）'))
        cap = cap and _pr_h and _one_btn and _dnd
        out.append('设置档位：普通=优化模式/格数/上限/色卡（%s）｜高级=全部参数（%s）'
                   % ('OK' if _bas_ok else 'FAIL', 'OK' if _adv_ok else 'FAIL'))
        cap = cap and _bas_ok and _adv_ok

        # 滑块数值标签不能被挤没 / 左栏控件要能全部够得着（可滚）
        w.rb_ui_basic.setChecked(True)
        w.apply_mode()
        _lbl_bas = (not w.lb_cell.isHidden() is False) and w.lb_cell.width() >= 30
        w.rb_ui_adv.setChecked(True)
        w.apply_mode()
        app.processEvents()
        _lbl_ok = all(x.width() >= 30 for x in (w.lb_cell, w.lb_maxc, w.lb_bg, w.lb_over))
        _sa = w.findChild(QScrollArea)
        _reach = bool(_sa) and (_sa.widget().height() <= _sa.viewport().height()
                                or _sa.verticalScrollBar().maximum() > 0)
        out.append('高级档：滑块数值可见=%s（宽 %d/%d/%d/%d）｜左栏可滚/可够=%s'
                   % ('OK' if _lbl_ok else 'FAIL', w.lb_cell.width(), w.lb_maxc.width(),
                      w.lb_bg.width(), w.lb_over.width(), 'OK' if _reach else 'FAIL'))
        cap = cap and _lbl_ok and _reach

        # 普通档也要能点格改色（v1.5.4：以前这里直接忽略点击，用户以为功能坏了）
        # 点格改色已删除：确认预览不再有涂色入口（点/右键都不改图）
        w.rb_ui_basic.setChecked(True)
        w.apply_mode()
        w.add_image(ip)
        w.lst.setCurrentRow(0)
        app.processEvents()
        w.generate()
        if w.worker:
            w.worker.wait(300000)
        app.processEvents()
        _before = w.pattern['idx'].copy() if w.pattern else None
        _no_edit = (not hasattr(w, 'ck_edit')) and (not hasattr(w, 'cb_color')) \
            and (not hasattr(Preview, 'cellClicked'))
        from PyQt6.QtCore import QPointF as _QPF
        from PyQt6.QtGui import QMouseEvent as _QME
        _ev = _QME(_QME.Type.MouseButtonPress, _QPF(50.0, 50.0), _QPF(50.0, 50.0),
                   __import__('PyQt6.QtCore', fromlist=['Qt']).Qt.MouseButton.LeftButton,
                   __import__('PyQt6.QtCore', fromlist=['Qt']).Qt.MouseButton.LeftButton,
                   __import__('PyQt6.QtCore', fromlist=['Qt']).Qt.KeyboardModifier.NoModifier)
        w.pv.mousePressEvent(_ev)
        app.processEvents()
        _same = (_before is not None and (w.pattern['idx'] == _before).all())
        out.append('点格改色已删除=%s ｜ 点/右键都不改图=%s'
                   % ('OK' if _no_edit else 'FAIL', 'OK' if _same else 'FAIL'))
        cap = cap and _no_edit and _same

        # 输入校验：非图片 / 不存在的路径 不进列表；文件夹自动挑图
        import tempfile as _tf2
        _d2 = _tf2.mkdtemp(prefix='cerabeads_val_')
        _bad = os.path.join(_d2, 'x.txt')
        open(_bad, 'w', encoding='utf-8').write('x')
        Image.fromarray(a).save(os.path.join(_d2, 'ok1.png'))
        os.makedirs(os.path.join(_d2, '子夹'), exist_ok=True)
        Image.fromarray(a).save(os.path.join(_d2, '子夹', 'ok2.png'))
        w.lst.clear()
        w._reset_canvas()
        w.add_image(_bad)
        _n1 = w.lst.count()
        w.add_image(os.path.join(_d2, '不存在.png'))
        _n2 = w.lst.count()
        w.add_image(_d2)
        _n3 = w.lst.count()
        _in_ok = (_n1 == 0 and _n2 == 0 and _n3 == 2)      # 文件夹递归挑到 2 张
        out.append('输入校验：非图片=%d 不存在=%d 文件夹挑图=%d（%s）'
                   % (_n1, _n2, _n3, 'OK' if _in_ok else 'FAIL'))
        cap = cap and _in_ok
        shutil.rmtree(_d2, ignore_errors=True)

        # 清空图片后画布与状态一起复位
        w.clear_imgs()
        app.processEvents()
        _clr = (w.pattern is None and w.src_path == '')
        out.append('清空后画布复位=%s' % ('OK' if _clr else 'FAIL'))
        cap = cap and _clr

        w.rb_ui_adv.setChecked(True)
        w.apply_mode()
        w.add_image(ip)
        w.lst.setCurrentRow(0)
        app.processEvents()
        # 诊断行真的写进了信息栏（线宽预警 + 色号预算）——这里曾经因为没导入 numpy 而静默失败
        w.auto_run = False
        w.generate()
        if getattr(w, 'worker', None):
            w.worker.wait(300000)
        app.processEvents()
        # 图片格式：常见格式 + HEIC/HEIF/AVIF 都要能读能出图
        _fmts = (('PNG', 'png'), ('JPEG', 'jpg'), ('WEBP', 'webp'), ('BMP', 'bmp'), ('GIF', 'gif'),
                 ('TIFF', 'tif'), ('AVIF', 'avif'), ('HEIF', 'heic'))
        _bad = []
        for _f, _e in _fmts:
            _p = os.path.join(d, 'fmt.' + _e)
            try:
                Image.fromarray(a).save(_p, format=_f)
                with Image.open(_p) as _im:
                    _im.load()
                _pt = B.build_pattern(_p, gw=20, gh=20)
                if not _pt or _pt.get('total', 0) <= 0:
                    _bad.append(_e)
            except Exception as _e2:
                _bad.append('%s(%s)' % (_e, str(_e2)[:40]))
        _fmt_ok = not _bad
        out.append('图片格式：%d/%d 种可用%s（HEIC/HEIF=%s）'
                   % (len(_fmts) - len(_bad), len(_fmts),
                      ('，缺：' + '、'.join(_bad)) if _bad else '',
                      '有' if getattr(B, 'HEIF_OK', False) else '无'))
        _pal_ok = len(B.palette_labels()) >= 11
        _pcounts = []
        try:
            for _lb in B.palette_labels():
                _pcounts.append('%d' % len(B.load_palette(_lb)['colors']))
        except Exception as _e3:
            _pal_ok = False
            _pcounts.append('读失败:%s' % str(_e3)[:30])
        out.append('色卡：%d 套（%s）默认=%s'
                   % (len(B.palette_labels()), '/'.join(_pcounts), B.PALETTE_DEFAULT))
        cap = cap and _fmt_ok and _pal_ok

        _itxt = w.lb_info.text()
        diag_ok = ('过渡带' in _itxt or '线宽' in _itxt) and '色号' in _itxt
        out.append('诊断行 = %s（%s）' % ('OK' if diag_ok else 'FAIL', _itxt[-110:]))
        cap = cap and diag_ok
        txt_ok = ok and cap
    else:
        txt_ok = False
    out.append('result = %s' % ('OK' if txt_ok else 'FAIL'))
    txt = '\n'.join(out)
    try:
        with open(os.path.join(app_dir(), '_selftest.txt'), 'w', encoding='utf-8') as f:
            f.write(txt)
    except Exception:
        pass
    return txt


def main():
    if '--selftest' in sys.argv:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        txt = selftest()
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
        print(txt)
        return 0
    app = QApplication(sys.argv)
    ic = icon_path()
    if ic:
        app.setWindowIcon(QIcon(ic))
    w = MainWindow()
    if ic:
        w.setWindowIcon(QIcon(ic))
    w.resize(1440, 940)
    w.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
