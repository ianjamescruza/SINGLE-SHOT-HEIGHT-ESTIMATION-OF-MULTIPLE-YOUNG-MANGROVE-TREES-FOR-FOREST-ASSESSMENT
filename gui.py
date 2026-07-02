import os
import sys
import cv2
from matplotlib import image
from matplotlib import image
import numpy as np
from datetime import datetime

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QFileDialog, QComboBox, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QScrollArea,
        QHeaderView, QMessageBox, QGroupBox, QSplitter, QToolButton,
        QGraphicsDropShadowEffect, QListView
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSize, QPropertyAnimation, QEasingCurve
    from PyQt5.QtGui import (
        QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush,
        QCursor, QLinearGradient, QPalette, QIcon
    )
except ImportError:
    print("PyQt5 not installed.")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed.")
    sys.exit(1)


# CONFIGURATION
BASE_DIR         = r"C:\Thesis"
WEIGHTS_DIR      = os.path.join(BASE_DIR, "weights")
CALIBRATION_FILE = os.path.join(BASE_DIR, "dataset", "calibration_matrix.npz")
OUTPUT_DIR       = os.path.join(BASE_DIR, "gui_results")

MODEL_OPTIONS = {
    "YOLOv8n": "mangrove_best_n.pt",
    "YOLOv8s": "mangrove_best_s.pt",
    "YOLOv8m": "mangrove_best_m.pt",
}
DEFAULT_MODEL = "YOLOv8s"

CLASS_TREE          = 0
CLASS_PIPE          = 1
PIPE_REAL_HEIGHT_M  = 1.0
CONF_THRESHOLD      = 0.25
IOU_THRESHOLD       = 0.45
INPUT_SIZE          = 640
UNDISTORT_ALPHA     = 0
MIN_PIPE_HEIGHT_PX      = 50     
MIN_TREE_HEIGHT_PX      = 60    
FRAME_EDGE_MARGIN       = 10
FOREGROUND_MARGIN       = 150    
DEPTH_TOLERANCE         = 80     
HORIZONTAL_THRESHOLD    = 500    
PIPE_RADIUS             = 315   
MAX_REASONABLE_HEIGHT_M = 5.0    

# CV colors (BGR)
CV_GREEN  = (34,  180,  34)
CV_ORANGE = (30,  140, 255)
CV_BLUE   = (255, 140,  30)
CV_RED    = (30,   30, 220)
CV_BLACK  = (15,   15,  15)

# THEME DEFINITIONS
THEMES = {
    "dark": {
        "name": "dark",
        "label": "🌙  Night Mode",
        "toggle_label": "☀  Day Mode",

        # Base palette
        "bg_root":       "#0d0f14",
        "bg_panel_l":    "#0f1117",
        "bg_panel_r":    "#0b0d12",
        "bg_header":     "#0c1510",
        "bg_img":        "#07090e",
        "bg_card":       "#111520",
        "bg_table":      "#0d1018",
        "bg_table_alt":  "#0f1420",
        "bg_statusbar":  "#0b0e13",
        "bg_legend":     "#0b0d14",
        "bg_combo":      "#141c28",
        "bg_tip":        "#0f1825",
        "bg_badge":      "#111a2a",

        # Borders
        "border_main":   "#1a2030",
        "border_header": "#1e4d2b",
        "border_panel":  "#1a2030",
        "border_combo":  "#1e3050",
        "border_tip":    "#1e3040",
        "border_badge":  "#1a2a3a",

        # Text
        "txt_primary":   "#e8eaf0",
        "txt_secondary": "#5a7080",
        "txt_dim":       "#2e3e4e",
        "txt_hint":      "#2e4858",
        "txt_header_sub":"#4a7055",
        "txt_header_lbl":"#4a6070",
        "txt_table_hdr": "#5a7a9a",
        "txt_model":     "#3a5a70",
        "txt_statusbar": "#4a6a5a",
        "txt_img_empty": "#2e3e4e",
        "txt_leg":       "#5a6a7a",
        "txt_infolbl":   "#3a5060",

        # Accent
        "accent_green":  "#4ade80",
        "accent_green_d":"#22b422",
        "accent_orange": "#ff8c1e",
        "accent_blue":   "#4aacff",
        "accent_red":    "#e02020",
        "accent_dim":    "#3a5060",

        # Buttons
        "btn_load_bg":   "#162d20",
        "btn_load_bg_h": "#1e3828",
        "btn_load_txt":  "#4ade80",
        "btn_load_bdr":  "#2a4f38",
        "btn_run_bg":    "#4ade80",
        "btn_run_txt":   "#040e07",
        "btn_run_bg_h":  "#6af09a",
        "btn_run_dis_bg":"#162a1c",
        "btn_run_dis_t": "#2e4838",
        "btn_save_bg":   "#141e2e",
        "btn_save_bg_h": "#1c2838",
        "btn_save_txt":  "#6a8aaa",
        "btn_save_bdr":  "#1c2d40",
        "btn_save_dis_bg":"#0e1520",
        "btn_save_dis_t": "#283040",

        # QT accent colors
        "QT_GREEN":  "#22b422",
        "QT_ORANGE": "#ff8c1e",
        "QT_BLUE":   "#4aacff",
        "QT_RED":    "#e02020",

        # Scrollbar
        "scroll_bg":     "#0b0d12",
        "scroll_handle": "#1e2a38",

        # Splitter
        "splitter":      "#1a2030",
        "group_color":   "#5a7a8a",
        "group_border":  "#1a2535",
    },
    "light": {
        "name": "light",
        "label": "☀  Day Mode",
        "toggle_label": "🌙  Night Mode",

        # Base palette
        "bg_root":       "#f0f2f5",
        "bg_panel_l":    "#ffffff",
        "bg_panel_r":    "#f7f9fc",
        "bg_header":     "#1b3a2a",
        "bg_img":        "#e8ecf0",
        "bg_card":       "#ffffff",
        "bg_table":      "#ffffff",
        "bg_table_alt":  "#f5f8fb",
        "bg_statusbar":  "#eaecf0",
        "bg_legend":     "#eef1f6",
        "bg_combo":      "#ffffff",
        "bg_tip":        "#ffffff",
        "bg_badge":      "#e8f0ea",

        # Borders
        "border_main":   "#d0d8e4",
        "border_header": "#2a5a3a",
        "border_panel":  "#d8dde8",
        "border_combo":  "#b0c0d8",
        "border_tip":    "#c8d4e4",
        "border_badge":  "#b8ccd0",

        # Text
        "txt_primary":   "#1a2030",
        "txt_secondary": "#5a6a7a",
        "txt_dim":       "#8090a0",
        "txt_hint":      "#607080",
        "txt_header_sub":"#a0c8b0",
        "txt_header_lbl":"#90b0c0",
        "txt_table_hdr": "#5a6a8a",
        "txt_model":     "#5a7080",
        "txt_statusbar": "#3a5a4a",
        "txt_img_empty": "#8090a8",
        "txt_leg":       "#6a7a8a",
        "txt_infolbl":   "#6a7a8a",

        # Accent
        "accent_green":  "#1a7a3a",
        "accent_green_d":"#158030",
        "accent_orange": "#c05010",
        "accent_blue":   "#1a60c0",
        "accent_red":    "#c02020",
        "accent_dim":    "#8090a8",

        # Buttons
        "btn_load_bg":   "#d8efe0",
        "btn_load_bg_h": "#c5e4d0",
        "btn_load_txt":  "#1a6a30",
        "btn_load_bdr":  "#90c8a8",
        "btn_run_bg":    "#1a7a3a",
        "btn_run_txt":   "#ffffff",
        "btn_run_bg_h":  "#1e8f44",
        "btn_run_dis_bg":"#c8dace",
        "btn_run_dis_t": "#7a9a88",
        "btn_save_bg":   "#dce4f0",
        "btn_save_bg_h": "#ccd8e8",
        "btn_save_txt":  "#3a5070",
        "btn_save_bdr":  "#b0c0d8",
        "btn_save_dis_bg":"#e8edf5",
        "btn_save_dis_t": "#a0afc0",

        # QT accent colors
        "QT_GREEN":  "#158030",
        "QT_ORANGE": "#c05010",
        "QT_BLUE":   "#1a60c0",
        "QT_RED":    "#c02020",

        # Scrollbar
        "scroll_bg":     "#e8ecf2",
        "scroll_handle": "#b0bccC",

        # Splitter
        "splitter":      "#d0d8e4",
        "group_color":   "#3a5a6a",
        "group_border":  "#c8d4e0",
    }
}


# DETECTION ENGINE
def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None, None
    d = np.load(CALIBRATION_FILE)
    return d["camera_matrix"], d["dist_coeffs"]


def undistort(img, cam, dist):
    if cam is None:
        return img
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam, dist, (w, h),
                                                  UNDISTORT_ALPHA, (w, h))
    out = cv2.undistort(img, cam, dist, None, new_mtx)
    if UNDISTORT_ALPHA == 0:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            out = out[y:y+rh, x:x+rw]
    return out


def assign_pipe(tree, valid_pipes):
    tree_cx, tree_base = tree["cx"], tree["y2"]
    best_pipe, best_score = None, float("inf")
    best_h, best_d, best_c = None, None, None

    for p in valid_pipes:
        pipe_cx, pipe_base = p["cx"], p["y2"]

        # 1 — foreground margin
        if tree_base < pipe_base - FOREGROUND_MARGIN:
            continue

        # 2 — depth tolerance
        depth_diff = abs(tree_base - pipe_base)
        if depth_diff > DEPTH_TOLERANCE:
            continue

        # 3 — horizontal proximity
        h_dist = abs(tree_cx - pipe_cx)
        if h_dist > HORIZONTAL_THRESHOLD:
            continue

        # 4 — circular distance from tree base to pipe base
        circle_dist = (h_dist ** 2 + depth_diff ** 2) ** 0.5
        if circle_dist > PIPE_RADIUS:
            continue

        # All four phases passed 
        score = h_dist + 0.5 * depth_diff + 0.5 * circle_dist
        if score < best_score:
            best_score, best_pipe = score, p
            best_h, best_d, best_c = h_dist, depth_diff, circle_dist

    return best_pipe, best_h, best_d, best_c


def run_estimation(model, img_bgr, cam, dist):
    img = undistort(img_bgr, cam, dist)
    img_h, img_w = img.shape[:2]

    raw = model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        imgsz=INPUT_SIZE, verbose=False)

    trees_raw, pipes_raw = [], []
    for r in raw:
        for box in r.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = dict(cls=cid, conf=conf,
                       x1=x1, y1=y1, x2=x2, y2=y2,
                       cx=(x1+x2)/2, cy=(y1+y2)/2,
                       width=x2-x1, height=y2-y1)
            if cid == CLASS_TREE:
                trees_raw.append(det)
            elif cid == CLASS_PIPE:
                pipes_raw.append(det)

    trees_raw.sort(key=lambda t: t["x1"])

    # Pipe quality filter 
    valid_pipes = []
    for p in pipes_raw:
        Ps = p["height"]
        if p["y2"] > img_h - FRAME_EDGE_MARGIN:
            continue  
        if Ps < MIN_PIPE_HEIGHT_PX:
            continue
        p["k"]  = PIPE_REAL_HEIGHT_M / Ps
        p["Ps"] = Ps
        valid_pipes.append(p)

    results = []
    for i, tree in enumerate(trees_raw):
        tid = f"T{i+1}"

        # 1 — SHORT_BOX 
        if tree["height"] < MIN_TREE_HEIGHT_PX:
            results.append(dict(
                tree_id=tid, conf=round(tree["conf"], 2), status="SHORT_BOX",
                height_m=None, k=None, h_dist=None, pipe_lbl="—",
                x1=tree["x1"], y1=tree["y1"], x2=tree["x2"], y2=tree["y2"],
            ))
            continue

        # 2 — frame-truncation check
        top_cut    = tree["y1"] <= FRAME_EDGE_MARGIN
        bottom_cut = tree["y2"] >= img_h - FRAME_EDGE_MARGIN
        if top_cut or bottom_cut:
            results.append(dict(
                tree_id=tid, conf=round(tree["conf"], 2), status="PARTIAL",
                height_m=None, k=None, h_dist=None, pipe_lbl="—",
                x1=tree["x1"], y1=tree["y1"], x2=tree["x2"], y2=tree["y2"],
            ))
            continue

        # 3 — pipe availability check
        if not valid_pipes:
            results.append(dict(
                tree_id=tid, conf=round(tree["conf"], 2), status="NO_PIPE",
                height_m=None, k=None, h_dist=None, pipe_lbl="—",
                x1=tree["x1"], y1=tree["y1"], x2=tree["x2"], y2=tree["y2"],
            ))
            continue

        # 4 — four-gate zone assignment 
        best_pipe, h_dist, depth_diff, circle_dist = assign_pipe(tree, valid_pipes)

        if best_pipe is None:
            results.append(dict(
                tree_id=tid, conf=round(tree["conf"], 2), status="TOO_FAR",
                height_m=None, k=None, h_dist=None, pipe_lbl="—",
                x1=tree["x1"], y1=tree["y1"], x2=tree["x2"], y2=tree["y2"],
            ))
            continue

        # 5 — pixel-to-metric conversion
        k_used   = best_pipe["k"]
        height_m = round(tree["height"] * k_used, 2)
        status   = "MEASURED"
        if height_m > MAX_REASONABLE_HEIGHT_M:
            status = "HEIGHT_SUSPECT"

        results.append(dict(
            tree_id  = tid,
            conf     = round(tree["conf"], 2),
            status   = status,
            height_m = height_m,
            k        = round(k_used, 5),
            h_dist   = round(h_dist, 0),
            pipe_lbl = f"Pipe{valid_pipes.index(best_pipe)+1}",
            x1=tree["x1"], y1=tree["y1"],
            x2=tree["x2"], y2=tree["y2"],
        ))

    return results, valid_pipes, img


def draw_boxes_cv(img, results, valid_pipes):
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, int(min(h, w) / 320))
    fs = max(0.4, min(w, h) / 1300)
    ft = max(1, th - 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(img, txt, x, y, color):
        (tw, tht), _ = cv2.getTextSize(txt, font, fs, ft)
        y0 = max(y - 5, tht + 5)
        cv2.rectangle(img, (int(x), int(y0-tht-4)),
                      (int(x+tw+7), int(y0+3)), CV_BLACK, -1)
        cv2.putText(img, txt, (int(x+3), int(y0)),
                    font, fs, color, ft, cv2.LINE_AA)

    for i, p in enumerate(valid_pipes):
        x1,y1,x2,y2 = int(p["x1"]),int(p["y1"]),int(p["x2"]),int(p["y2"])
        cv2.rectangle(out, (x1,y1),(x2,y2), CV_RED, th)
        label(out, f"Pipe{i+1}", x1, y1, CV_RED)

    color_map = dict(MEASURED=CV_GREEN, TOO_FAR=CV_ORANGE,
                     PARTIAL=CV_BLUE, NO_PIPE=CV_ORANGE,
                     SHORT_BOX=CV_BLUE, HEIGHT_SUSPECT=CV_ORANGE)
    for r in results:
        x1,y1 = int(r["x1"]),int(r["y1"])
        x2,y2 = int(r["x2"]),int(r["y2"])
        color  = color_map.get(r["status"], CV_ORANGE)
        cv2.rectangle(out, (x1,y1),(x2,y2), color, th)
        if r["status"] == "MEASURED":
            cl = int(min(x2-x1, y2-y1) * 0.15)
            ct = th + 1
            for cx,cy,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),
                                  (x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out,(cx,cy),(cx+sx*cl,cy),color,ct)
                cv2.line(out,(cx,cy),(cx,cy+sy*cl),color,ct)
            label(out, f"{r['tree_id']}  {r['height_m']:.2f} m", x1, y1, color)
        elif r["status"] == "PARTIAL":
            label(out, f"{r['tree_id']}  partial", x1, y1, color)
        elif r["status"] == "TOO_FAR":
            label(out, f"{r['tree_id']}  detected", x1, y1, color)
        elif r["status"] == "SHORT_BOX":
            label(out, f"{r['tree_id']}  short box", x1, y1, color)
        elif r["status"] == "HEIGHT_SUSPECT":
            label(out, f"{r['tree_id']}  suspect {r['height_m']:.2f} m", x1, y1, color)
        elif r["status"] == "NO_PIPE":
            label(out, f"{r['tree_id']}  no pipe", x1, y1, color)
        else:
            label(out, f"{r['tree_id']}  {r['status'].lower()}", x1, y1, color)

    return out

# HOVER IMAGE LABEL
class HoverImageLabel(QLabel):
    hovered_index = pyqtSignal(int)

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self._results    = []
        self._pipes      = []
        self._pix        = None
        self._img_rect   = QRect()
        self._hov_idx    = -1

        # Tooltip card
        self._tip = QFrame(self)
        self._tip.setObjectName("tipCard")
        self._tip.setFixedWidth(210)
        self._tip.hide()

        tip_lay = QVBoxLayout(self._tip)
        tip_lay.setContentsMargins(14, 12, 14, 12)
        tip_lay.setSpacing(5)

        self._tip_id = QLabel()
        self._tip_id.setObjectName("tipId")
        tip_lay.addWidget(self._tip_id)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("tipSep")
        tip_lay.addWidget(sep)

        self._tip_rows = {}
        for key, label in [("status","Status"), ("height","Height (m)"),
                            ("conf","Confidence"), ("pipe","Nearest Pipe"),
                            ("k","Scale Factor k"), ("dist","Pipe Distance")]:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setObjectName("tipLabel")
            val = QLabel("—")
            val.setObjectName("tipVal")
            val.setAlignment(Qt.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            tip_lay.addLayout(row)
            self._tip_rows[key] = val

    def update_theme(self, theme):
        self.theme = theme

    def load_results(self, results, pipes, pixmap):
        self._results  = results
        self._pipes    = pipes
        self._pix      = pixmap
        self._hov_idx  = -1
        self._tip.hide()
        self._update_img_rect()
        self.update()

    def _update_img_rect(self):
        if self._pix is None:
            return
        lw, lh = self.width(), self.height()
        pw, ph = self._pix.width(), self._pix.height()
        scale  = min(lw / pw, lh / ph)
        dw     = int(pw * scale)
        dh     = int(ph * scale)
        ox     = (lw - dw) // 2
        oy     = (lh - dh) // 2
        self._img_rect = QRect(ox, oy, dw, dh)
        self._scale    = scale

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_img_rect()
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._pix is None or not self._results:
            return

        p  = QPainter(self)
        r  = self._img_rect
        sc = self._scale

        scaled = self._pix.scaled(r.width(), r.height(),
                                   Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
        p.drawPixmap(r.x(), r.y(), scaled)

        T = self.theme
        color_map = dict(
            MEASURED=T["QT_GREEN"],
            TOO_FAR=T["QT_ORANGE"],
            PARTIAL=T["QT_BLUE"],
            NO_PIPE=T["QT_ORANGE"],
            SHORT_BOX=T["QT_BLUE"],
            HEIGHT_SUSPECT=T["QT_ORANGE"]
        )

        for i, pipe in enumerate(self._pipes):
            x1 = int(pipe["x1"] * sc) + r.x()
            y1 = int(pipe["y1"] * sc) + r.y()
            x2 = int(pipe["x2"] * sc) + r.x()
            y2 = int(pipe["y2"] * sc) + r.y()
            pen = QPen(QColor(T["QT_RED"]), 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(x1, y1, x2-x1, y2-y1)
            p.setPen(QColor(T["QT_RED"]))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(x1+3, y1-4, f"Pipe{i+1}")

        for idx, res in enumerate(self._results):
            is_hov = idx == self._hov_idx
            color  = QColor(color_map.get(res["status"], T["QT_ORANGE"]))
            x1 = int(res["x1"] * sc) + r.x()
            y1 = int(res["y1"] * sc) + r.y()
            x2 = int(res["x2"] * sc) + r.x()
            y2 = int(res["y2"] * sc) + r.y()
            bw = x2 - x1
            bh = y2 - y1

            pen = QPen(color, 3 if is_hov else 2)
            p.setPen(pen)

            if is_hov:
                fill = QColor(color)
                fill.setAlpha(40)
                p.setBrush(QBrush(fill))
            else:
                p.setBrush(Qt.NoBrush)

            p.drawRect(x1, y1, bw, bh)

            if res["status"] == "MEASURED":
                cl = int(min(bw, bh) * (0.20 if is_hov else 0.15))
                ct = 3 if is_hov else 2
                pen2 = QPen(color, ct)
                p.setPen(pen2)
                for cx, cy, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),
                                        (x1,y2,1,-1),(x2,y2,-1,-1)]:
                    p.drawLine(cx, cy, cx+sx*cl, cy)
                    p.drawLine(cx, cy, cx, cy+sy*cl)

            p.setFont(QFont("Segoe UI", 9,
                            QFont.Bold if is_hov else QFont.Normal))
            p.setPen(QColor(color))
            p.drawText(x1+4, y1-4, res["tree_id"])

        p.end()

    def mouseMoveEvent(self, e):
        if not self._results or self._pix is None:
            return
        r   = self._img_rect
        sc  = self._scale
        mx  = e.x()
        my  = e.y()

        hov = -1
        for idx, res in enumerate(self._results):
            x1 = int(res["x1"] * sc) + r.x()
            y1 = int(res["y1"] * sc) + r.y()
            x2 = int(res["x2"] * sc) + r.x()
            y2 = int(res["y2"] * sc) + r.y()
            if x1 <= mx <= x2 and y1 <= my <= y2:
                hov = idx
                break

        if hov != self._hov_idx:
            self._hov_idx = hov
            self.hovered_index.emit(hov)
            self.update()

        if hov >= 0:
            self._show_tip(hov, mx, my)
        else:
            self._tip.hide()

    def leaveEvent(self, e):
        self._hov_idx = -1
        self._tip.hide()
        self.hovered_index.emit(-1)
        self.update()

    def _show_tip(self, idx, mx, my):
        T = self.theme
        res = self._results[idx]

        TIP_BG       = "#111c14"
        TIP_TXT      = "#c8dcc8"
        TIP_DIM      = "#5a7a5a"
        TIP_MEASURED = "#4ade80"   

        status_colors = dict(MEASURED="#4ade80", TOO_FAR="#ff8c1e",
                             PARTIAL="#4aacff", NO_PIPE="#ff8c1e",
                             SHORT_BOX="#4aacff", HEIGHT_SUSPECT="#ff8c1e")
        color = status_colors.get(res["status"], "#ff8c1e")

        self._tip_id.setText(f"  {res['tree_id']}")
        self._tip_id.setStyleSheet(
            f"font-size:15px; font-weight:700; color:{color}; "
            f"letter-spacing:1px; background:transparent;"
        )

        self._tip_rows["status"].setText(res["status"])
        self._tip_rows["status"].setStyleSheet(
            f"color:{color}; font-weight:600; background:transparent;"
        )

        if res["height_m"] is not None:
            self._tip_rows["height"].setText(f"{res['height_m']:.2f} m")
            self._tip_rows["height"].setStyleSheet(
                f"color:{TIP_MEASURED}; font-weight:700; font-size:14px; background:transparent;"
            )
        else:
            self._tip_rows["height"].setText("—")
            self._tip_rows["height"].setStyleSheet(
                f"color:{TIP_DIM}; background:transparent;"
            )

        for key in ("conf", "pipe"):
            val_txt = str(res["conf"]) if key == "conf" else res.get("pipe_lbl", "—")
            self._tip_rows[key].setText(val_txt)
            self._tip_rows[key].setStyleSheet(
                f"color:{TIP_TXT}; background:transparent;"
            )

        if res["k"] is not None:
            self._tip_rows["k"].setText(str(res["k"]))
            self._tip_rows["k"].setStyleSheet(
                f"color:{TIP_TXT}; background:transparent;"
            )
        else:
            self._tip_rows["k"].setText("—")
            self._tip_rows["k"].setStyleSheet(
                f"color:{TIP_DIM}; background:transparent;"
            )

        if res["h_dist"] is not None:
            self._tip_rows["dist"].setText(f"{int(res['h_dist'])} px")
            self._tip_rows["dist"].setStyleSheet(
                f"color:{TIP_TXT}; background:transparent;"
            )
        else:
            self._tip_rows["dist"].setText("—")
            self._tip_rows["dist"].setStyleSheet(
                f"color:{TIP_DIM}; background:transparent;"
            )

        self._tip.adjustSize()
        tw = self._tip.width()
        th = self._tip.height()
        lw = self.width()
        lh = self.height()
        tx = mx + 16
        ty = my - th // 2
        if tx + tw > lw:
            tx = mx - tw - 10
        ty = max(4, min(ty, lh - th - 4))
        self._tip.move(tx, ty)
        self._tip.show()
        self._tip.raise_()

# BACKGROUND DETECTION THREAD
class DetectionWorker(QThread):
    finished = pyqtSignal(list, list, object)
    error    = pyqtSignal(str)

    def __init__(self, model, img, cam, dist):
        super().__init__()
        self.model = model
        self.img   = img
        self.cam   = cam
        self.dist  = dist

    def run(self):
        try:
            results, pipes, clean = run_estimation(
                self.model, self.img, self.cam, self.dist
            )
            self.finished.emit(results, pipes, clean)
        except Exception as ex:
            self.error.emit(str(ex))



# STAT CARD WIDGET
class StatCard(QFrame):
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        self._color = color
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        # Color accent bar at top
        self._accent = QFrame()
        self._accent.setFixedHeight(3)
        self._accent.setStyleSheet(f"background:{color}; border-radius:2px;")
        layout.addWidget(self._accent)

        # Value label
        self.val_lbl = QLabel("—")
        self.val_lbl.setObjectName("cardVal")
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setStyleSheet(f"color:{color}; font-size:26px; font-weight:700; font-family:'Segoe UI',monospace;")
        layout.addWidget(self.val_lbl)

        # Metric label
        lbl = QLabel(label)
        lbl.setObjectName("cardLbl")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    def set_value(self, v):
        self.val_lbl.setText(str(v))

# MAIN WINDOW
class MangroveGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.model        = None
        self.model_name   = None
        self.cam          = None
        self.dist         = None
        self.current_img  = None
        self.current_path = None
        self.clean_img    = None
        self.results      = []
        self.pipes        = []
        self.worker       = None
        self._theme_name  = "dark"   

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.cam, self.dist = load_calibration()

        self._setup_ui()
        self._apply_style()
        self._load_model(DEFAULT_MODEL)

    # THEME ACCESSOR
    @property
    def T(self):
        return THEMES[self._theme_name]

    # UI SETUP
    def _setup_ui(self):
        self.setWindowTitle("Mangrove Tree Height Estimation System")
        self.setMinimumSize(1280, 780)
        self.resize(1440, 860)

        root = QWidget()
        self.setCentralWidget(root)
        lay  = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._make_header())

        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(2)
        body.addWidget(self._make_left())
        body.addWidget(self._make_right())
        body.setSizes([940, 400])
        lay.addWidget(body, 1)

        lay.addWidget(self._make_statusbar())

    # HEADER
    def _make_header(self):
        self._header_frame = QFrame()
        self._header_frame.setObjectName("header")
        self._header_frame.setFixedHeight(68)

        lay = QHBoxLayout(self._header_frame)
        lay.setContentsMargins(16, 0, 20, 0)
        lay.setSpacing(0)

        tb = QVBoxLayout()
        tb.setSpacing(3)
        tb.setContentsMargins(0, 0, 0, 0)

        t1 = QLabel("SINGLE-SHOT MANGROVE TREE HEIGHT ESTIMATION SYSTEM")
        t1.setObjectName("hTitle")
        t1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        tb.addWidget(t1)

        title_wrap = QWidget()
        title_wrap.setLayout(tb)
        title_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(title_wrap, 1)

        # Vertical divider
        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.VLine)
        vdiv.setObjectName("headerVDiv")
        lay.addWidget(vdiv)

        # Model selector
        mb = QVBoxLayout()
        mb.setSpacing(3)
        mb.setContentsMargins(0, 0, 0, 0)
        ml = QLabel("ACTIVE MODEL")
        ml.setObjectName("hLabel")
        self.combo = QComboBox()
        self.combo.setObjectName("modelCombo")
        self.combo.setFixedWidth(155)
        self.combo.setFixedHeight(30)
        self.combo.setView(QListView())
        self.combo.view().setObjectName("modelComboView")
        self.combo.view().setMinimumWidth(self.combo.width())
        self.combo.view().setSpacing(0)
        self.combo.setMaxVisibleItems(len(MODEL_OPTIONS))

        for name in MODEL_OPTIONS:
            self.combo.addItem(name)
        self.combo.setCurrentText(DEFAULT_MODEL)
        self.combo.currentTextChanged.connect(self._load_model)
        mb.addWidget(ml)
        mb.addWidget(self.combo)
        lay.addLayout(mb)

        # Vertical divider
        vdiv2 = QFrame()
        vdiv2.setFrameShape(QFrame.VLine)
        vdiv2.setObjectName("headerVDiv")
        lay.addWidget(vdiv2)

        # Theme toggle button 
        theme_box = QWidget()
        theme_box.setFixedWidth(190)

        theme_layout = QHBoxLayout(theme_box)
        theme_layout.setContentsMargins(0, 0, 0, 0)
        theme_layout.setSpacing(0)

        self.theme_btn = QPushButton(self.T["label"])
        self.theme_btn.setObjectName("themeBtn")
        self.theme_btn.setFixedHeight(36)
        self.theme_btn.setFixedWidth(140)
        self.theme_btn.clicked.connect(self._toggle_theme)

        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_btn)
        theme_layout.addStretch()

        lay.addWidget(theme_box)

        return self._header_frame

    # LEFT PANEL
    def _make_left(self):
        f = QFrame()
        f.setObjectName("leftPanel")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 10, 12)
        lay.setSpacing(10)

        # Toolbar 
        tb = QHBoxLayout()
        tb.setSpacing(8)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setObjectName("btnLoad")
        self.btn_load.setFixedHeight(38)
        self.btn_load.clicked.connect(self._load_image)

        self.btn_run = QPushButton("Run Estimation")
        self.btn_run.setObjectName("btnRun")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.setFixedHeight(38)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)

        tb.addWidget(self.btn_load)
        tb.addWidget(self.btn_run)
        tb.addWidget(self.btn_save)
        tb.addStretch()
        lay.addLayout(tb)

        # Thin separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("hSepLine")
        lay.addWidget(sep)

        # Image label 
        self.img_lbl = HoverImageLabel(self.T)
        self.img_lbl.setObjectName("imgLbl")
        self.img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_lbl.setText(
            "No image loaded\n\nClick Load Image to begin"
        )
        self.img_lbl.hovered_index.connect(self._on_hover)
        lay.addWidget(self.img_lbl, 1)

        # Legend 
        leg = QFrame()
        leg.setObjectName("legend")
        ll  = QHBoxLayout(leg)
        ll.setContentsMargins(10, 6, 10, 6)
        ll.setSpacing(16)

        self._legend_items = []
        for color_key, label in [
            ("QT_GREEN",  "Measured"),
            ("QT_ORANGE", "Detected Only"),
            ("QT_BLUE",   "Partial Tree"),
            ("QT_RED",    "Reference Pipe"),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{self.T[color_key]};font-size:14px;")
            dot.setProperty("color_key", color_key)
            lbl = QLabel(label)
            lbl.setObjectName("legTxt")
            ll.addWidget(dot)
            ll.addWidget(lbl)
            self._legend_items.append(dot)

        ll.addStretch()
        lay.addWidget(leg)

        self.hint_lbl = QLabel("  Hover over any bounding box to inspect that tree")
        self.hint_lbl.setObjectName("hintLbl")
        lay.addWidget(self.hint_lbl)

        return f

    # RIGHT PANEL
    def _make_right(self):
        f = QFrame()
        f.setObjectName("rightPanel")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(10, 14, 16, 12)
        lay.setSpacing(12)

        # Summary section label 
        sec_lbl = QLabel("  DETECTION SUMMARY")
        sec_lbl.setObjectName("sectionLbl")
        lay.addWidget(sec_lbl)

        # Stat cards grid
        self.stat_cards = {}
        color_key_map = {
            "total":    None,
            "measured": "QT_GREEN",
            "far":      "QT_ORANGE",
            "partial":  "QT_BLUE",
            "pipes":    "QT_RED",
        }
        cards_meta = [
            ("total",    "Trees Detected"),
            ("measured", "Height Estimated"),
            ("far",      "Detected Only"),
            ("partial",  "Partial Trees"),
            ("pipes",    "Reference Pipes"),
        ]

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        for i, (key, label) in enumerate(cards_meta):
            ck = color_key_map[key]
            color = self.T[ck] if ck else "#8090a8"
            card = StatCard(label, color)
            card.setMinimumHeight(78)
            self.stat_cards[key] = card
            if i < 2:
                row1.addWidget(card)
            elif i < 4:
                row2.addWidget(card)
            else:
                row3.addWidget(card, 1)
                row3.addStretch(1)

        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(row3)

        # Results table 
        sec_lbl2 = QLabel(" MANGROVE TREE RESULTS")
        sec_lbl2.setObjectName("sectionLbl")
        lay.addWidget(sec_lbl2)

        self.table = QTableWidget()
        self.table.setObjectName("resTable")
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Tree", "Status", "Height (m)", "Conf"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        lay.addWidget(self.table, 1)

        # Info label 
        self.info_lbl = QLabel("No image loaded")
        self.info_lbl.setObjectName("infoLbl")
        self.info_lbl.setWordWrap(True)
        lay.addWidget(self.info_lbl)

        return f

    # STATUS BAR
    def _make_statusbar(self):
        sb = QFrame()
        sb.setObjectName("statusBar")
        sb.setFixedHeight(28)
        sl = QHBoxLayout(sb)
        sl.setContentsMargins(14, 0, 14, 0)
        sl.setSpacing(8)

        # Green dot indicator
        self._status_dot = QLabel("●")
        self._status_dot.setObjectName("statusDot")
        self._status_dot.setFixedWidth(14)
        sl.addWidget(self._status_dot)

        # Status text
        self.status_lbl = QLabel("System ready — load an image to begin")
        self.status_lbl.setObjectName("statusTxt")
        self.status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.status_lbl.setMinimumWidth(0)
        sl.addWidget(self.status_lbl, stretch=1)

        # Thin separator
        div = QLabel("·")
        div.setObjectName("statusTxt")
        div.setFixedWidth(10)
        div.setAlignment(Qt.AlignCenter)
        sl.addWidget(div)

        # Model info 
        self.model_lbl = QLabel()
        self.model_lbl.setObjectName("modelTxt")
        self.model_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.model_lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sl.addWidget(self.model_lbl)

        return sb

    # THEME TOGGLE
    def _toggle_theme(self):
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self.theme_btn.setText(self.T["label"])
        self.img_lbl.update_theme(self.T)

        for dot in self._legend_items:
            ck = dot.property("color_key")
            dot.setStyleSheet(f"color:{self.T[ck]};font-size:14px;")

        color_key_map = {
            "measured": "QT_GREEN",
            "far":      "QT_ORANGE",
            "partial":  "QT_BLUE",
            "pipes":    "QT_RED",
        }
        for key, ck in color_key_map.items():
            card = self.stat_cards[key]
            color = self.T[ck]
            card._accent.setStyleSheet(f"background:{color}; border-radius:2px;")
            card.val_lbl.setStyleSheet(f"color:{color}; font-size:26px; font-weight:700; font-family:'Segoe UI',monospace;")

        self._apply_style()
        self.update()

    # STYLESHEET
    def _apply_style(self):
        T = self.T
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {T['bg_root']};
            color: {T['txt_primary']};
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 13px;
        }}

        /* Header */
        QFrame#header {{
            background: {T['bg_header']};
            border-bottom: 2px solid {T['border_header']};
        }}
        QFrame#header QWidget,
        QFrame#header QFrame {{
            background: transparent;
        }}
        #headerAccentBar {{
            background: {T['accent_green']};
            border-radius: 2px;
        }}
        #headerVDiv {{
            color: {T['border_header']};
            max-width: 1px;
        }}
        #hTitle {{
            font-size: 15px;
            font-weight: 800;
            color: #4ade80;
            letter-spacing: 2px;
            background: transparent;
        }}
        #hSub {{
            font-size: 10px;
            color: {T['txt_header_sub']};
            letter-spacing: 0.3px;
            background: transparent;
        }}
        #hLabel {{
            font-size: 9px;
            color: {T['txt_header_lbl']};
            font-weight: 700;
            letter-spacing: 1px;
            background: transparent;
        }}

        /* Theme button */
        #themeBtn {{
            background: {T['btn_load_bg']};
            color: #4ade80;
            border: 1px solid {T['btn_load_bdr']};
            border-radius: 6px;
            padding: 0 12px;
            font-size: 12px;
            font-weight: 700;
        }}
        #themeBtn:hover {{
            background: {T['btn_load_bg_h']};
        }}

        /* Panels */
        #leftPanel {{
            background: {T['bg_panel_l']};
            border-right: 1px solid {T['border_panel']};
        }}
        #rightPanel {{
            background: {T['bg_panel_r']};
        }}

        /* Image label */
        #imgLbl {{
            background: {T['bg_img']};
            color: {T['txt_img_empty']};
            font-size: 15px;
            border: 1px solid {T['border_main']};
            border-radius: 8px;
        }}

        /* Separator */
        #hSepLine {{
            color: {T['border_main']};
        }}

        /* Action Buttons */
        QPushButton {{
            border-radius: 6px;
            padding: 0 18px;
            font-size: 13px;
            font-weight: 600;
        }}
        #btnLoad {{
            background: {T['btn_load_bg']};
            color: {T['btn_load_txt']};
            border: 1px solid {T['btn_load_bdr']};
        }}
        #btnLoad:hover {{ background: {T['btn_load_bg_h']}; }}

        #btnRun {{
            background: {T['btn_run_bg']};
            color: {T['btn_run_txt']};
            border: none;
        }}
        #btnRun:hover {{ background: {T['btn_run_bg_h']}; }}
        #btnRun:disabled {{
            background: {T['btn_run_dis_bg']};
            color: {T['btn_run_dis_t']};
        }}

        #btnSave {{
            background: {T['btn_save_bg']};
            color: {T['btn_save_txt']};
            border: 1px solid {T['btn_save_bdr']};
        }}
        #btnSave:hover {{ background: {T['btn_save_bg_h']}; }}
        #btnSave:disabled {{
            background: {T['btn_save_dis_bg']};
            color: {T['btn_save_dis_t']};
        }}

        /* Model Combo */
        #modelCombo {{
            background: {T['bg_combo']};
            color: {T['accent_blue']};
            border: 1px solid {T['border_combo']};
            border-radius: 5px;
            padding: 4px 28px 4px 10px;
            font-size: 12px;
            font-weight: 600;
        }}

        #modelCombo::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: 1px solid {T['border_combo']};
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
            background: transparent;
        }}

        #modelCombo::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {T['accent_blue']};
            margin-right: 7px;
        }}
        QComboBox QAbstractItemView,
        QListView#modelComboView {{
            background: {T['bg_combo']};
            color: {T['accent_blue']};
            border: 1px solid {T['border_combo']};
            selection-background-color: {T['btn_load_bg_h']};
            selection-color: {T['accent_green']};
            outline: 0px;
            padding: 2px;
            font-size: 12px;
            font-weight: 600;
        }}
        QListView#modelComboView::item {{
            min-height: 24px;
            padding-left: 8px;
            padding-right: 8px;
            background: transparent;
        }}
        QListView#modelComboView::item:hover {{
            background: {T['btn_load_bg_h']};
            color: {T['accent_green']};
        }}
        QListView#modelComboView::item:selected {{
            background: {T['btn_load_bg_h']};
            color: {T['accent_green']};
        }}

        /* Legend */
        #legend {{
            background: {T['bg_legend']};
            border: 1px solid {T['border_main']};
            border-radius: 6px;
        }}
        #legTxt {{
            font-size: 11px;
            color: {T['txt_leg']};
        }}
        #hintLbl {{
            font-size: 11px;
            color: {T['txt_hint']};
            padding: 2px 4px;
        }}

        /* Section labels */
        #sectionLbl {{
            font-size: 10px;
            font-weight: 700;
            color: {T['txt_secondary']};
            letter-spacing: 1.2px;
            padding: 2px 0;
        }}

        /* Stat cards */
        #statCard {{
            background: {T['bg_card']};
            border: 1px solid {T['border_main']};
            border-radius: 8px;
        }}
        #cardLbl {{
            font-size: 11px;
            color: {T['txt_secondary']};
            font-weight: 500;
        }}

        /* Results table */
        #resTable {{
            background: {T['bg_table']};
            alternate-background-color: {T['bg_table_alt']};
            color: {T['txt_primary']};
            gridline-color: {T['border_main']};
            border: 1px solid {T['border_main']};
            border-radius: 6px;
            font-size: 12px;
        }}
        #resTable QHeaderView::section {{
            background: {T['bg_panel_r']};
            color: {T['txt_table_hdr']};
            border: none;
            border-bottom: 2px solid {T['border_main']};
            padding: 6px 8px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
        }}
        #resTable::item:selected {{
            background: {T['btn_load_bg']};
            color: {T['txt_primary']};
        }}
        #resTable::item {{
            padding: 4px 0;
        }}

        /* Info label */
        #infoLbl {{
            color: {T['txt_infolbl']};
            font-size: 10px;
        }}

        /* Status bar */
        #statusBar {{
            background: {T['bg_statusbar']};
            border-top: 1px solid {T['border_main']};
        }}
        #statusDot {{
            color: {T['accent_green']};
            font-size: 10px;
        }}
        #statusTxt {{
            color: {T['txt_statusbar']};
            font-size: 11px;
        }}
        #modelTxt {{
            color: {T['txt_model']};
            font-size: 11px;
        }}

        /* Tooltip card */
        #tipCard {{
            background: #111c14;
            border: 1px solid #2a4a32;
            border-radius: 8px;
        }}
        #tipCard QLabel {{
            background: transparent;
            color: #c8dcc8;
        }}
        #tipLabel {{
            font-size: 11px;
            color: #6a9070;
        }}
        #tipVal {{
            font-size: 12px;
            color: #d8ead8;
            font-weight: 600;
        }}
        #tipSep {{
            color: #2a4a32;
        }}

        /* Scrollbar */
        QScrollBar:vertical {{
            background: {T['scroll_bg']};
            width: 7px;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {T['scroll_handle']};
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{ height: 0; }}

        QSplitter::handle {{ background: {T['splitter']}; }}
        """)

    # MODEL
    def _load_model(self, name):
        fname = MODEL_OPTIONS.get(name)
        if not fname:
            return
        path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(path):
            self.status_lbl.setText(f"Weights not found: {path}")
            self.model_lbl.setText("Model: NOT LOADED")
            self.model = None
            return
        try:
            self.status_lbl.setText(f"Loading {name}…")
            QApplication.processEvents()
            self.model      = YOLO(path)
            self.model_name = name
            self.status_lbl.setText(f"Model ready — {name}")
            self.model_lbl.setText(
                f"Model: {fname}  |  Conf: {CONF_THRESHOLD}  |  IOU: {IOU_THRESHOLD}"
            )
            if self.current_img is not None:
                self.btn_run.setEnabled(True)
        except Exception as ex:
            self.status_lbl.setText(f"Error loading model: {ex}")
            self.model = None

    # IMAGE LOAD
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image",
            os.path.join(BASE_DIR, "dataset", "test_images"),
            "Images (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{path}")
            return

        self.current_img  = img
        self.current_path = path
        self.results      = []
        self.pipes        = []
        self.clean_img    = None

        pix = self._cv_to_pix(img)
        self.img_lbl.load_results([], [], pix)
        self.img_lbl.setText("")

        fname = os.path.basename(path)
        h, w  = img.shape[:2]
        self.info_lbl.setText(f"File: {fname}   ·   {w} × {h} px   ·   {path}")
        self.status_lbl.setText(f"Image loaded — {fname}")
        self.btn_run.setEnabled(self.model is not None)
        self.btn_save.setEnabled(False)
        self._reset_stats()

    def _cv_to_pix(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qi)

    # RUN
    def _run(self):
        if self.model is None or self.current_img is None:
            return
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.status_lbl.setText("Running detection and height estimation…")
        QApplication.processEvents()

        self.worker = DetectionWorker(
            self.model, self.current_img.copy(), self.cam, self.dist
        )
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, results, pipes, clean_img):
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_save.setEnabled(True)

        self.results   = results
        self.pipes     = pipes
        self.clean_img = clean_img

        pix = self._cv_to_pix(clean_img)
        self.img_lbl.load_results(results, pipes, pix)

        self._update_stats(results, pipes)
        self._fill_table(results)

        m = sum(1 for r in results if r["status"] == "MEASURED")
        self.status_lbl.setText(
            f"Done  —  {len(results)} tree(s) detected  ·  "
            f"{m} height(s) estimated  ·  {len(pipes)} pipe(s) found"
        )

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.status_lbl.setText(f"✗  Error: {msg}")
        QMessageBox.critical(self, "Detection Error", msg)

    # HOVER CALLBACK
    def _on_hover(self, idx):
        if idx < 0:
            self.table.clearSelection()
        else:
            self.table.selectRow(idx)
            self.table.scrollToItem(self.table.item(idx, 0))

    # STATS & TABLE
    def _reset_stats(self):
        for card in self.stat_cards.values():
            card.set_value("—")
        self.table.setRowCount(0)

    def _update_stats(self, results, pipes):
        measured = sum(1 for r in results if r["status"] in ("MEASURED", "HEIGHT_SUSPECT"))
        far      = sum(1 for r in results if r["status"] in ("TOO_FAR", "NO_PIPE"))
        partial  = sum(1 for r in results if r["status"] in ("PARTIAL", "SHORT_BOX"))
        self.stat_cards["total"].set_value(len(results))
        self.stat_cards["measured"].set_value(measured)
        self.stat_cards["far"].set_value(far)
        self.stat_cards["partial"].set_value(partial)
        self.stat_cards["pipes"].set_value(len(pipes))

    def _fill_table(self, results):
        T = self.T
        cmap = dict(
            MEASURED=T["QT_GREEN"],
            TOO_FAR=T["QT_ORANGE"],
            PARTIAL=T["QT_BLUE"],
            NO_PIPE=T["QT_ORANGE"],
            SHORT_BOX=T["QT_BLUE"],
            HEIGHT_SUSPECT=T["QT_ORANGE"]
        )
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            ht_str = f"{r['height_m']:.2f}" if r["height_m"] is not None else "—"
            items  = [r["tree_id"], r["status"], ht_str, str(r["conf"])]
            color  = QColor(cmap.get(r["status"], T["QT_ORANGE"]))
            for col, txt in enumerate(items):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col in (0, 1):
                    item.setForeground(color)
                if col == 2 and r["height_m"] is not None:
                    item.setForeground(QColor(T["QT_GREEN"]))
                self.table.setItem(row, col, item)

    # SAVE
    def _save(self):
        if self.clean_img is None:
            return
        annotated = draw_boxes_cv(self.clean_img, self.results, self.pipes)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.splitext(os.path.basename(self.current_path))[0]
        model_tag = (self.model_name or "NO_MODEL").strip().replace(" ", "_")
        out = os.path.join(OUTPUT_DIR, f"{fname}_{model_tag}_result_{ts}.jpg")
        cv2.imwrite(out, annotated)
        self.status_lbl.setText(f"✓  Saved [{model_tag}]: {out}")
        QMessageBox.information(
            self, "Result Saved",
            f"Annotated image saved using {model_tag}:\n{out}"
        )


# ENTRY POINT
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MangroveGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()