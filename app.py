import os
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from modules import module1_enhance as M1
from modules import module2_segment as M2
from modules import module3_transform as M3

def cv_to_pil(img_bgr):
    if img_bgr is None:
        return None
    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Toolbox (Tkinter)")
        self.geometry("1200x700")

        self.original = None
        self.reference = None
        self.result = None

        # NEW: fixed preview box for both panes (prevents “getting bigger”)
        self.display_box = (560, 560)  # (width, height) in pixels for preview

        self._build_layout()

    def _build_layout(self):
        top = Frame(self)
        top.pack(side=TOP, fill=X)

        Button(top, text="Load Image", command=self.load_image).pack(side=LEFT, padx=5, pady=5)
        Button(top, text="Load Reference (for Hist Match)", command=self.load_reference).pack(side=LEFT, padx=5, pady=5)
        Button(top, text="Save Result", command=self.save_result).pack(side=LEFT, padx=5, pady=5)
        Button(top, text="Clear Result", command=self.clear_result).pack(side=LEFT, padx=5, pady=5)

        self.status = StringVar(value="Ready")
        Label(top, textvariable=self.status).pack(side=RIGHT, padx=10)

        panels = Frame(self)
        panels.pack(side=TOP, fill=BOTH, expand=True)

        self.left_label = Label(panels, text="Original")
        self.left_label.pack(side=LEFT, expand=True, fill=BOTH)

        self.right_label = Label(panels, text="Result")
        self.right_label.pack(side=RIGHT, expand=True, fill=BOTH)

        tabs = ttk.Notebook(self)
        tabs.pack(side=BOTTOM, fill=X)

        tabs.add(self._build_module1(tabs), text="Module 1 — Enhancement")
        tabs.add(self._build_module2(tabs), text="Module 2 — Segmentation & Edges")
        tabs.add(self._build_module3(tabs), text="Module 3 — Geometric & Interp")

    # Image I/O
    def load_image(self):
        path = filedialog.askopenfilename(title="Choose image",
                                          filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Failed to read image")
            return
        self.original = img
        self.result = None
        self._refresh_panels()
        self.status.set(f"Loaded: {os.path.basename(path)}")

    def load_reference(self):
        path = filedialog.askopenfilename(title="Choose reference",
                                          filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Failed to read reference image")
            return
        self.reference = img
        self.status.set(f"Reference: {os.path.basename(path)}")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Info", "No result to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG","*.png"),("JPG","*.jpg;*.jpeg"),("BMP","*.bmp")])
        if not path:
            return
        cv2.imwrite(path, self.result if self.result.ndim == 3 else self.result)
        self.status.set(f"Saved: {os.path.basename(path)}")

    def clear_result(self):
        self.result = None
        self._refresh_panels()
        self.status.set("Result cleared")

    def _refresh_panels(self):
        # Left (Original) — always fit to fixed display box
        if self.original is not None:
            pil = cv_to_pil(self.original)
            pil = self._fit_to_panel(pil, self.left_label)
            self.left_imgtk = ImageTk.PhotoImage(pil)
            self.left_label.configure(image=self.left_imgtk)
        else:
            self.left_label.configure(image="", text="Original")

        # Right (Result) — always fit to fixed display box
        if self.result is not None:
            if self.result.ndim == 2:
                disp = cv2.cvtColor(self.result, cv2.COLOR_GRAY2BGR)
            else:
                disp = self.result
            pil = cv_to_pil(disp)
            pil = self._fit_to_panel(pil, self.right_label)
            self.right_imgtk = ImageTk.PhotoImage(pil)
            self.right_label.configure(image=self.right_imgtk)
        else:
            self.right_label.configure(image="", text="Result")

    def _fit_to_panel(self, pil_img, label):
        """
        Resize PIL image to fit inside a fixed box (self.display_box),
        preserving aspect ratio. Ignores the label's current size so the
        preview never "grows" between operations.
        """
        bw, bh = self.display_box
        w, h = pil_img.size
        scale = min(bw / max(1, w), bh / max(1, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return pil_img.copy().resize((new_w, new_h), Image.LANCZOS)

    def _require_image(self):
        if self.original is None:
            messagebox.showwarning("Missing Image", "Load an image first.")
            return False
        return True

    # Module 1
    def _build_module1(self, parent):
        frm = Frame(parent)

        row1 = Frame(frm); row1.pack(anchor="w", fill=X, pady=3)
        Label(row1, text="Histogram: ").pack(side=LEFT)
        Button(row1, text="Equalization", command=self._m1_hist_eq).pack(side=LEFT, padx=3)
        Button(row1, text="Match (needs reference)", command=self._m1_hist_match).pack(side=LEFT, padx=3)

        row2 = Frame(frm); row2.pack(anchor="w", fill=X, pady=3)
        Label(row2, text="Brightness / Contrast: ").pack(side=LEFT)
        self.alpha = DoubleVar(value=1.0)
        self.beta = IntVar(value=0)
        Spinbox(row2, from_=0.2, to=3.0, increment=0.1, textvariable=self.alpha, width=6).pack(side=LEFT)
        Spinbox(row2, from_=-100, to=100, increment=5, textvariable=self.beta, width=6).pack(side=LEFT, padx=5)
        Button(row2, text="Apply", command=self._m1_bright_contrast).pack(side=LEFT, padx=3)

        row3 = Frame(frm); row3.pack(anchor="w", fill=X, pady=3)
        Label(row3, text="Smoothing: ").pack(side=LEFT)
        self.ksize = IntVar(value=3)
        Spinbox(row3, from_=3, to=15, increment=2, textvariable=self.ksize, width=6).pack(side=LEFT)
        Button(row3, text="Mean", command=self._m1_mean).pack(side=LEFT, padx=3)
        Button(row3, text="Median", command=self._m1_median).pack(side=LEFT, padx=3)

        row4 = Frame(frm); row4.pack(anchor="w", fill=X, pady=3)
        Label(row4, text="Sharpen: ").pack(side=LEFT)
        Button(row4, text="Laplacian", command=self._m1_laplacian).pack(side=LEFT, padx=3)
        Button(row4, text="High-pass", command=self._m1_highpass).pack(side=LEFT, padx=3)

        row5 = Frame(frm); row5.pack(anchor="w", fill=X, pady=3)
        Label(row5, text="Frequency Domain (Ideal): d0 / band: ").pack(side=LEFT)
        self.d0 = IntVar(value=30)
        self.dlow = IntVar(value=10)
        self.dhigh = IntVar(value=60)
        Spinbox(row5, from_=5, to=200, textvariable=self.d0, width=6).pack(side=LEFT)
        Spinbox(row5, from_=1, to=200, textvariable=self.dlow, width=6).pack(side=LEFT, padx=4)
        Spinbox(row5, from_=5, to=300, textvariable=self.dhigh, width=6).pack(side=LEFT, padx=4)
        Button(row5, text="Low-pass", command=self._m1_lowpass).pack(side=LEFT, padx=3)
        Button(row5, text="High-pass", command=self._m1_highpass_freq).pack(side=LEFT, padx=3)
        Button(row5, text="Band-pass", command=self._m1_bandpass).pack(side=LEFT, padx=3)

        return frm

    def _m1_hist_eq(self):
        if not self._require_image(): return
        self.result = M1.hist_equalize(self.original.copy())
        self._refresh_panels()

    def _m1_hist_match(self):
        if not self._require_image(): return
        if self.reference is None:
            messagebox.showwarning("Reference missing", "Load a reference image first.")
            return
        self.result = M1.hist_match(self.original.copy(), self.reference.copy())
        self._refresh_panels()

    def _m1_bright_contrast(self):
        if not self._require_image(): return
        self.result = M1.adjust_brightness_contrast(self.original.copy(), float(self.alpha.get()), int(self.beta.get()))
        self._refresh_panels()

    def _m1_mean(self):
        if not self._require_image(): return
        self.result = M1.mean_filter(self.original.copy(), int(self.ksize.get()))
        self._refresh_panels()

    def _m1_median(self):
        if not self._require_image():
            return
        k = int(self.ksize.get())
        # make sure the kernel size is odd and >= 3
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        self.result = M1.median_filter(self.original.copy(), k)
        self._refresh_panels()


    def _m1_laplacian(self):
        if not self._require_image(): return
        self.result = M1.laplacian_sharpen(self.original.copy())
        self._refresh_panels()

    def _m1_highpass(self):
        if not self._require_image(): return
        self.result = M1.high_pass_filter(self.original.copy())
        self._refresh_panels()

    def _m1_lowpass(self):
        if not self._require_image(): return
        self.result = M1.freq_lowpass(self.original.copy(), int(self.d0.get()))
        self._refresh_panels()

    def _m1_highpass_freq(self):
        if not self._require_image(): return
        self.result = M1.freq_highpass(self.original.copy(), int(self.d0.get()))
        self._refresh_panels()

    def _m1_bandpass(self):
        if not self._require_image(): return
        self.result = M1.freq_bandpass(self.original.copy(), int(self.dlow.get()), int(self.dhigh.get()))
        self._refresh_panels()

    # Module 2
    def _build_module2(self, parent):
        frm = Frame(parent)

        r1 = Frame(frm); r1.pack(anchor="w", fill=X, pady=3)
        Label(r1, text="Edges: ").pack(side=LEFT)
        Button(r1, text="Sobel", command=self._m2_sobel).pack(side=LEFT, padx=3)
        Button(r1, text="Prewitt", command=self._m2_prewitt).pack(side=LEFT, padx=3)
        self.canny_t1 = IntVar(value=100)
        self.canny_t2 = IntVar(value=200)
        Label(r1, text="Canny t1/t2:").pack(side=LEFT, padx=5)
        Spinbox(r1, from_=0, to=500, textvariable=self.canny_t1, width=6).pack(side=LEFT)
        Spinbox(r1, from_=0, to=500, textvariable=self.canny_t2, width=6).pack(side=LEFT, padx=4)
        Button(r1, text="Canny", command=self._m2_canny).pack(side=LEFT, padx=3)

        r2 = Frame(frm); r2.pack(anchor="w", fill=X, pady=3)
        Button(r2, text="Otsu Threshold (binary)", command=self._m2_otsu).pack(side=LEFT, padx=3)
        Label(r2, text="Morph (op, ksize, iter):").pack(side=LEFT, padx=6)
        self.morph_op = StringVar(value="dilate")
        self.morph_ks = IntVar(value=3)
        self.morph_it = IntVar(value=1)
        ttk.Combobox(r2, values=["dilate","erode","open","close"], textvariable=self.morph_op, width=8).pack(side=LEFT)
        Spinbox(r2, from_=1, to=31, increment=2, textvariable=self.morph_ks, width=6).pack(side=LEFT, padx=4)
        Spinbox(r2, from_=1, to=10, textvariable=self.morph_it, width=6).pack(side=LEFT, padx=4)
        Button(r2, text="Apply", command=self._m2_morph).pack(side=LEFT, padx=3)

        r3 = Frame(frm); r3.pack(anchor="w", fill=X, pady=3)
        Label(r3, text="HSV segmentation [Hlow,Hhigh] & [S,V] min:").pack(side=LEFT)
        self.hlow = IntVar(value=35); self.hhigh = IntVar(value=85)
        self.smin = IntVar(value=50); self.vmin = IntVar(value=50)
        Spinbox(r3, from_=0, to=179, textvariable=self.hlow, width=5).pack(side=LEFT)
        Spinbox(r3, from_=0, to=179, textvariable=self.hhigh, width=5).pack(side=LEFT)
        Spinbox(r3, from_=0, to=255, textvariable=self.smin, width=5).pack(side=LEFT)
        Spinbox(r3, from_=0, to=255, textvariable=self.vmin, width=5).pack(side=LEFT)
        Button(r3, text="Segment", command=self._m2_hsv_seg).pack(side=LEFT, padx=3)

        return frm

    def _m2_sobel(self):
        if not self._require_image(): return
        self.result = M2.sobel_edges(self.original.copy())
        self._refresh_panels()

    def _m2_prewitt(self):
        if not self._require_image(): return
        self.result = M2.prewitt_edges(self.original.copy())
        self._refresh_panels()

    def _m2_canny(self):
        if not self._require_image(): return
        self.result = M2.canny_edges(self.original.copy(), int(self.canny_t1.get()), int(self.canny_t2.get()))
        self._refresh_panels()

    def _m2_otsu(self):
        if not self._require_image(): return
        self.result = M2.otsu_threshold(self.original.copy())
        self._refresh_panels()

    def _m2_morph(self):
        if not self._require_image(): return
        self.result = M2.morphology(self.original.copy(), self.morph_op.get(), int(self.morph_ks.get()), int(self.morph_it.get()))
        self._refresh_panels()

    def _m2_hsv_seg(self):
        if not self._require_image(): return
        lower = (int(self.hlow.get()), int(self.smin.get()), int(self.vmin.get()))
        upper = (int(self.hhigh.get()), 255, 255)
        self.result = M2.color_segmentation_hsv(self.original.copy(), lower, upper)
        self._refresh_panels()

    # Module 3
    def _build_module3(self, parent):
        frm = Frame(parent)

        r1 = Frame(frm); r1.pack(anchor="w", fill=X, pady=3)
        Label(r1, text="Translate (tx, ty):").pack(side=LEFT)
        self.tx = IntVar(value=50); self.ty = IntVar(value=30)
        Spinbox(r1, from_=-500, to=500, textvariable=self.tx, width=6).pack(side=LEFT)
        Spinbox(r1, from_=-500, to=500, textvariable=self.ty, width=6).pack(side=LEFT, padx=4)
        Button(r1, text="Apply", command=self._m3_translate).pack(side=LEFT, padx=3)

        Label(r1, text="   Scale (sx, sy):").pack(side=LEFT)
        self.sx = DoubleVar(value=1.5); self.sy = DoubleVar(value=1.5)
        Spinbox(r1, from_=0.1, to=4.0, increment=0.1, textvariable=self.sx, width=6).pack(side=LEFT)
        Spinbox(r1, from_=0.1, to=4.0, increment=0.1, textvariable=self.sy, width=6).pack(side=LEFT, padx=4)
        self.interp = StringVar(value="bilinear")
        ttk.Combobox(r1, values=["nearest","bilinear","bicubic"], textvariable=self.interp, width=8).pack(side=LEFT)
        Button(r1, text="Resize", command=self._m3_scale).pack(side=LEFT, padx=3)

        r2 = Frame(frm); r2.pack(anchor="w", fill=X, pady=3)
        Label(r2, text="Rotate (deg):").pack(side=LEFT)
        self.angle = DoubleVar(value=15.0)
        Spinbox(r2, from_=-180, to=180, increment=1, textvariable=self.angle, width=6).pack(side=LEFT)
        Button(r2, text="Rotate", command=self._m3_rotate).pack(side=LEFT, padx=3)

        r3 = Frame(frm); r3.pack(anchor="w", fill=X, pady=3)
        Label(r3, text="Radial correction k1:").pack(side=LEFT)
        self.k1 = DoubleVar(value=-1e-6)
        Entry(r3, textvariable=self.k1, width=10).pack(side=LEFT)
        Button(r3, text="Correct", command=self._m3_radial).pack(side=LEFT, padx=3)

        return frm

    def _m3_translate(self):
        if not self._require_image(): return
        self.result = M3.translate(self.original.copy(), int(self.tx.get()), int(self.ty.get()))
        self._refresh_panels()

    def _m3_scale(self):
        if not self._require_image(): return
        self.result = M3.scale(self.original.copy(), float(self.sx.get()), float(self.sy.get()), self.interp.get())
        self._refresh_panels()

    def _m3_rotate(self):
        if not self._require_image(): return
        self.result = M3.rotate(self.original.copy(), float(self.angle.get()), self.interp.get())
        self._refresh_panels()

    def _m3_radial(self):
        if not self._require_image(): return
        self.result = M3.radial_correction(self.original.copy(), float(self.k1.get()), 0.0)
        self._refresh_panels()

if __name__ == "__main__":
    App().mainloop()
