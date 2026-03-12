import json
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
from psd_tools import PSDImage

class PsdWarpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python PSD Warp Engine Pro")
        self.root.geometry("900x700")  # Made the window much bigger
        
        self.psd_background = None
        self.psd_json = None
        self.user_img = None
        self.preview_photo = None
        
        # --- UI STYLING ---
        self.bg_color = "#f3f4f6"
        self.accent_color = "#6366f1"
        self.root.configure(bg=self.bg_color)

        # Main Title
        tk.Label(root, text="Mockup Warp Studio", font=('Arial', 24, 'bold'), 
                 bg=self.bg_color, fg="#1f2937").pack(pady=20)

        # --- CONTROL PANEL ---
        ctrl_frame = tk.Frame(root, bg=self.bg_color)
        ctrl_frame.pack(pady=10, padx=20, fill='x')

        # File Selection & Status
        self.status_labels = {}
        self.create_file_row(ctrl_frame, "1. PSD Template:", self.load_psd, 0)
        self.create_file_row(ctrl_frame, "2. Warp JSON:", self.load_json, 1)
        self.create_file_row(ctrl_frame, "3. User Image:", self.load_image, 2)

        # Process Button
        self.proc_btn = tk.Button(root, text="GENERATE FINAL MOCKUP", command=self.process_warp, 
                                 bg=self.accent_color, fg="white", font=('Arial', 14, 'bold'),
                                 height=2, cursor="hand2")
        self.proc_btn.pack(pady=20, padx=50, fill='x')

        # --- PREVIEW AREA ---
        tk.Label(root, text="Preview Window", font=('Arial', 10, 'italic'), bg=self.bg_color).pack()
        self.preview_canvas = tk.Canvas(root, width=800, height=400, bg="#e5e7eb", highlightthickness=1)
        self.preview_canvas.pack(pady=10, expand=True)

    def create_file_row(self, parent, label_text, cmd, row):
        tk.Label(parent, text=label_text, font=('Arial', 11), bg=self.bg_color, width=15, anchor="w").grid(row=row, column=0, pady=5)
        tk.Button(parent, text="Choose File", command=cmd, width=15).grid(row=row, column=1, padx=10)
        
        status = tk.Label(parent, text="Not Loaded", font=('Arial', 10), fg="#ef4444", bg=self.bg_color)
        status.grid(row=row, column=2, sticky="w")
        self.status_labels[label_text] = status

    def update_status(self, key, text, success=True):
        color = "#10b981" if success else "#ef4444"
        self.status_labels[key].config(text=text, fg=color)

    def load_psd(self):
        path = filedialog.askopenfilename(filetypes=[("PSD", "*.psd *.psb")])
        if path:
            psd = PSDImage.open(path)
            self.psd_background = psd.composite()
            self.update_status("1. PSD Template:", f"Loaded: {path.split('/')[-1]}")
            self.update_preview(self.psd_background)

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f:
                self.psd_json = json.load(f)
            self.update_status("2. Warp JSON:", f"Loaded: {path.split('/')[-1]}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.user_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.update_status("3. User Image:", f"Loaded: {path.split('/')[-1]}")

    def update_preview(self, pil_img):
        # Resize for display while maintaining aspect ratio
        display_w, display_h = 800, 400
        img_w, img_h = pil_img.size
        ratio = min(display_w/img_w, display_h/img_h)
        new_size = (int(img_w*ratio), int(img_h*ratio))
        
        preview_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(display_w//2, display_h//2, image=self.preview_photo)

    def bezier_calc(self, u, v, h_pts, v_pts):
        def B(i, t): return [(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3][i]
        x, y = 0, 0
        for i in range(4):
            for j in range(4):
                coeff = B(i, v) * B(j, u)
                x += h_pts[i * 4 + j] * coeff
                y += v_pts[i * 4 + j] * coeff
        return x, y

    def process_warp(self):
        if not all([self.psd_background, self.psd_json, self.user_img is not None]):
            messagebox.showwarning("Incomplete Data", "Please load all 3 files before processing.")
            return

        canvas_w, canvas_h = self.psd_background.size
        try:
            layers = self.psd_json.get('children', [])
            warp_layer = next(l for l in layers if 'placedLayer' in l and 'warp' in l['placedLayer'])
            mesh = warp_layer['placedLayer']['warp']['customEnvelopeWarp']['meshPoints']
            h_pts = next(m['values'] for m in mesh if m['type'] == 'horizontal')
            v_pts = next(m['values'] for m in mesh if m['type'] == 'vertical')
        except:
            messagebox.showerror("Error", "Could not find warp mesh in JSON.")
            return

        # Warp Math
        subdiv = 40
        grid_u = np.linspace(0, 1, subdiv)
        grid_v = np.linspace(0, 1, subdiv)
        dst_pts = np.zeros((subdiv, subdiv, 2), dtype=np.float32)
        for i, v in enumerate(grid_v):
            for j, u in enumerate(grid_u):
                dst_pts[i, j] = self.bezier_calc(u, v, h_pts, v_pts)

        map_x = np.full((canvas_h, canvas_w), -1, dtype=np.float32)
        map_y = np.full((canvas_h, canvas_w), -1, dtype=np.float32)
        img_h, img_w = self.user_img.shape[:2]

        for i in range(subdiv - 1):
            for j in range(subdiv - 1):
                src_q = np.array([
                    [j*img_w/subdiv, i*img_h/subdiv], 
                    [(j+1)*img_w/subdiv, i*img_h/subdiv], 
                    [(j+1)*img_w/subdiv, (i+1)*img_h/subdiv], 
                    [j*img_w/subdiv, (i+1)*img_h/subdiv]
                ], dtype=np.float32)
                
                dst_q = np.array([
                    dst_pts[i, j], 
                    dst_pts[i, j+1], 
                    dst_pts[i+1, j+1], 
                    dst_pts[i+1, j]
                ], dtype=np.float32)

                H, _ = cv2.findHomography(dst_q, src_q)
                
                # CORRECTED AXIS LOGIC HERE
                min_c = np.floor(np.min(dst_q, axis=0)).astype(int)
                max_c = np.ceil(np.max(dst_q, axis=0)).astype(int)
                
                min_x, min_y = max(0, min_c[0]), max(0, min_c[1])
                max_x, max_y = min(canvas_w, max_c[0]), min(canvas_h, max_c[1])
                
                for py in range(min_y, max_y):
                    for px in range(min_x, max_x):
                        v_dst = np.array([px, py, 1.0])
                        v_src = H @ v_dst
                        v_src /= v_src[2]
                        map_x[py, px], map_y[py, px] = v_src[0], v_src[1]

        warped = cv2.remap(self.user_img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        final_pil = self.psd_background.copy().convert("RGBA")
        final_pil.alpha_composite(Image.fromarray(warped))
        
        self.update_preview(final_pil)
        final_pil.save("output_combined.png")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = PsdWarpApp(root)
    root.mainloop()