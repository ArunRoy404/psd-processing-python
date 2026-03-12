import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
from psd_tools import PSDImage

class PsdWarpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSD Multi-Layer Warp Engine")
        self.root.geometry("1000x850") # Slightly taller for progress bar
        
        self.psd_obj = None
        self.psd_json = None
        self.user_img = None
        self.preview_photo = None
        
        self.status_labels = {}
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f3f4f6")
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text="Layered Mockup Generator", font=('Arial', 20, 'bold'), bg="#f3f4f6").pack(pady=10)

        # Control Panel
        ctrl = tk.Frame(main_frame, bg="#f3f4f6")
        ctrl.pack(pady=10)

        self.create_row(ctrl, "PSD File:", self.load_psd, 0)
        self.create_row(ctrl, "Mesh JSON:", self.load_json, 1)
        self.create_row(ctrl, "User Image:", self.load_image, 2)

        # Progress Section
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=(20, 0), padx=50, fill='x')
        
        self.progress_label = tk.Label(main_frame, text="Ready", font=('Arial', 10), bg="#f3f4f6", fg="#4b5563")
        self.progress_label.pack(pady=5)

        tk.Button(main_frame, text="GENERATE MOCKUP", command=self.process_warp, 
                  bg="#6366f1", fg="white", font=('Arial', 12, 'bold'), height=2).pack(pady=10, padx=50, fill='x')

        self.preview_canvas = tk.Canvas(main_frame, width=800, height=450, bg="#d1d5db")
        self.preview_canvas.pack(pady=10)

    def create_row(self, parent, label, cmd, row):
        tk.Label(parent, text=label, width=15, anchor="w", bg="#f3f4f6").grid(row=row, column=0)
        tk.Button(parent, text="Browse", command=cmd, width=10).grid(row=row, column=1, padx=5, pady=5)
        status = tk.Label(parent, text="Missing", fg="red", bg="#f3f4f6")
        status.grid(row=row, column=2, padx=5)
        self.status_labels[label] = status

    def load_psd(self):
        path = filedialog.askopenfilename(filetypes=[("PSD", "*.psd *.psb")])
        if path:
            self.psd_obj = PSDImage.open(path)
            self.status_labels["PSD File:"].config(text="Loaded", fg="green")
            self.update_preview(self.psd_obj.composite())

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f:
                self.psd_json = json.load(f)
            self.status_labels["Mesh JSON:"].config(text="Loaded", fg="green")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.user_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.status_labels["User Image:"].config(text="Loaded", fg="green")

    def bezier_calc(self, u, v, h_pts, v_pts):
        def B(i, t): return [(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3][i]
        x, y = 0, 0
        for i in range(4):
            for j in range(4):
                coeff = B(i, v) * B(j, u)
                x += h_pts[i*4+j] * coeff
                y += v_pts[i*4+j] * coeff
        return x, y

    def get_psd_layer_by_name(self, name):
        for layer in self.psd_obj:
            if layer.name == name:
                return layer.composite()
        return None

    def process_warp(self):
        if not all([self.psd_obj, self.psd_json, self.user_img is not None]):
            messagebox.showerror("Error", "All files must be loaded!")
            return

        canvas_w, canvas_h = self.psd_obj.width, self.psd_obj.height
        final_image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        layers_data = self.psd_json.get('children', [])
        total_layers = len(layers_data)

        for idx, layer_data in enumerate(layers_data):
            name = layer_data.get('name', 'Unknown')
            
            # Update Progress UI
            percent = ((idx) / total_layers) * 100
            self.progress_var.set(percent)
            self.progress_label.config(text=f"Processing layer {idx+1}/{total_layers}: {name}")
            self.root.update_idletasks() # Force UI to update

            # 1. Warp Layer
            if 'placedLayer' in layer_data:
                warped_arr = self.run_warp_math(canvas_w, canvas_h, layer_data)
                warped_pil = Image.fromarray(warped_arr)
                final_image.alpha_composite(warped_pil)
            
            # 2. Standard Layer
            else:
                psd_layer_img = self.get_psd_layer_by_name(name)
                if psd_layer_img:
                    layer_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
                    layer_canvas.paste(psd_layer_img, (layer_data['left'], layer_data['top']))
                    final_image.alpha_composite(layer_canvas)

        # Final Wrap up
        self.progress_var.set(100)
        self.progress_label.config(text="Complete!")
        self.update_preview(final_image)
        final_image.save("final_mockup.png")

    def run_warp_math(self, canvas_w, canvas_h, layer_data):
        mesh = layer_data['placedLayer']['warp']['customEnvelopeWarp']['meshPoints']
        h_pts = next(m['values'] for m in mesh if m['type'] == 'horizontal')
        v_pts = next(m['values'] for m in mesh if m['type'] == 'vertical')
        
        subdiv = 40
        grid_u = np.linspace(0, 1, subdiv)
        grid_v = np.linspace(0, 1, subdiv)
        dst_pts = np.zeros((subdiv, subdiv, 2), dtype=np.float32)
        for i, v in enumerate(grid_v):
            for j, u in enumerate(grid_u):
                dst_pts[i, j] = self.bezier_calc(u, v, h_pts, v_pts)

        map_x, map_y = np.full((canvas_h, canvas_w), -1, dtype=np.float32), np.full((canvas_h, canvas_w), -1, dtype=np.float32)
        img_h, img_w = self.user_img.shape[:2]

        for i in range(subdiv - 1):
            for j in range(subdiv - 1):
                src_q = np.array([[j*img_w/subdiv, i*img_h/subdiv], [(j+1)*img_w/subdiv, i*img_h/subdiv], [(j+1)*img_w/subdiv, (i+1)*img_h/subdiv], [j*img_w/subdiv, (i+1)*img_h/subdiv]], dtype=np.float32)
                dst_q = np.array([dst_pts[i,j], dst_pts[i,j+1], dst_pts[i+1,j+1], dst_pts[i+1,j]], dtype=np.float32)
                H, _ = cv2.findHomography(dst_q, src_q)
                min_c, max_c = np.floor(np.min(dst_q, axis=0)).astype(int), np.ceil(np.max(dst_q, axis=0)).astype(int)
                for py in range(max(0, min_c[1]), min(canvas_h, max_c[1])):
                    for px in range(max(0, min_c[0]), min(canvas_w, max_c[0])):
                        v_src = H @ np.array([px, py, 1.0])
                        v_src /= v_src[2]
                        if 0 <= v_src[0] < img_w and 0 <= v_src[1] < img_h:
                            map_x[py, px], map_y[py, px] = v_src[0], v_src[1]

        return cv2.remap(self.user_img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    def update_preview(self, pil_img):
        img_w, img_h = pil_img.size
        ratio = min(800/img_w, 450/img_h)
        new_size = (int(img_w*ratio), int(img_h*ratio))
        preview_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(400, 225, image=self.preview_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = PsdWarpApp(root)
    root.mainloop()