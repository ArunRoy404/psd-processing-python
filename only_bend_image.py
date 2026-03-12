import json
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

class PsdWarpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python PSD Warp Engine")
        
        self.psd_json = None
        self.user_img = None
        
        # UI Setup
        self.label = tk.Label(root, text="Photoshop Warp Emulator", font=('Arial', 16))
        self.label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        # Changed px=5 to padx=5 below
        tk.Button(btn_frame, text="1. Load PSD JSON", command=self.load_json).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="2. Load User Image", command=self.load_image).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="3. Process & Save", command=self.process_warp, bg="#6366f1", fg="white").grid(row=0, column=2, padx=5)
        
    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            with open(path, 'r') as f:
                self.psd_json = json.load(f)
            messagebox.showinfo("Success", "JSON Loaded")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.user_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.user_img = cv2.cvtColor(self.user_img, cv2.COLOR_BGR2RGBA)
            messagebox.showinfo("Success", "Image Loaded")

    def bezier_calc(self, u, v, h_pts, v_pts):
        """ Cubic Bezier Surface Calculation """
        def B(i, t):
            if i == 0: return (1-t)**3
            if i == 1: return 3*t*(1-t)**2
            if i == 2: return 3*t**2*(1-t)
            if i == 3: return t**3
            return 0

        x, y = 0, 0
        for i in range(4):
            for j in range(4):
                coeff = B(i, v) * B(j, u)
                x += h_pts[i * 4 + j] * coeff
                y += v_pts[i * 4 + j] * coeff
        return x, y

    def process_warp(self):
        if not self.psd_json or self.user_img is None:
            messagebox.showerror("Error", "Please load both JSON and Image")
            return

        canvas_w = self.psd_json['width']
        canvas_h = self.psd_json['height']

        # 1. Extract Mesh Data
        try:
            # Finding the layer with the warp data
            layers = self.psd_json.get('children', [])
            warp_layer = next(l for l in layers if 'placedLayer' in l and 'warp' in l['placedLayer'])
            mesh = warp_layer['placedLayer']['warp']['customEnvelopeWarp']['meshPoints']
            h_pts = next(m['values'] for m in mesh if m['type'] == 'horizontal')
            v_pts = next(m['values'] for m in mesh if m['type'] == 'vertical')
        except StopIteration:
            messagebox.showerror("Error", "Could not find warp data in JSON")
            return

        # 2. Create the Warp Maps (Inverse Mapping)
        # We create a grid to interpolate the Bezier points
        subdiv = 60
        grid_u = np.linspace(0, 1, subdiv)
        grid_v = np.linspace(0, 1, subdiv)
        
        # Source points (normalized 0-1) to destination pixels
        dst_pts = np.zeros((subdiv, subdiv, 2), dtype=np.float32)
        for i, v in enumerate(grid_v):
            for j, u in enumerate(grid_u):
                dst_pts[i, j] = self.bezier_calc(u, v, h_pts, v_pts)

        # 3. Generate Remap Matrices
        # We map the user image (src) onto the canvas (dst)
        map_x = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        map_y = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # We need the source image coordinates
        img_h, img_w = self.user_img.shape[:2]
        
        # Interpolate the grid to every pixel using OpenCV
        # This creates a smooth mapping for the entire warp area
        src_grid_x, src_grid_y = np.meshgrid(np.linspace(0, img_w-1, subdiv), np.linspace(0, img_h-1, subdiv))
        
        # Flatten for mapping
        src_pts = np.stack([src_grid_x, src_grid_y], axis=-1).astype(np.float32)
        
        # Use thin plate spline or simple remap logic
        # For efficiency, we'll create a full-canvas remap
        # Initialize with out-of-bounds values
        map_x.fill(-1)
        map_y.fill(-1)

        # Create a mesh for the destination and map back to source
        for i in range(subdiv - 1):
            for j in range(subdiv - 1):
                # Define a small quad in the destination mesh
                src_quad = np.array([
                    [j * img_w/subdiv, i * img_h/subdiv],
                    [(j+1) * img_w/subdiv, i * img_h/subdiv],
                    [(j+1) * img_w/subdiv, (i+1) * img_h/subdiv],
                    [j * img_w/subdiv, (i+1) * img_h/subdiv]
                ], dtype=np.float32)

                dst_quad = np.array([
                    dst_pts[i, j],
                    dst_pts[i, j+1],
                    dst_pts[i+1, j+1],
                    dst_pts[i+1, j]
                ], dtype=np.float32)

                # Compute Homography for this small patch
                H, _ = cv2.findHomography(dst_quad, src_quad)
                
                # Mask out the bounding box of the destination quad
                min_x, min_y = np.floor(np.min(dst_quad, axis=0)).astype(int)
                max_x, max_y = np.ceil(np.max(dst_quad, axis=0)).astype(int)
                
                for py in range(max(0, min_y), min(canvas_h, max_y)):
                    for px in range(max(0, min_x), min(canvas_w, max_x)):
                        # Project canvas pixel back to user image pixel
                        v_dst = np.array([px, py, 1.0])
                        v_src = H @ v_dst
                        v_src /= v_src[2]
                        map_x[py, px] = v_src[0]
                        map_y[py, px] = v_src[1]

        # 4. Final Remap
        result = cv2.remap(self.user_img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        # 5. Save and Show
        out_path = "warped_output.png"
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        
        # Convert to PIL for showing in Tkinter if desired
        res_img = Image.fromarray(result)
        res_img.show()
        messagebox.showinfo("Done", f"Warped image saved as {out_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PsdWarpApp(root)
    root.mainloop()