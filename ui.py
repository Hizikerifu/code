import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import paddle
import numpy as np
from main import FireDetectionModel
import threading
import time

class DetectionUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fire Detection System")
        self.root.geometry("1200x800")
        
        self.model = FireDetectionModel()
        self.model.load_dict(paddle.load('best_model.pdparams'))
        self.model.eval()
        
        self.setup_ui()
        
    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.display_label = ttk.Label(self.left_frame)
        self.display_label.pack(fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.main_frame, width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        ttk.Style().configure('Custom.TButton', padding=10)
        
        self.image_btn = ttk.Button(self.right_frame, text="Select Image", 
                                  command=self.select_image, style='Custom.TButton')
        self.image_btn.pack(fill=tk.X, pady=5)
        
        self.video_btn = ttk.Button(self.right_frame, text="Select Video", 
                                  command=self.select_video, style='Custom.TButton')
        self.video_btn.pack(fill=tk.X, pady=5)
        
        self.camera_btn = ttk.Button(self.right_frame, text="Open Camera", 
                                   command=self.toggle_camera, style='Custom.TButton')
        self.camera_btn.pack(fill=tk.X, pady=5)
        
        self.detect_btn = ttk.Button(self.right_frame, text="Detect", 
                                   command=self.start_detection, style='Custom.TButton')
        self.detect_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = ttk.Button(self.right_frame, text="Stop", 
                                 command=self.stop_detection, style='Custom.TButton')
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        self.conf_frame = ttk.Frame(self.right_frame)
        self.conf_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.conf_scale = ttk.Scale(self.conf_frame, from_=0, to=1, orient=tk.HORIZONTAL)
        self.conf_scale.set(0.5)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.log_text = tk.Text(self.right_frame, height=10)
        self.log_text.pack(fill=tk.X, pady=5)
        
        self.is_running = False
        self.cap = None
        
    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if path:
            self.current_image = cv2.imread(path)
            self.display_image(self.current_image)  # 只显示原图
            self.log_text.insert(tk.END, f"Image loaded: {path}\n")
            self.log_text.see(tk.END)
            
    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.stop_detection()
            self.cap = cv2.VideoCapture(path)
            self.current_video_path = path
            self.log_text.insert(tk.END, f"Video loaded: {path}\n")
            self.log_text.see(tk.END)
            
    def toggle_camera(self):
        if self.is_running:
            self.stop_detection()
        else:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.log_text.insert(tk.END, "Camera opened\n")
                self.log_text.see(tk.END)
                ret, frame = self.cap.read()
                if ret:
                    self.display_image(frame)  # 显示摄像头原始画面
            else:
                self.log_text.insert(tk.END, "Failed to open camera\n")
                self.log_text.see(tk.END)
            
    def stop_detection(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            # 如果是视频文件,显示第一帧
            if hasattr(self, 'current_video_path'):
                cap = cv2.VideoCapture(self.current_video_path)
                ret, frame = cap.read()
                if ret:
                    self.display_image(frame)
                cap.release()
            
    def process_image(self, image):
        h, w = image.shape[:2]
        processed = cv2.resize(image, (416, 416))
        processed = processed / 255.0
        processed = processed.transpose(2, 0, 1)
        processed = processed[np.newaxis, ...]
        processed = paddle.to_tensor(processed, dtype='float32')
        
        with paddle.no_grad():
            predictions = self.model(processed)
            boxes = self.decode_predictions(predictions, self.conf_scale.get())
        
        for box in boxes[0]:
            x1, y1, x2, y2 = box
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 0, 255), 2)
        
        self.display_image(image)
        
    def video_detection_loop(self):
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if hasattr(self, 'current_video_path'):  
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:  
                    break
                
            original_frame = frame.copy()
            
            h, w = frame.shape[:2]
            processed = cv2.resize(frame, (416, 416))
            processed = processed / 255.0
            processed = processed.transpose(2, 0, 1)
            processed = processed[np.newaxis, ...]
            processed = paddle.to_tensor(processed, dtype='float32')
            
            with paddle.no_grad():
                predictions = self.model(processed)
                predictions_np = predictions.numpy()  # 转换为numpy数组
                boxes = self.decode_predictions(predictions, self.conf_scale.get())
            
            for box in boxes[0]:
                x1, y1, x2, y2 = box
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                             (0, 0, 255), 2)
                
                # 使用numpy数组获取置信度
                conf = 1 / (1 + np.exp(-predictions_np[0, 0, int(y1*13/h), int(x1*13/w)]))
                label = f"Fire: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.display_image(frame)
            time.sleep(0.03)
            
            if len(boxes[0]) > 0:
                self.log_text.insert(tk.END, f"Fire detected! Boxes: {len(boxes[0])}\n")
                self.log_text.see(tk.END)
        
        self.stop_detection()
        self.log_text.insert(tk.END, "Detection stopped\n")
        self.log_text.see(tk.END)
        
    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        display_w = self.display_label.winfo_width()
        display_h = self.display_label.winfo_height()
        
        if display_w > 0 and display_h > 0:
            scale = min(display_w/w, display_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            image = cv2.resize(image, (new_w, new_h))
        
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.display_label.configure(image=photo)
        self.display_label.image = photo
        
    def decode_predictions(self, predictions, conf_threshold=0.5):
        if isinstance(predictions, paddle.Tensor):
            predictions = predictions.numpy()
        
        predictions = np.copy(predictions)
        predictions[:, 0] = 1 / (1 + np.exp(-predictions[:, 0]))
        predictions[:, 1:5] = 1 / (1 + np.exp(-predictions[:, 1:5]))
        
        batch_size = predictions.shape[0]
        boxes_list = []
        
        for b in range(batch_size):
            boxes = []
            pred = predictions[b]
            
            for i in range(13):
                for j in range(13):
                    confidence = pred[0, i, j]
                    if confidence > conf_threshold:
                        x = pred[1, i, j]
                        y = pred[2, i, j]
                        w = pred[3, i, j]
                        h = pred[4, i, j]
                        
                        x1 = max(0, min(1, x - w/2))
                        y1 = max(0, min(1, y - h/2))
                        x2 = max(0, min(1, x + w/2))
                        y2 = max(0, min(1, y + h/2))
                        
                        boxes.append([x1, y1, x2, y2])
            
            boxes_list.append(np.array(boxes) if boxes else np.zeros((0, 4)))
        
        return boxes_list
    
    def start_detection(self):
        if hasattr(self, 'current_image'):
            # 图片检测
            self.process_image(self.current_image.copy())
            self.log_text.insert(tk.END, "Detection completed\n")
            self.log_text.see(tk.END)
        elif self.cap is not None:
            # 视频检测
            self.is_running = True
            threading.Thread(target=self.video_detection_loop).start()
            self.log_text.insert(tk.END, "Detection started\n")
            self.log_text.see(tk.END)
        else:
            self.log_text.insert(tk.END, "Please select an image or video first\n")
            self.log_text.see(tk.END)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DetectionUI()
    app.run()