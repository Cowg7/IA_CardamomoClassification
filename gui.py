
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import numpy as np
import logging
from queue import Queue, Empty
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedGUI:
    def __init__(self, camera_stream=None, arduino_controller=None, db_handler=None):
        """
        GUI optimizada con visión en tiempo real y estadísticas

        Args:
            camera_stream: Instancia de OptimizedCameraStream
            arduino_controller: Instancia de ArduinoController  
            db_handler: Instancia de OptimizedDBHandler
        """
        self.camera_stream = camera_stream
        self.arduino = arduino_controller
        self.db = db_handler

        # Crear raíz de tkinter ANTES de usar StringVar
        self.root = tk.Tk()
        self.root.title("Clasificador de Cardamomo por Color")
        self.root.geometry("1200x720")

        # Variables de control
        self.var_class = tk.StringVar(self.root)
        self.is_running = False
        self.update_thread = None
        self.stats_update_thread = None

        # Colas para comunicación thread-safe
        self.gui_update_queue = Queue(maxsize=50)
        self.stats_queue = Queue(maxsize=10)

        # Variables de estado
        self.current_frame = None
        self.current_result = None
        self.last_update_time = time.time()
        self.fps_gui = 0.0
        self.frame_count = 0

        # Estadísticas acumuladas
        self.session_stats = {
            'verde': 0,
            'amarillo': 0,
            'cafe': 0,
            'total': 0
        }

        self._setup_ui()

    def _setup_ui(self):
        frame_main = ttk.Frame(self.root)
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Etiqueta de clase detectada
        label_result = ttk.Label(frame_main, text="Clase Detectada:", font=("Segoe UI", 14))
        label_result.pack(anchor=tk.W)

        value_result = ttk.Label(frame_main, textvariable=self.var_class, font=("Segoe UI", 18, "bold"), foreground="green")
        value_result.pack(anchor=tk.W)

        # Mostrar imagen de cámara
        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

    def update_frame(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
        except Exception as e:
            logger.error(f"❌ Error actualizando imagen: {e}")

    def start_gui(self):
        self.root.mainloop()
