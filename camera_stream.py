import cv2
import threading
import time
from queue import Queue
import numpy as np
from clasificador import predict_cardamomo
from arduino_controller import ArduinoController
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedCameraStream:
    def __init__(self, arduino_controller, db_handler):
        """
        Stream de cámara optimizado para RTX4050 y iPhone15 Pro via CamoStudio
        Configurado para detección en tiempo real de cardamomo por color
        """
        self.arduino = arduino_controller
        self.db = db_handler
        
        # Configuración de video optimizada para iPhone15 Pro
        self.target_fps = 30
        self.frame_width = 1920
        self.frame_height = 1080
        self.detection_area_size = 400  # Área central para detección
        
        # Control de threading y buffering
        self.frame_queue = Queue(maxsize=5)  # Buffer limitado para evitar lag
        self.result_queue = Queue(maxsize=10)
        self.capture_thread = None
        self.processing_thread = None
        self.display_thread = None
        
        # Flags de control
        self.is_running = False
        self.show_detection_area = True
        self.show_statistics = True
        
        # Estadísticas de rendimiento
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.total_detections = 0

        # Clasificación del frame
        try:
            clase, confianza = predict_cardamomo(frame)

            # Actualizar GUI si existe
            if hasattr(self, 'gui') and self.gui:
                self.gui.var_class.set(f"{clase} ({confianza:.2f})")
                self.gui.update_frame(frame)

            # Decisión basada en confianza
            if clase == "Verde" and confianza >= 0.85:
                self.arduino.enviar_comando("ACEPTAR")
            else:
                self.arduino.enviar_comando("RECHAZAR")

            # Guardar en la base de datos
            if self.db:
                self.db._update_daily_stats(clase, confianza)

        except Exception as e:
            logger.error(f"❌ Error en clasificación de frame: {e}")
        self.quality_accepted = 0
        
        # Variables para display
        self.last_result = None
        self.detection_history = []
    
    def detectar_camaras_disponibles(self, max_index=20):
        """
        Detectar cámaras disponibles incluyendo CamoStudio
        Optimizado para detectar iPhone via Camo
        """
        print("🔍 Buscando cámaras disponibles (incluyendo CamoStudio)...")
        camaras_disponibles = []
        
        # Detectar cámaras DirectShow (Windows)
        for i in range(max_index):
            # Probar con DirectShow primero (mejor para Camo)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Obtener nombre de la cámara si es posible
                backend_name = cap.getBackendName()
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✅ Cámara detectada en índice {i}")
                    print(f"   Backend: {backend_name}")
                    print(f"   Resolución: {width}x{height}")
                    
                    # Detectar si es probablemente CamoStudio
                    if width >= 1280 or "Camo" in backend_name:
                        print(f"   📱 Posible iPhone/CamoStudio detectado")
                    
                    camaras_disponibles.append({
                        'index': i,
                        'backend': backend_name,
                        'resolution': f"{width}x{height}",
                        'is_likely_camo': width >= 1280
                    })
                cap.release()
        
        if not camaras_disponibles:
            print("❌ No se encontraron cámaras.")
            print("🔧 Para usar iPhone con Camo:")
            print("   1. Instalar Camo en iPhone")
            print("   2. Instalar Camo Studio en PC")
            print("   3. Conectar iPhone por USB o WiFi")
            print("   4. Iniciar Camo Studio")
        
        return camaras_disponibles
    
    def seleccionar_camara(self, camaras):
        """
        Seleccionar cámara con preferencia por CamoStudio
        """
        if len(camaras) == 1:
            print(f"Usando la única cámara detectada en el índice {camaras[0]['index']}")
            return camaras[0]['index']
        
        print("\n📷 Cámaras disponibles:")
        for cam in camaras:
            camo_indicator = " 📱 (Posible CamoStudio)" if cam['is_likely_camo'] else ""
            print(f"  [{cam['index']}] {cam['resolution']} - {cam['backend']}{camo_indicator}")
        
        # Sugerir CamoStudio si está disponible
        camo_cameras = [cam for cam in camaras if cam['is_likely_camo']]
        if camo_cameras:
            print(f"\n💡 Recomendado: Índice {camo_cameras[0]['index']} (CamoStudio)")
        
        while True:
            try:
                seleccion = int(input("➡️ Ingresa el índice de la cámara: "))
                indices_validos = [cam['index'] for cam in camaras]
                if seleccion in indices_validos:
                    return seleccion
                else:
                    print("Índice inválido. Intenta de nuevo.")
            except ValueError:
                print("Entrada inválida. Ingresa un número.")
    
    def configure_camera(self, cap, cam_index):
        """
        Configurar cámara para máximo rendimiento con iPhone/Camo
        """
        try:
            # Configurar resolución optimizada
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Configuraciones adicionales para calidad
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baja latencia
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)   # Autofocus activado
            
            # Verificar configuración aplicada
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📹 Cámara configurada:")
            print(f"   Resolución: {actual_width}x{actual_height}")
            print(f"   FPS objetivo: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando cámara: {e}")
            return False
    
    def capture_frames(self, cap):
        """
        Thread para captura continua de frames
        Optimizado para evitar lag con buffer limitado
        """
        while self.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("⚠️ No se pudo capturar frame")
                time.sleep(0.1)
                continue
            
            # Agregar timestamp
            timestamp = time.time()
            
            # Agregar al queue sin bloquear
            if not self.frame_queue.full():
                self.frame_queue.put((frame, timestamp))
            else:
                # Descartar frame más antiguo si el buffer está lleno
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, timestamp))
                except:
                    pass
    
    def process_frames(self):
        """
        Thread para procesamiento de detección
        Ejecuta predicciones en frames capturados
        """
        while self.is_running:
            try:
                # Obtener frame del queue
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Extraer área de detección (centro de la imagen)
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                half_size = self.detection_area_size // 2
                
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(w, center_x + half_size)
                y2 = min(h, center_y + half_size)
                
                detection_area = frame[y1:y2, x1:x2]
                
                # Realizar predicción
                result = predict_cardamomo(detection_area)
                result['timestamp'] = timestamp
                result['detection_coordinates'] = (x1, y1, x2, y2)
                
                # Enviar comando a Arduino
                if self.arduino:
                    self.arduino.enviar_dato(result)
                
                # Guardar en base de datos
                if self.db:
                    self.db.insert_result(result)
                
                # Actualizar estadísticas
                self.total_detections += 1
                if result.get('quality', False):
                    self.quality_accepted += 1
                
                # Agregar a queue de resultados
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # Mantener historial para análisis
                self.detection_history.append(result)
                if len(self.detection_history) > 100:
                    self.detection_history.pop(0)
                
            except Exception as e:
                logger.error(f"❌ Error procesando frame: {e}")
                time.sleep(0.1)
    
    def draw_overlay(self, frame):
        """
        Dibujar overlay con información de detección
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Dibujar área de detección
        if self.show_detection_area:
            center_x, center_y = w // 2, h // 2
            half_size = self.detection_area_size // 2
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
            
            # Rectángulo del área de detección
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(overlay, "AREA DE DETECCION", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Mostrar último resultado
        if self.last_result:
            result_color = (0, 255, 0) if self.last_result.get('quality', False) else (0, 0, 255)
            action_text = self.last_result.get('action', 'N/A')
            class_text = self.last_result.get('class', 'N/A')
            confidence = self.last_result.get('confidence', 0)
            
            # Panel de información
            cv2.rectangle(overlay, (20, 20), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(overlay, (20, 20), (400, 120), (255, 255, 255), 2)
            
            cv2.putText(overlay, f"Clase: {class_text}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, f"Confianza: {confidence:.3f}", (30, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, f"Accion: {action_text}", (30, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        
        # Mostrar estadísticas
        if self.show_statistics:
            stats_y = h - 100
            cv2.rectangle(overlay, (20, stats_y), (300, h-20), (0, 0, 0), -1)
            cv2.rectangle(overlay, (20, stats_y), (300, h-20), (255, 255, 255), 2)
            
            cv2.putText(overlay, f"FPS: {self.current_fps:.1f}", (30, stats_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Detecciones: {self.total_detections}", (30, stats_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Calidad: {self.quality_accepted}", (30, stats_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return overlay
    
    def start_camera(self):
        """
        Iniciar sistema completo de cámara con detección
        """
        # Detectar y seleccionar cámara
        camaras = self.detectar_camaras_disponibles()
        if not camaras:
            return False
        
        cam_index = self.seleccionar_camara(camaras)
        
        # Abrir cámara con DirectShow para mejor compatibilidad
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara en índice {cam_index}")
            return False
        
        # Configurar cámara
        if not self.configure_camera(cap, cam_index):
            cap.release()
            return False
        
        print("🚀 Iniciando sistema de detección...")
        print("📋 Controles:")
        print("   ESC - Salir")
        print("   SPACE - Pausar/Reanudar")
        print("   S - Mostrar/Ocultar estadísticas")
        print("   D - Mostrar/Ocultar área de detección")
        
        # Iniciar threads
        self.is_running = True
        
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(cap,))
        self.processing_thread = threading.Thread(target=self.process_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        # Loop principal de display
        paused = False
        
        try:
            while self.is_running:
                # Actualizar FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Obtener frame más reciente
                if not self.frame_queue.empty():
                    frame, _ = self.frame_queue.get()
                    
                    # Obtener último resultado si está disponible
                    if not self.result_queue.empty():
                        self.last_result = self.result_queue.get()
                    
                    # Dibujar overlay
                    display_frame = self.draw_overlay(frame)
                    
                    # Mostrar frame
                    if not paused:
                        cv2.imshow("Clasificador de Cardamomo - iPhone15 Pro", display_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print("⏸️ Pausado" if paused else "▶️ Reanudado")
                elif key == ord('s'):  # S
                    self.show_statistics = not self.show_statistics
                elif key == ord('d'):  # D
                    self.show_detection_area = not self.show_detection_area
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupción manual detectada")
        
        finally:
            # Limpiar recursos
            print("🧹 Cerrando sistema...")
            self.is_running = False
            
            # Esperar a que terminen los threads
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("✅ Sistema cerrado correctamente")
            return True

# Función de compatibilidad con código existente
def start_camera(arduino, db):
    """
    Función de compatibilidad con el código original
    """
    camera_stream = OptimizedCameraStream(arduino, db)
    return camera_stream.start_camera()

# Función de prueba independiente
def test_camera_stream():
    """
    Probar el stream de cámara sin Arduino ni DB
    """
    print("🧪 Probando stream de cámara...")
    
    # Crear stream sin Arduino ni DB para pruebas
    camera_stream = OptimizedCameraStream(None, None)
    camera_stream.start_camera()

if __name__ == "__main__":
    test_camera_stream()