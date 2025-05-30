import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Configurar GPU para RTX4050
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")

class CardamomoColorClassifier:
    def __init__(self, model_path='model/cardamomo_color_classifier.h5'):
        """
        Clasificador optimizado para detección de cardamomo por color
        Umbral de confianza: Verde > 0.85 = Calidad, resto = Descarte
        """
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ['Verde', 'Amarillo', 'Cafe']
        self.confidence_threshold = 0.85
        
        # Buffer para estabilizar predicciones (evitar falsos positivos)
        self.prediction_buffer = deque(maxlen=5)
        
        # Métricas de rendimiento
        self.processing_times = deque(maxlen=100)
        
        print("✅ Clasificador de cardamomo por color cargado")
        print(f"🎯 Umbral de confianza para Verde: {self.confidence_threshold}")
    
    def preprocess_image(self, frame):
        """
        Preprocesar imagen para mejorar detección de color
        Optimizado para condiciones de iluminación variables
        """
        # Redimensionar manteniendo aspecto
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Mejorar contraste y brillo automáticamente
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Recombinar canales
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Normalizar para el modelo
        normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized
    
    def analyze_color_characteristics(self, image):
        """
        Análisis adicional de características de color
        para validar la predicción del modelo
        """
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color para cardamomo
        # Verde: Hue 35-85, Saturation 50-255, Value 50-255
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        # Amarillo: Hue 15-35, Saturation 50-255, Value 100-255
        yellow_lower = np.array([15, 50, 100])
        yellow_upper = np.array([35, 255, 255])
        
        # Café/Marrón: Hue 5-15, Saturation 50-255, Value 20-150
        brown_lower = np.array([5, 50, 20])
        brown_upper = np.array([15, 255, 150])
        
        # Crear máscaras para cada color
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # Calcular porcentajes de cada color
        total_pixels = image.shape[0] * image.shape[1]
        green_percentage = np.sum(green_mask > 0) / total_pixels
        yellow_percentage = np.sum(yellow_mask > 0) / total_pixels
        brown_percentage = np.sum(brown_mask > 0) / total_pixels
        
        return {
            'green_percentage': green_percentage,
            'yellow_percentage': yellow_percentage,
            'brown_percentage': brown_percentage,
            'dominant_color': max([
                ('Verde', green_percentage),
                ('Amarillo', yellow_percentage),
                ('Cafe', brown_percentage)
            ], key=lambda x: x[1])
        }
    
    def predict_cardamomo(self, frame):
        """
        Predecir calidad del cardamomo basado en color
        Retorna: calidad (True/False), clase, confianza, detalles
        """
        start_time = time.time()
        
        # Preprocesar imagen
        processed_frame = self.preprocess_image(frame)
        input_tensor = np.expand_dims(processed_frame, axis=0)
        
        # Predicción del modelo
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        
        # Obtener clase y confianza
        class_idx = np.argmax(predictions)
        predicted_class = self.classes[class_idx]
        confidence = float(predictions[class_idx])
        
        # Análisis adicional de color
        color_analysis = self.analyze_color_characteristics(frame)
        
        # Determinar calidad basada en color verde y confianza
        is_quality = (predicted_class == 'Verde' and 
                     confidence > self.confidence_threshold and
                     color_analysis['green_percentage'] > 0.3)
        
        # Añadir a buffer para estabilizar
        self.prediction_buffer.append({
            'class': predicted_class,
            'confidence': confidence,
            'quality': is_quality
        })
        
        # Decisión final basada en últimas predicciones
        recent_quality_votes = sum(1 for pred in self.prediction_buffer if pred['quality'])
        stable_quality = recent_quality_votes >= 3  # Mayoría de predicciones positivas
        
        # Registrar tiempo de procesamiento
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'quality': stable_quality,
            'action': 'ACEPTAR' if stable_quality else 'DESCARTAR',
            'color_analysis': color_analysis,
            'processing_time_ms': processing_time * 1000,
            'all_predictions': {
                'Verde': float(predictions[0]),
                'Amarillo': float(predictions[1]),
                'Cafe': float(predictions[2])
            }
        }
        
        return result
    
    def get_performance_stats(self):
        """
        Obtener estadísticas de rendimiento del clasificador
        """
        if not self.processing_times:
            return None
            
        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_processing_time_ms': avg_time * 1000,
            'estimated_fps': fps,
            'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0
        }

# Función de conveniencia para compatibilidad con código existente
def predict_cardamomo(frame):
    """
    Función de compatibilidad con el código existente
    """
    global classifier
    
    # Inicializar clasificador si no existe
    if 'classifier' not in globals():
        try:
            classifier = CardamomoColorClassifier()
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return {
                'class': 'Error',
                'confidence': 0.0,
                'quality': False,
                'action': 'DESCARTAR',
                'error': str(e)
            }
    
    return classifier.predict_cardamomo(frame)

# Función para probar el clasificador
def test_classifier():
    """
    Función de prueba para verificar el funcionamiento
    """
    print("🧪 Probando clasificador de cardamomo...")
    
    # Crear imagen de prueba verde
    test_green = np.zeros((224, 224, 3), dtype=np.uint8)
    test_green[:, :, 1] = 150  # Canal verde
    test_green[:, :, 0] = 50   # Un poco de rojo
    
    # Crear imagen de prueba amarilla
    test_yellow = np.zeros((224, 224, 3), dtype=np.uint8)
    test_yellow[:, :, 0] = 200  # Rojo
    test_yellow[:, :, 1] = 200  # Verde
    test_yellow[:, :, 2] = 50   # Poco azul
    
    classifier = CardamomoColorClassifier()
    
    print("\n🟢 Prueba con imagen verde:")
    result_green = classifier.predict_cardamomo(test_green)
    print(f"   Clase: {result_green['class']}")
    print(f"   Confianza: {result_green['confidence']:.3f}")
    print(f"   Acción: {result_green['action']}")
    
    print("\n🟡 Prueba con imagen amarilla:")
    result_yellow = classifier.predict_cardamomo(test_yellow)
    print(f"   Clase: {result_yellow['class']}")
    print(f"   Confianza: {result_yellow['confidence']:.3f}")
    print(f"   Acción: {result_yellow['action']}")
    
    # Mostrar estadísticas de rendimiento
    stats = classifier.get_performance_stats()
    if stats:
        print(f"\n📊 Rendimiento:")
        print(f"   Tiempo promedio: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"   FPS estimado: {stats['estimated_fps']:.1f}")
        print(f"   GPU disponible: {'Sí' if stats['gpu_available'] else 'No'}")

if __name__ == "__main__":
    test_classifier()