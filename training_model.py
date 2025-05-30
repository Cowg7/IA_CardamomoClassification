import tensorflow as tf
from keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os

# Configurar GPU para optimizar rendimiento con RTX4050
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento de memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")

class CardamomoColorClassifier:
    def __init__(self):
        """
        Clasificador de cardamomo basado en color
        Clases: Verde (calidad), Amarillo (descarte), Cafe (descarte)
        """
        self.classes = ['Verde', 'Amarillo', 'Cafe']
        self.img_size = (224, 224)
        self.model = None
        
        # Crear directorio del modelo
        os.makedirs('model', exist_ok=True)
    
    def create_model(self):
        """
        Crear modelo CNN optimizado para clasificación de colores
        Arquitectura diseñada para detectar características de color
        """
        model = Sequential([
            Input(shape=(224, 224, 3)),
            
            # Primer bloque convolucional - detectar bordes y formas básicas
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Segundo bloque - características de color más complejas
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Tercer bloque - patrones de color específicos
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Aplanar y clasificar
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')  # 3 clases de color
        ])
        
        # Compilar con optimizador Adam y learning rate adaptativo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        self.model = model
        return model
    
    def generate_color_based_data(self, samples_per_class=200):
        """
        Generar datos de entrenamiento basados en características de color
        para entrenar el modelo inicial antes de usar muestras reales
        """
        X = []
        y = []
        
        for class_idx, color_name in enumerate(self.classes):
            print(f"🎨 Generando muestras para clase: {color_name}")
            
            for _ in range(samples_per_class):
                # Crear imagen base con ruido
                img = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                
                if color_name == 'Verde':
                    # Generar tonos verdes variados (cardamomo fresco)
                    green_intensity = np.random.randint(80, 200)
                    img[:, :, 1] = np.clip(img[:, :, 1] + green_intensity, 0, 255)
                    img[:, :, 0] = np.clip(img[:, :, 0] + np.random.randint(20, 80), 0, 255)
                    img[:, :, 2] = np.clip(img[:, :, 2] + np.random.randint(10, 60), 0, 255)
                    
                elif color_name == 'Amarillo':
                    # Generar tonos amarillos (cardamomo maduro)
                    yellow_r = np.random.randint(180, 255)
                    yellow_g = np.random.randint(150, 220)
                    img[:, :, 0] = np.clip(img[:, :, 0] + yellow_r, 0, 255)
                    img[:, :, 1] = np.clip(img[:, :, 1] + yellow_g, 0, 255)
                    img[:, :, 2] = np.clip(img[:, :, 2] + np.random.randint(0, 50), 0, 255)
                    
                elif color_name == 'Cafe':
                    # Generar tonos café/marrón (cardamomo deteriorado)
                    brown_intensity = np.random.randint(60, 120)
                    img[:, :, 0] = np.clip(img[:, :, 0] + brown_intensity, 0, 255)
                    img[:, :, 1] = np.clip(img[:, :, 1] + brown_intensity * 0.7, 0, 255)
                    img[:, :, 2] = np.clip(img[:, :, 2] + brown_intensity * 0.4, 0, 255)
                
                # Añadir ruido gaussiano para mayor realismo
                noise = np.random.normal(0, 10, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
                # Normalizar imagen
                img_normalized = img.astype(np.float32) / 255.0
                
                # Crear etiqueta one-hot
                label = np.zeros(3)
                label[class_idx] = 1
                
                X.append(img_normalized)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train_initial_model(self, epochs=50):
        """
        Entrenar el modelo inicial con datos sintéticos
        """
        print("🚀 Iniciando entrenamiento del modelo de clasificación por color...")
        
        # Crear modelo
        model = self.create_model()
        print("📊 Arquitectura del modelo:")
        model.summary()
        
        # Generar datos de entrenamiento
        X_train, y_train = self.generate_color_based_data(samples_per_class=300)
        X_val, y_val = self.generate_color_based_data(samples_per_class=50)
        
        print(f"📈 Datos generados - Entrenamiento: {X_train.shape}, Validación: {X_val.shape}")
        
        # Callbacks para optimizar entrenamiento
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        history = model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        
        # Guardar modelo
        model.save('model/cardamomo_color_classifier.h5')
        print("✅ Modelo guardado en: model/cardamomo_color_classifier.h5")
        
        return history
    
    def extract_color_features(self, image):
        """
        Extraer características de color para análisis adicional
        """
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calcular histogramas de color
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Estadísticas de color
        mean_hsv = np.mean(hsv, axis=(0, 1))
        std_hsv = np.std(hsv, axis=(0, 1))
        
        return {
            'mean_hue': mean_hsv[0],
            'mean_saturation': mean_hsv[1],
            'mean_value': mean_hsv[2],
            'std_hue': std_hsv[0],
            'std_saturation': std_hsv[1],
            'std_value': std_hsv[2]
        }

# Función principal para entrenar
def main():
    classifier = CardamomoColorClassifier()
    
    print("🎯 Clasificador de Cardamomo por Color")
    print("📋 Clases: Verde (Calidad), Amarillo (Descarte), Café (Descarte)")
    print("🔧 Optimizado para RTX4050 con CUDA")
    
    # Entrenar modelo inicial
    history = classifier.train_initial_model(epochs=50)
    
    print("\n✅ Entrenamiento completado!")
    print("📝 Próximos pasos:")
    print("   1. Toma fotos de tus 3 muestras de cardamomo")
    print("   2. Usa fine_tune_with_real_samples() para mejorar el modelo")
    print("   3. Prueba el clasificador con clasificador.py")

if __name__ == "__main__":
    main()