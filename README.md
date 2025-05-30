# 🎯 Clasificador de Cardamomo por Color

Sistema de visión artificial para clasificar cardamomo según su color utilizando IA, cámara iPhone (CamoStudio), y control Arduino. Optimizado para detección en tiempo real.

## 🧠 Funcionalidad
- Clasifica cardamomo en Verde (aceptado), Amarillo o Café (rechazado)
- Solo acepta productos con confianza ≥ 0.85
- Visualización en GUI con Tkinter
- Control automático por Arduino
- Registro en base de datos PostgreSQL

## 🚀 Requerimientos

- Python 3.10+
- TensorFlow
- OpenCV
- Pillow
- psycopg2

## ⚙️ Instalación rápida

```bash
git clone https://github.com/TU_USUARIO/IA_CardamomoClassification.git
cd IA_CardamomoClassification
pip install -r requirements.txt
