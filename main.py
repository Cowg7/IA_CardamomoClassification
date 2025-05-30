#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Principal de Clasificación de Cardamomo
Integra GUI, cámara, Arduino y base de datos
Optimizado para RTX4050 y iPhone15 Pro
"""

import threading
import time
import logging
import sys
from pathlib import Path

# Importar módulos del proyecto
from gui import OptimizedGUI
from camera_stream import OptimizedCameraStream
from arduino_controller import ArduinoController
from db_handler import OptimizedDBHandler

# Configurar logging principal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cardamomo_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CardamomoClassificationSystem:
    """
    Sistema principal que coordina todos los componentes
    """
    def __init__(self, arduino_port='COM5'):
        """
        Inicializar sistema completo
        
        Args:
            arduino_port (str): Puerto COM del Arduino
        """
        self.arduino_port = arduino_port
        self.arduino = None
        self.db = None
        self.camera_stream = None
        self.gui = None
        
        # Threads del sistema
        self.camera_thread = None
        self.gui_thread = None
        
        # Estado del sistema
        self.system_running = False
        self.initialization_success = False
        
        logger.info("🚀 Inicializando Sistema de Clasificación de Cardamomo")
    
    def initialize_components(self):
        """
        Inicializar todos los componentes del sistema
        """
        try:
            # 1. Inicializar base de datos
            logger.info("📊 Inicializando base de datos...")
            self.db = OptimizedDBHandler()
            if not hasattr(self.db, 'connection_params'):
                logger.error("❌ Error inicializando base de datos")
                return False
            logger.info("✅ Base de datos inicializada")
            
            # 2. Inicializar Arduino
            logger.info("🔌 Inicializando controlador Arduino...")
            self.arduino = ArduinoController(port=self.arduino_port)
            
            # Verificar conexión Arduino (no crítico si falla)
            if not self.arduino.is_connected:
                logger.warning("⚠️ Arduino no conectado - continuando sin control físico")
            else:
                logger.info("✅ Arduino conectado correctamente")
            
            # 3. Inicializar stream de cámara
            logger.info("📹 Inicializando stream de cámara...")
            self.camera_stream = OptimizedCameraStream(self.arduino, self.db)
            logger.info("✅ Stream de cámara inicializado")
            
            # 4. Inicializar GUI
            logger.info("🖥️ Inicializando interfaz gráfica...")
            self.gui = OptimizedGUI(self.camera_stream, self.arduino, self.db)
            logger.info("✅ GUI inicializada")
            
            self.initialization_success = True
            logger.info("🎉 Todos los componentes inicializados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error durante inicialización: {e}")
            return False
    
    def start_system(self):
        """
        Iniciar el sistema completo
        """
        if not self.initialization_success:
            logger.error("❌ Sistema no inicializado correctamente")
            return False
        
        try:
            logger.info("🚀 Iniciando sistema completo...")
            self.system_running = True
            
            # Iniciar stream de cámara en thread separado
            self.camera_thread = threading.Thread(
                target=self._run_camera_stream,
                daemon=True,
                name="CameraThread"
            )
            self.camera_thread.start()
            
            # Pequeña pausa para que la cámara se inicialice
            time.sleep(2)
            
            # Iniciar GUI en thread principal (tkinter requiere thread principal)
            logger.info("🖥️ Iniciando interfaz gráfica...")
            self.gui.start_gui()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema: {e}")
            self.stop_system()
            return False
    
    def _run_camera_stream(self):
        """
        Ejecutar stream de cámara en thread separado
        """
        try:
            logger.info("📹 Iniciando detección por cámara...")
            self.camera_stream.start_camera()
        except Exception as e:
            logger.error(f"❌ Error en stream de cámara: {e}")
    
    def stop_system(self):
        """
        Detener sistema completo de forma segura
        """
        logger.info("🛑 Deteniendo sistema...")
        self.system_running = False
        
        # Detener stream de cámara
        if self.camera_stream:
            self.camera_stream.is_running = False
        
        # Esperar a que termine el thread de cámara
        if self.camera_thread and self.camera_thread.is_alive():
            logger.info("⏳ Esperando finalización de cámara...")
            self.camera_thread.join(timeout=5)
        
        # Cerrar Arduino
        if self.arduino:
            logger.info("🔌 Cerrando conexión Arduino...")
            self.arduino.close()
        
        # Cerrar base de datos
        if self.db and hasattr(self.db, 'connection_pool'):
            logger.info("📊 Cerrando conexiones de base de datos...")
            for conn in self.db.connection_pool:
                try:
                    conn.close()
                except:
                    pass
        
        logger.info("✅ Sistema detenido correctamente")
    
    def get_system_status(self):
        """
        Obtener estado actual del sistema
        """
        return {
            'system_running': self.system_running,
            'arduino_connected': self.arduino.is_connected if self.arduino else False,
            'camera_active': self.camera_stream.is_running if self.camera_stream else False,
            'database_active': bool(self.db),
            'gui_active': bool(self.gui)
        }

def print_startup_banner():
    """
    Mostrar banner de inicio del sistema
    """
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║        SISTEMA DE CLASIFICACIÓN DE            ║
    ║              CARDAMOMO v2.0                   ║
    ║                                               ║
    ║  🎯 Detección en tiempo real                  ║
    ║  📱 Compatible con iPhone15 Pro + CamoStudio  ║
    ║  🚀 Optimizado para RTX4050                   ║
    ║  🤖 Control automático con Arduino            ║
    ║  📊 Base de datos PostgreSQL                  ║
    ╚═══════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """
    Función principal del sistema
    """
    print_startup_banner()
    
    # Verificar dependencias críticas
    try:
        import cv2
        import tkinter as tk
        import psycopg2
        import serial
        logger.info("✅ Todas las dependencias verificadas")
    except ImportError as e:
        logger.error(f"❌ Dependencia faltante: {e}")
        print("\n🔧 Instala las dependencias faltantes:")
        print("pip install opencv-python tkinter psycopg2-binary pyserial pillow numpy")
        return False
    
    # Configurar puerto Arduino (cambiar según tu sistema)
    arduino_port = 'COM5'  # Windows
    # arduino_port = '/dev/ttyUSB0'  # Linux
    # arduino_port = '/dev/tty.usbmodem1411'  # macOS
    
    # Crear e inicializar sistema
    system = CardamomoClassificationSystem(arduino_port=arduino_port)
    
    if not system.initialize_components():
        logger.error("❌ Fallo en inicialización del sistema")
        return False
    
    # Mostrar estado del sistema
    status = system.get_system_status()
    logger.info("📋 Estado del sistema:")
    logger.info(f"   Arduino: {'✅ Conectado' if status['arduino_connected'] else '⚠️ Desconectado'}")
    logger.info(f"   Base de datos: {'✅ Activa' if status['database_active'] else '❌ Inactiva'}")
    logger.info(f"   Cámara: {'✅ Lista' if status['camera_active'] else '⚠️ Pendiente'}")
    logger.info(f"   GUI: {'✅ Lista' if status['gui_active'] else '❌ Error'}")
    
    try:
        # Iniciar sistema principal
        success = system.start_system()
        
        if success:
            logger.info("🎉 Sistema ejecutado exitosamente")
        else:
            logger.error("❌ Error durante ejecución del sistema")
            
    except KeyboardInterrupt:
        logger.info("⌨️ Interrupción manual detectada")
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
    finally:
        # Limpiar recursos
        system.stop_system()
        logger.info("👋 Sistema finalizado")

if __name__ == '__main__':
    main()