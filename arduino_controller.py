import serial
import time
import threading
from collections import deque
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoController:
    def __init__(self, port='COM5', baudrate=9600, timeout=2):
        """
        Controlador optimizado para Arduino
        Maneja comunicación con la banda transportadora basada en calidad del cardamomo
        
        Comandos:
        - 'A': ACEPTAR (cardamomo verde de calidad)
        - 'R': RECHAZAR (cardamomo amarillo/café o baja confianza)
        - 'S': STOP (emergencia)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False
        
        # Buffer de comandos para evitar spam
        self.command_buffer = deque(maxlen=10)
        self.last_command_time = 0
        self.min_command_interval = 0.5  # Mínimo 500ms entre comandos
        
        # Estadísticas
        self.commands_sent = {'ACEPTAR': 0, 'RECHAZAR': 0, 'STOP': 0}
        self.connection_errors = 0
        
        # Lock para thread safety
        self.serial_lock = threading.Lock()
        
        # Intentar conexión inicial
        self.connect()
    
    def connect(self):
        """
        Establecer conexión con Arduino
        """
        try:
            logger.info(f"🔌 Intentando conectar a Arduino en {self.port}...")
            
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=2
            )
            
            # Esperar a que Arduino se inicialice
            time.sleep(2)
            
            # Enviar comando de prueba
            self.serial_connection.write(b'T')  # Test command
            self.serial_connection.flush()
            
            self.is_connected = True
            logger.info("✅ Arduino conectado exitosamente")
            
            # Limpiar buffer inicial
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
        except serial.SerialException as e:
            logger.error(f"❌ Error conectando Arduino: {e}")
            self.is_connected = False
            self.connection_errors += 1
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            self.is_connected = False
            self.connection_errors += 1
    
    def reconnect(self):
        """
        Intentar reconectar con Arduino
        """
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except:
                pass
        
        time.sleep(1)
        self.connect()
    
    def send_command(self, command, force=False):
        """
        Enviar comando al Arduino con control de frecuencia
        
        Args:
            command (str): 'A' (Aceptar), 'R' (Rechazar), 'S' (Stop)
            force (bool): Forzar envío ignorando intervalo mínimo
        """
        if not self.is_connected:
            logger.warning("⚠️ Arduino no conectado, intentando reconectar...")
            self.reconnect()
            if not self.is_connected:
                return False
        
        # Control de frecuencia de comandos
        current_time = time.time()
        if not force and (current_time - self.last_command_time) < self.min_command_interval:
            return False
        
        try:
            with self.serial_lock:
                # Validar comando
                if command not in ['A', 'R', 'S']:
                    logger.error(f"❌ Comando inválido: {command}")
                    return False
                
                # Enviar comando
                self.serial_connection.write(command.encode())
                self.serial_connection.flush()
                
                # Actualizar estadísticas
                command_name = {'A': 'ACEPTAR', 'R': 'RECHAZAR', 'S': 'STOP'}[command]
                self.commands_sent[command_name] += 1
                
                # Agregar al buffer
                self.command_buffer.append({
                    'command': command,
                    'timestamp': current_time,
                    'name': command_name
                })
                
                self.last_command_time = current_time
                
                logger.info(f"📤 Comando enviado: {command_name}")
                return True
                
        except serial.SerialException as e:
            logger.error(f"❌ Error enviando comando: {e}")
            self.is_connected = False
            self.connection_errors += 1
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado enviando comando: {e}")
            return False
    
    def enviar_dato(self, prediction_result):
        """
        Enviar comando basado en resultado de predicción
        Método de compatibilidad con código existente
        
        Args:
            prediction_result (dict): Resultado del clasificador
        """
        if not isinstance(prediction_result, dict):
            logger.error("❌ Formato de resultado inválido")
            return False
        
        # Determinar comando basado en calidad
        if prediction_result.get('quality', False):
            command = 'A'  # Aceptar cardamomo de calidad
            action = "ACEPTAR"
        else:
            command = 'R'  # Rechazar cardamomo
            action = "RECHAZAR"
        
        success = self.send_command(command)
        
        if success:
            logger.info(f"🎯 Acción: {action} - Clase: {prediction_result.get('class', 'N/A')} "
                       f"(Confianza: {prediction_result.get('confidence', 0):.3f})")
        
        return success
    
    def emergency_stop(self):
        """
        Comando de parada de emergencia
        """
        logger.warning("🛑 PARADA DE EMERGENCIA")
        return self.send_command('S', force=True)
    
    def read_arduino_response(self):
        """
        Leer respuesta del Arduino (si está configurado para enviar)
        """
        if not self.is_connected or not self.serial_connection.in_waiting:
            return None
        
        try:
            with self.serial_lock:
                response = self.serial_connection.readline().decode().strip()
                if response:
                    logger.info(f"📥 Respuesta Arduino: {response}")
                    return response
        except Exception as e:
            logger.error(f"❌ Error leyendo respuesta: {e}")
        
        return None
    
    def get_statistics(self):
        """
        Obtener estadísticas del controlador
        """
        return {
            'connected': self.is_connected,
            'port': self.port,
            'commands_sent': self.commands_sent.copy(),
            'connection_errors': self.connection_errors,
            'recent_commands': list(self.command_buffer),
            'total_commands': sum(self.commands_sent.values())
        }
    
    def close(self):
        """
        Cerrar conexión con Arduino
        """
        if self.serial_connection and self.is_connected:
            try:
                # Enviar comando de parada antes de cerrar
                self.send_command('S', force=True)
                time.sleep(0.5)
                
                self.serial_connection.close()
                logger.info("🔌 Conexión Arduino cerrada")
            except Exception as e:
                logger.error(f"❌ Error cerrando conexión: {e}")
            finally:
                self.is_connected = False
    
    def __del__(self):
        """
        Destructor - cerrar conexión automáticamente
        """
        self.close()

# Función de prueba del controlador
def test_arduino_controller():
    """
    Función de prueba para verificar comunicación con Arduino
    """
    print("🧪 Probando controlador Arduino...")
    
    # Crear controlador (cambia COM5 por tu puerto)
    controller = ArduinoController('COM5')
    
    if not controller.is_connected:
        print("❌ No se pudo conectar al Arduino")
        print("🔧 Verifica:")
        print("   - Puerto COM correcto")
        print("   - Arduino conectado y encendido")
        print("   - Driver USB instalado")
        return
    
    print("✅ Arduino conectado exitosamente")
    
    # Simular resultados de clasificación
    test_results = [
        {'class': 'Verde', 'confidence': 0.92, 'quality': True},
        {'class': 'Amarillo', 'confidence': 0.78, 'quality': False},
        {'class': 'Verde', 'confidence': 0.83, 'quality': False},  # Baja confianza
        {'class': 'Cafe', 'confidence': 0.95, 'quality': False}
    ]
    
    print("\n🎯 Enviando comandos de prueba...")
    for i, result in enumerate(test_results, 1):
        print(f"\nPrueba {i}:")
        print(f"   Clase: {result['class']}")
        print(f"   Confianza: {result['confidence']}")
        print(f"   Calidad: {result['quality']}")
        
        success = controller.enviar_dato(result)
        print(f"   Resultado: {'✅ Enviado' if success else '❌ Error'}")
        
        time.sleep(1)  # Pausa entre comandos
    
    # Mostrar estadísticas
    stats = controller.get_statistics()
    print(f"\n📊 Estadísticas:")
    print(f"   Comandos ACEPTAR: {stats['commands_sent']['ACEPTAR']}")
    print(f"   Comandos RECHAZAR: {stats['commands_sent']['RECHAZAR']}")
    print(f"   Total comandos: {stats['total_commands']}")
    print(f"   Errores de conexión: {stats['connection_errors']}")
    
    # Cerrar conexión
    controller.close()
    print("\n✅ Prueba completada")

if __name__ == "__main__":
    test_arduino_controller()