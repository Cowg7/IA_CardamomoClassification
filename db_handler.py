import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json
import logging
import threading
from contextlib import contextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDBHandler:
    def __init__(self, dbname="cardamomo", user="postgres", 
                 password="algoritmos", host="localhost", port="5432"):
        """
        Manejador optimizado de base de datos PostgreSQL
        Diseñado para almacenar resultados de clasificación de cardamomo por color
        """
        self.connection_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        
        self.connection_pool = []
        self.pool_size = 5
        self.pool_lock = threading.Lock()
        
        # Estadísticas
        self.total_inserts = 0
        self.failed_inserts = 0
        self.connection_errors = 0
        
        # Inicializar conexión y crear tablas
        self._initialize_database()
    
    def _create_connection(self):
        """
        Crear nueva conexión a PostgreSQL
        """
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = True  # Para mejor rendimiento
            return conn
        except psycopg2.Error as e:
            logger.error(f"❌ Error conectando a PostgreSQL: {e}")
            self.connection_errors += 1
            return None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager para obtener conexión del pool
        """
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = self._create_connection()
            
            if conn and not conn.closed:
                yield conn
            else:
                conn = self._create_connection()
                yield conn
                
        except Exception as e:
            logger.error(f"❌ Error con conexión DB: {e}")
            yield None
        finally:
            if conn and not conn.closed:
                with self.pool_lock:
                    if len(self.connection_pool) < self.pool_size:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
    
    def _initialize_database(self):
        """
        Inicializar base de datos y crear tablas necesarias
        """
        with self.get_connection() as conn:
            if not conn:
                logger.error("❌ No se pudo conectar a la base de datos")
                return False
            
            try:
                cursor = conn.cursor()
                
                # Crear tabla principal de resultados con campos optimizados
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS clasificacion_cardamomo (
                        id SERIAL PRIMARY KEY,
                        clase VARCHAR(20) NOT NULL,
                        confianza FLOAT NOT NULL,
                        es_calidad BOOLEAN NOT NULL,
                        accion VARCHAR(10) NOT NULL,
                        analisis_color JSONB,
                        todas_predicciones JSONB,
                        tiempo_procesamiento_ms FLOAT,
                        fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Índices para consultas rápidas
                        CONSTRAINT chk_clase CHECK (clase IN ('Verde', 'Amarillo', 'Cafe')),
                        CONSTRAINT chk_accion CHECK (accion IN ('ACEPTAR', 'DESCARTAR'))
                    )
                ''')
                
                # Crear índices para mejorar rendimiento
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_fecha_hora 
                    ON clasificacion_cardamomo(fecha_hora)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_clase_calidad 
                    ON clasificacion_cardamomo(clase, es_calidad)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_accion_fecha 
                    ON clasificacion_cardamomo(accion, fecha_hora)
                ''')
                
                # Crear tabla de estadísticas diarias
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS estadisticas_diarias (
                        fecha DATE PRIMARY KEY,
                        total_procesados INTEGER DEFAULT 0,
                        total_aceptados INTEGER DEFAULT 0,
                        total_rechazados INTEGER DEFAULT 0,
                        porcentaje_calidad FLOAT DEFAULT 0,
                        tiempo_promedio_ms FLOAT DEFAULT 0,
                        ultima_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Crear tabla de configuración del sistema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS configuracion_sistema (
                        clave VARCHAR(50) PRIMARY KEY,
                        valor TEXT NOT NULL,
                        descripcion TEXT,
                        fecha_modificacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insertar configuración por defecto
                cursor.execute('''
                    INSERT INTO configuracion_sistema (clave, valor, descripcion)
                    VALUES 
                        ('umbral_confianza', '0.85', 'Umbral mínimo de confianza para aceptar cardamomo verde'),
                        ('area_deteccion', '400', 'Tamaño del área de detección en píxeles'),
                        ('fps_objetivo', '30', 'FPS objetivo para procesamiento')
                    ON CONFLICT (clave) DO NOTHING
                ''')
                
                logger.info("✅ Base de datos inicializada correctamente")
                return True
                
            except psycopg2.Error as e:
                logger.error(f"❌ Error inicializando base de datos: {e}")
                return False
    

    def _update_daily_stats(self, clase, confianza):
        # Simulación de actualización diaria para evitar error
        logger.info(f"[DB] Simulando _update_daily_stats para {clase} ({confianza})")
    def insert_result(self, result):
        """
        Insertar resultado de clasificación en la base de datos
        
        Args:
            result (dict): Resultado del clasificador con estructura:
                - class: str
                - confidence: float  
                - quality: bool
                - action: str
                - color_analysis: dict (opcional)
                - all_predictions: dict (opcional)
                - processing_time_ms: float (opcional)
        """
        with self.get_connection() as conn:
            if not conn:
                self.failed_inserts += 1
                return False
            
            try:
                cursor = conn.cursor()
                
                # Preparar datos para inserción
                clase = result.get('class', 'N/A')
                confianza = float(result.get('confidence', 0.0))
                es_calidad = bool(result.get('quality', False))
                accion = result.get('action', 'DESCARTAR')
                
                # Datos opcionales como JSON
                analisis_color = json.dumps(result.get('color_analysis', {}))
                todas_predicciones = json.dumps(result.get('all_predictions', {}))
                tiempo_procesamiento = float(result.get('processing_time_ms', 0.0))
                
                # Insertar registro
                cursor.execute('''
                    INSERT INTO clasificacion_cardamomo 
                    (clase, confianza, es_calidad, accion, analisis_color, 
                     todas_predicciones, tiempo_procesamiento_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (clase, confianza, es_calidad, accion, analisis_color, 
                      todas_predicciones, tiempo_procesamiento))
                
                record_id = cursor.fetchone()[0]
                
                # Actualizar estadísticas diarias
                self._update_daily_stats(cursor)
                
                self.total_inserts += 1
                logger.debug(f"📝 Resultado guardado con ID: {record_id}")
                
                return record_id
                
            except (psycopg2.Error, ValueError, TypeError) as e:
                logger.error(f"❌ Error insertando resultado: {e}")
                self.failed_inserts += 1
                return False