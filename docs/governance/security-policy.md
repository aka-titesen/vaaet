<!-- context: VAAET/docs/governance/security-policy.md — Política de seguridad y privacidad detallada.
Complementa SECURITY.md (raíz) y DATA_LINEAGE.md. -->

# Política de Seguridad y Privacidad — VAAET

## Identificación del Proyecto

| Campo | Detalles |
|---|---|
| **Nombre del Proyecto** | VAAET — Video Advanced Analysis of Traffic |
| **Versión** | 4.0.0 |
| **Estado** | Aprobado |
| **Responsable Técnico** | Facundo Nicolás González |
| **Última Revisión** | 2026-07-23 |

---

## 1. Principios de Seguridad

VAAET adopta el principio de **Seguridad por Diseño (Security by Design)**:

1. **Mínimo privilegio**: Solo las operaciones estrictamente necesarias tienen acceso a datos sensibles
2. **Defensa en profundidad**: Múltiples capas de protección (código, infra, políticas)
3. **Degradación segura**: Ante fallo de seguridad, el sistema degrada a un estado seguro (sin BD)

---

## 2. Gestión de Credenciales

### 2.1 Reglas Obligatorias

| Regla | Implementación |
|---|---|
| Credenciales por variables de entorno | `src/vaaet/data/database.py` lee `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` |
| Nunca hardcodear | Verificado por `test_repo_hygiene.py` |
| Nunca imprimir en outputs | `src/vaaet/data/database.py` no logea credenciales |
| `.env` en `.gitignore` | Configurado |
| Template documentado | `.env.example` disponible |

### 2.2 Flujo de Credenciales en Colab

```
Usuario → getpass() → os.environ → vaaet.data.database → SQLAlchemy Engine → AWS RDS
                         ↑
                    Nunca persiste en disco
                    Nunca aparece en outputs
```

---

## 3. Categorización de Datos y Base Legal

| **Categoría de Dato** | **Ejemplos en VAAET** | **Base Legal** |
|---|---|---|
| Telemetría vehicular | Conteos por tipo, velocidad promedio | Interés legítimo (monitoreo vial) |
| Datos temporales | Timestamps por minuto | Interés legítimo (análisis de tráfico) |
| Metadatos de video | Clip ID, duración | Ejecución del propósito del sistema |
| Datos de modelo | Pesos `.keras`, scalers `.joblib` | Interés legítimo (inferencia) |

> **Nota**: VAAET no recolecta datos personales. No aplican categorías de datos biométricos, identificatorios ni transaccionales.

---

## 4. Medidas de Seguridad Técnica y Organizativa

### 4.1 Datos NO Recolectados

VAAET **no** extrae, almacena ni procesa:
- Patentes de vehículos individuales
- Imágenes de personas o conductores
- Datos de identidad personal
- Frames individuales del video (solo agregados por minuto)
- Tracking individual fuera del video procesado

### 3.2 Datos Recolectados

| Dato | Granularidad | Sensibilidad | Retención |
|---|---|---|---|
| Conteos de vehículos por tipo | Por minuto | Baja | Indefinida en BD |
| Velocidad promedio | Por minuto | Baja | Indefinida en BD |
| Estado del tráfico | Por minuto | Baja | Indefinida en BD |
| Timestamp | Por minuto | Media (patrón temporal) | Indefinida en BD |
| Clip ID | Por video | Baja | Indefinida en BD |

### 3.3 Video de Salida

- El video anotado es **efímero** (se pierde al cerrar la sesión de Colab)
- Contiene bounding boxes y velocidades, pero no datos personales
- La descarga es responsabilidad del usuario

---

## 5. Derechos del Titular de los Datos

Dado que VAAET **no recolecta datos personales identificables**, los derechos ARCO (Acceso, Rectificación, Cancelación, Oposición) no aplican directamente. Sin embargo, se garantiza:

- **Transparencia**: Documentación pública del pipeline, datos recolectados, y limitaciones
- **Minimización**: Solo se persisten datos agregados (por minuto), nunca frames individuales
- **Portabilidad**: Los datos de telemetría se pueden exportar como CSV desde PostgreSQL
- **Eliminación**: Los datos en Colab son efímeros; los datos en BD pueden eliminarse bajo solicitud

---

## 6. Seguridad de Infraestructura

### 4.1 Google Colab

| Aspecto | Estado | Recomendación |
|---|---|---|
| Aislamiento de runtime | ✅ (Google gestiona) | — |
| Almacenamiento efímero | ✅ (seguro por diseño) | Descargar artefactos antes de cerrar |
| Secretos en Colab | ⚠️ Variables de entorno en memoria | Usar Colab Secrets (feature nativa) |

### 4.2 AWS RDS

| Aspecto | Estado | Recomendación |
|---|---|---|
| Encriptación en reposo | Depende de config de RDS | Habilitar encriptación AES-256 |
| SSL en tránsito | ❌ No implementado | **Implementar SSL** (prioridad alta) |
| Backups automáticos | Depende de config de RDS | Habilitar (retención 7 días) |
| Grupos de seguridad | Depende de config de VPC | Restringir acceso por IP |

---

## 5. Seguridad del Código

### 5.1 Consultas SQL

- Todas las consultas usan **parámetros con nombre** via SQLAlchemy (`text()` + bind parameters)
- **No hay concatenación de strings** en consultas SQL
- Las consultas están definidas como constantes en `src/vaaet/data/persistence.py`

### 5.2 Dependencias

- Dependencias especificadas con versiones mínimas en `pyproject.toml`
- Auditoría de vulnerabilidades: `pip audit` (ejecución manual recomendada)
- **Recomendación**: Integrar Dependabot en GitHub

### 5.3 Archivos Sensibles en `.gitignore`

```
.env                    # Variables de entorno
*.pt                    # Modelos YOLO
*.keras                 # Modelo MLP
*.joblib                # Scaler y label mapping
data/raw/               # Backups de BD
```

---

## 8. Procedimiento ante Brechas de Seguridad

En caso de una brecha de seguridad, se compromete a:

1. **Notificación**: Comunicar al responsable técnico dentro de las 24 horas de detección
2. **Contención**: Rotar credenciales comprometidas inmediatamente
3. **Remediación**: Aplicar correcciones técnicas y documentar el incidente

### Plan de Respuesta Detallado

| Paso | Acción | Responsable |
|---|---|---|
| 1 | Detectar exposición de credenciales | Desarrollador / CI |
| 2 | Rotar credenciales de BD inmediatamente | Administrador AWS |
| 3 | Auditar logs de acceso a RDS | Administrador AWS |
| 4 | Verificar historial de git por commits con secretos | Desarrollador |
| 5 | Ejecutar `git filter-branch` si es necesario | Desarrollador |
| 6 | Documentar el incidente y actualizar política | Desarrollador |

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23
