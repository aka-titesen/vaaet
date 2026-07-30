# Guía de Contribución — VAAET

Gracias por tu interés en contribuir a VAAET. Este documento establece las convenciones y reglas para modificar el proyecto.

---

## Reglas Fundamentales de Arquitectura

1. **El código compartido vive en `src/`** — Módulos reutilizables (config, db, features, labeling, perception) compartidos por todos los notebooks
2. **Los notebooks son orquestadores** — Llaman funciones de `src/` y proveen la interfaz de Colab
3. **El Módulo 0 (`archive/00_bootstrap/`) está CONGELADO** — Nunca modificar
4. **Debe funcionar en Google Colab** — Todos los cambios deben ser compatibles con Colab Free
5. **Leer `AGENTS.md` antes de empezar** — Contiene las reglas de gobernanza Always/Ask/Never

---

## Convenciones de Código

### Estilo General

- Python 3.8+ compatible
- Formato PEP 8 (excepto longitud de línea, que puede extenderse en notebooks)
- Type hints en todas las funciones públicas de `src/`
- Docstrings en español para todas las funciones

### Prints con Emoji

El sistema usa `print()` con prefijos de emoji en vez de `logging`:

```python
print("✅ Operación exitosa")
print("⚠️ Advertencia: parámetro fuera de rango")
print("🔴 Error: no se pudo conectar a la BD")
print("📊 Resultado: 42 vehículos detectados")
```

### Configuración

- **Todas las constantes y umbrales**: `src/config.py` (fuente única de verdad)
- **Credenciales de BD**: `src/db.py` vía variables de entorno exclusivamente
- **Credenciales**: NUNCA hardcodear. Usar variables de entorno o `getpass`

---

## Flujo de Contribución

1. Leer el ADR relevante en `docs/adr/` si tu cambio afecta una decisión arquitectónica
2. Si no existe un ADR y tu cambio es significativo, redactar uno antes de implementar
3. Verificar que todos los notebooks activos compilan sin errores después de tus cambios
4. Módulo 1: Verificar F1-macro ≥ 0.85 tras re-entrenamiento
5. Módulo 2: Verificar que el pipeline de percepción + clasificación produce salida válida
6. Actualizar la documentación correspondiente si tu cambio altera comportamiento observable
7. Ejecutar `pytest tests/ -v --tb=short` antes de solicitar revisión

---

## Estructura de Documentación

| Archivo | Propósito | Cuándo Actualizar |
|---|---|---|
| `README.md` | Visión general y uso | Cambios en features o dependencias |
| `AGENTS.md` | Contexto para agentes de IA | Cambios en arquitectura o reglas |
| `docs/PRD.md` | Requisitos del producto | Nuevos requisitos o cambios funcionales |
| `docs/SAD.md` | Arquitectura de software | Cambios en algoritmos o componentes |
| `docs/SRS.md` | Especificación de requisitos | Nuevos requisitos |
| `docs/USER_GUIDE.md` | Guía de usuario | Cambios en UX o flujo de ejecución |
| `docs/KPIs/KPIs.md` | Métricas | Nuevas métricas o benchmarks |
| `docs/adr/` | Decisiones arquitectónicas | Decisiones nuevas o revocadas |
| `CHANGELOG.md` | Historial de cambios | Cada PR o cambio significativo |

---

## ADRs (Architecture Decision Records)

Si querés proponer un cambio que contradice una decisión existente:

1. Leer el ADR original en `docs/adr/`
2. Crear un nuevo ADR con el próximo número disponible (ADR-XXX)
3. Usar estado "Propuesto" hasta su aprobación
4. Referenciar el ADR que se supersede

Formato: usar cualquier ADR existente en `docs/adr/` como plantilla.

---

## Lo que NO se debe hacer

- Hardcodear credenciales de AWS RDS
- Modificar `archive/00_bootstrap/01_legacy_collection.ipynb`
- Commitear archivos `.pt` (modelos YOLO) ni artefactos `.keras`/`.joblib`
- Eliminar tests existentes sin justificación
- Romper compatibilidad con Colab Free
- Modificar esquemas de tablas de BD sin un nuevo ADR
- Eliminar campos HITL de `traffic_classifications`

---

## Guías por Módulo

### Módulo 1 (Preparación de Datos)

- Ejecutar `data_preparation.ipynb` completo después de cambios
- Verificar F1-macro ≥ 0.85 en el conjunto de test
- Leer [ADR-008](docs/adr/ADR-008-tensorflow-keras-traffic-classifier.md) antes de modificar umbrales de auto-etiquetado o arquitectura MLP
- No commitear `*.keras`, `*.joblib`, ni `data/processed/*.csv`

### Módulo 2 (Producción)

- Verificar que el pipeline de percepción produce un DataFrame de telemetría válido
- Verificar que la clasificación asigna uno de 4 estados válidos
- Verificar que la persistencia escribe en ambas tablas (`telemetry_raw` y `traffic_classifications`)
- Leer [ADR-009](docs/adr/ADR-009-modular-three-stage-architecture.md) para la especificación completa

### Módulos Compartidos `src/`

- Todos los módulos deben ser importables desde notebooks (sin entrypoints CLI, sin bloques `if __name__`)
- `config.py` es la fuente única de verdad — no duplicar constantes en otros archivos
- `db.py` es el único punto de configuración de BD — no crear métodos de conexión alternativos
- 19 features en `FEATURE_COLS` son canónicas — no agregar ni quitar sin actualizar todos los módulos y crear un ADR

---

Responsable del documento: Facundo Nicolás González
Fecha de revisión: 2026-07-23
