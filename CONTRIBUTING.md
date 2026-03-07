# Contribuir a VAAET

Gracias por tu interés en contribuir a VAAET. Este documento establece las convenciones y reglas para modificar el proyecto.

## Arquitectura — Reglas Fundamentales

1. **Todo el código vive en `vaaet.ipynb`** — no crear módulos `.py` separados
2. **Ejecutar en Google Colab** — toda modificación debe ser compatible con Colab Free Tier
3. **Leer `AGENTS.md`** antes de empezar — contiene los límites arquitectónicos y el sistema Always/Ask/Never

## Convenciones de Código

### Estilo General

- Python 3.8+ compatible
- PEP 8 para formato (excepto longitud de línea, que puede extenderse en notebooks)
- Type hints en funciones públicas de `VAAETHybrid`
- Docstrings en español para todas las funciones

### Prints con Emoji

El sistema usa `print()` con prefijos emoji en lugar de `logging`:

```python
print("✅ Operación exitosa")
print("⚠️ Advertencia: parámetro fuera de rango")
print("🔴 Error: no se pudo conectar a la BD")
print("📊 Resultado: 42 vehículos detectados")
```

### Configuración

- **Parámetros del puente**: Solo en Cell 5 (`BRIDGE_CONFIG`)
- **Parámetros de BD**: Solo en Cell 1
- **Credenciales**: NUNCA hardcodeadas. Usar variables de entorno o `getpass`

## Flujo de Contribución

1. Leer el ADR relevante en `Docs/adr/` si tu cambio afecta una decisión arquitectónica
2. Si no hay ADR y tu cambio es significativo, redactar uno antes de implementar
3. Verificar que `test_sistema()` (Cell 7) pasa después de tus cambios
4. Generar un video demo sintético (Cells 8-9) para validar end-to-end
5. Actualizar la documentación correspondiente si tu cambio altera comportamiento observable

## Estructura de Documentación

| Archivo | Propósito | Cuándo actualizar |
|---|---|---|
| `README.md` | Visión general y uso | Cambios en features o dependencias |
| `AGENTS.md` | Contexto para agentes IA | Cambios en arquitectura o reglas |
| `Docs/PRD.md` | Requisitos del producto | Nuevos requisitos o cambios funcionales |
| `Docs/DDS.md` | Diseño técnico | Cambios en algoritmos o componentes |
| `Docs/GUIA_USUARIO.md` | Guía de usuario | Cambios en UX o flujo de ejecución |
| `Docs/KPIs/KPIs.md` | Métricas | Nuevas métricas o benchmarks |
| `Docs/adr/` | Decisiones arquitectónicas | Decisiones nuevas o revocadas |
| `CHANGELOG.md` | Historial de cambios | Cada PR o cambio significativo |

## ADRs (Architecture Decision Records)

Si quieres proponer un cambio que contradiga una decisión existente:

1. Lee el ADR original en `Docs/adr/`
2. Crea un nuevo ADR con el siguiente ADR-XXX disponible
3. Usa el estado "Propuesto" hasta que sea aprobado
4. Referencia el ADR que se está superando

Formato: ver cualquier ADR existente en `Docs/adr/` como plantilla.

## Lo que NO se debe hacer

- Hardcodear credenciales de AWS RDS
- Crear archivos `.py` fuera del notebook
- Hacer commit de archivos `.pt` (modelos YOLO)
- Eliminar `test_sistema()` o el generador de demos
- Romper compatibilidad con Colab Free Tier
