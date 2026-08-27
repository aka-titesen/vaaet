# Política de Seguridad — VAAET

## Reporte de Vulnerabilidades

Si descubrís una vulnerabilidad de seguridad en VAAET, por favor **no** la reportes como un Issue público. En su lugar:

1. Enviá un correo a: **[dirección de contacto del responsable]**
2. Incluí una descripción detallada de la vulnerabilidad
3. Si es posible, adjuntá pasos para reproducir el problema
4. Recibirás una confirmación dentro de las 48 horas hábiles

## Alcance de la Seguridad

### Dentro del alcance
- Exposición de credenciales de base de datos en outputs de notebooks
- Vulnerabilidades en dependencias declaradas en `vaaet-core/pyproject.toml` o
  `vaaet-ml/pyproject.toml`
- Inyección SQL en `vaaet-ml/src/vaaet/data/persistence.py` o `vaaet-ml/src/vaaet/data/database.py`
- Acceso no autorizado a cualquier instancia PostgreSQL configurada para VAAET

### Fuera del alcance
- Seguridad de la infraestructura de Google Colab (responsabilidad de Google)
- Cifrado en reposo, parcheo y logs del proveedor PostgreSQL (responsabilidad administrativa)
- Contenido de los videos de vigilancia SISE (datos de terceros)

## Prácticas de Seguridad Implementadas

- Las credenciales se obtienen mediante Colab Secrets o variables de entorno;
  `.env` sólo se carga explícitamente en desarrollo local
- Ningún secreto es hardcodeado en el código fuente
- El archivo `.env` está en `.gitignore`
- Los archivos de modelo (`.keras`, `.joblib`, `.pt`) están en `.gitignore`
- La persistencia en BD usa consultas parametrizadas vía SQLAlchemy (prevención de inyección SQL)
- TLS `verify-full` es el valor recomendado y `disable` sólo funciona en localhost
- Cuatro roles aplican mínimo privilegio; Alembic y el administrador no se usan en Colab
- `vaaet_ops.pipeline_runs` registra categorías de error sin mensajes, DSN ni secretos
- La ausencia de credenciales deshabilita la persistencia de forma visible y conserva outputs locales

## Dependencias y Actualizaciones

Las dependencias se revisan periódicamente. Para verificar vulnerabilidades conocidas:

```bash
pip audit
```

## Versiones Soportadas

| Versión | Soportada |
|---|---|
| 4.x.x | ✅ Activa |
| 3.x.x | ❌ Sin soporte |
| 2.x.x | ❌ Sin soporte |
| 1.x.x | ❌ Sin soporte |

---

Responsable: Facundo Nicolás González
Fecha de revisión: 2026-08-06
