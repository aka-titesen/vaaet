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
- Vulnerabilidades en dependencias listadas en `requirements.txt`
- Inyección SQL en consultas de `src/persistence.py` o `src/db.py`
- Acceso no autorizado a la instancia de AWS RDS

### Fuera del alcance
- Seguridad de la infraestructura de Google Colab (responsabilidad de Google)
- Seguridad de AWS RDS (responsabilidad del administrador de la instancia)
- Contenido de los videos de vigilancia SISE (datos de terceros)

## Prácticas de Seguridad Implementadas

- Las credenciales se obtienen exclusivamente por variables de entorno o `getpass`
- Ningún secreto es hardcodeado en el código fuente
- El archivo `.env` está en `.gitignore`
- Los archivos de modelo (`.keras`, `.joblib`, `.pt`) están en `.gitignore`
- La persistencia en BD usa consultas parametrizadas vía SQLAlchemy (prevención de inyección SQL)
- El sistema degrada silenciosamente si las credenciales no están presentes

## Dependencias y Actualizaciones

Las dependencias se revisan periódicamente. Para verificar vulnerabilidades conocidas:

```bash
pip audit
```

## Versiones Soportadas

| Versión | Soportada |
|---|---|
| 3.x.x | ✅ Activa |
| 2.x.x | ❌ Sin soporte |
| 1.x.x | ❌ Sin soporte |

---

Responsable: Facundo Nicolás González
Fecha de revisión: 2026-07-23
