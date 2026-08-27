# Runbook: demo temporal de VAAET en AWS

Este runbook aplica únicamente después de aprobar una API HTTP versionada y el
[checklist AGPL de demo](../governance/agpl-demo-release-checklist.md). No crea
ni prescribe un framework, servicio AWS o infraestructura permanente.

## Antes de aprovisionar

1. Registrar el tag público, la ventana de inicio y fin, y un responsable de
   apagado. La vía AGPL evita una licencia Enterprise de Ultralytics, pero AWS
   puede generar cargos: configurar límites, alertas de facturación y revisar
   los precios vigentes del servicio elegido antes de iniciar.
2. Usar una cuenta y permisos de mínimo privilegio. No usar la cuenta root ni
   credenciales de larga duración en código, imágenes, notebooks o Git.
3. Guardar secretos sólo en el mecanismo seguro del proveedor. Las variables de
   desarrollo, claves y DSN nunca se imprimen, versionan ni se devuelven por la
   API.
4. Mantener privados los objetos de entrada y salida; bloquear acceso público
   no intencional, cifrar tráfico y almacenamiento, y limitar el acceso al
   worker de la demo.

## Durante la demo

1. El navegador sube un video mediante la API; no recibe rutas locales, bundles,
   DVC, Drive, PostgreSQL ni módulos Python.
2. El worker provee al core un directorio local con el bundle aprobado, ejecuta
   `vaaet.artifacts.validate_manifest()` antes de cargarlo y sólo publica
   resultados de transporte autorizados.
3. Limitar tamaño y duración de uploads según el alcance aprobado de la API; no
   aceptar archivos arbitrarios ni usar datos SISE como muestra pública.
4. Registrar sólo eventos operativos redaccionados: identificador de trabajo,
   tiempos, estado y errores sin payloads, URLs firmadas, secretos o contenido
   de video.
5. La interfaz debe conservar visibles el enlace al tag de código fuente, la
   licencia AGPL y la atribución a Ultralytics.

## Retención y limpieza

1. Definir antes del inicio la hora de vencimiento de cada video de entrada,
   resultado anotado, objeto temporal y log de demo; el valor por defecto es
   borrar al finalizar la ventana registrada.
2. Al cerrar, detener cómputo, eliminar almacenamiento temporal, revocar
   secretos o credenciales efímeras, deshabilitar endpoints y revisar que no
   queden recursos facturables.
3. Conservar sólo evidencia no sensible: tag, fecha, gates aprobados, lista de
   recursos eliminados y confirmación de revisión de facturación.

## Evidencia de cierre

- [ ] La web dejó de aceptar tráfico y los endpoints temporales están apagados.
- [ ] No quedan videos, resultados, snapshots, buckets/objetos o discos de demo
      accesibles fuera de la retención aprobada.
- [ ] Las credenciales temporales fueron revocadas y no aparecen en logs o Git.
- [ ] Se verificó la facturación y no quedan recursos de la demostración.

Si una comprobación falla, detener la demostración, retirar el acceso público y
corregir el incidente antes de reabrirla.
