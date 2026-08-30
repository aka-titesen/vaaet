# ADR-0023: Registro DVC portable por configuración local

- Estado: aceptada
- Fecha: 2026-08-30
- Decisores: Facundo Nicolás González
- Sustituye: la elección de Google Drive por defecto y los remotos de ejemplo de ADR-0011

## Contexto

DVC continúa siendo el registro de los bundles v2, pero un remoto versionado
para Google Drive no permite cambiar de proveedor sin editar el repositorio ni
separa con claridad destinos, perfiles y credenciales. AWS S3 y Cloudflare R2
son alternativas válidas para el mismo cache DVC; la selección no debe afectar
al core ni al formato del bundle.

## Decisión

`.dvc/config` permanece neutral y versionado. Cada entorno configura en
`.dvc/config.local`, ignorado por Git, un único remoto operativo llamado
`vaaet-registry`. Se admiten Google Drive y S3-compatible mediante los extras
`dvc-gdrive` y `dvc-s3`; Cloudflare R2 usa el segundo con su endpoint privado.

Git identifica una versión de bundle mediante commit o tag. DVC guarda los
binarios y `model-manifest.json` describe lifecycle, procedencia, elegibilidad e
integridad. `model_version` es metadato, no un selector único. `current.json`
permanece reservado para snapshots, catálogos HITL y holdouts, nunca para elegir
un bundle DVC.

`vaaet-registry` es el único adaptador de laboratorio para configurar, registrar,
publicar, listar y materializar bundles. Valida el manifiesto antes de `dvc add`,
antes de listar metadatos y antes de devolver una recuperación. No hace commits,
no deserializa modelos, no conoce secretos y no agrega DVC al core.

## Consecuencias

- Un registro compartido tiene un proveedor canónico por vez; varios remotos no
  son fuentes de verdad simultáneas.
- La migración entre proveedores es un runbook manual: se replica el histórico,
  se recuperan revisiones por tag y se validan manifests/checksums antes de
  reemplazar el remoto local.
- CI verifica paquetes, CLI y configuración neutral sin autenticarse ni ejecutar
  `dvc pull` o `dvc push`.
- Drive, S3 y R2 requieren validación manual de permisos y conectividad antes de
  usarse con un bundle real.
