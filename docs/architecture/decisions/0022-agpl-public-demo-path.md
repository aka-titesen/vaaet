# ADR-0022: Vía AGPL-3.0 para demo web pública

- Estado: aceptada
- Fecha: 2026-08-27
- Decisor: Facundo Nicolás González
- Complementa: ADR-0021

## Contexto

VAAET usa `ultralytics-opencv-headless` mediante el extra
`vaaet-core[vision]`. La futura demo académica se desplegará temporalmente en
AWS y su código seguirá siendo público. El gate vigente exige una licencia
Ultralytics Enterprise para cualquier serving, lo que impide innecesariamente
una demostración pública que cumpla AGPL-3.0.

La guía de Ultralytics establece dos vías: publicar bajo AGPL-3.0 el código
fuente correspondiente del proyecto que incorpora YOLO, o contratar una
licencia Enterprise para evitar esas obligaciones. Esta ADR no sustituye el
contrato del bundle v2 ni los límites core--ML--API de ADR-0021.

## Decisión

El monorepo VAAET se relicenciará como `AGPL-3.0-only`. La futura demo pública
podrá ejecutar `vaaet-core[vision]` sin licencia Enterprise sólo cuando cumpla
el checklist AGPL: código fuente público y reproducible, atribución a
Ultralytics, licencia visible y revisión de los activos usados.

El gate de serving ofrecerá dos vías explícitas:

1. **Demo pública AGPL-3.0:** permitida tras completar el checklist público.
2. **Aplicación privada o comercial:** requiere una licencia Ultralytics
   Enterprise vigente, verificada fuera de Git.

No se crea todavía API, frontend, framework, infraestructura AWS, cuenta de
usuario, cola ni persistencia de aplicación. La Web App futura continúa
consumiendo sólo una API HTTP versionada; sus workers usarán `vaaet-core`,
validarán `vaaet.artifacts.validate_manifest()` antes de deserializar y no
expondrán DVC, Drive, PostgreSQL, binarios ni secretos.

## Invariantes

- Se mantienen un único Git y DVC, el bundle v2, las 19 features, las tres
  salidas aprendidas y la política humana exclusiva para `Accident`.
- Videos SISE, datos HITL, credenciales, DSN, Drive, remotos DVC y activos sin
  permiso de redistribución no se publican por la adopción de AGPL.
- Un peso ajustado, bundle o dataset usado en la demo sólo podrá publicarse
  después de registrar su procedencia, licencia, checksum y permiso de
  redistribución.
- La licencia no modifica APIs, versiones de distribuciones ni comportamiento
  ML; cualquier release versionado se decidirá por separado.

## Consecuencias

La licencia AGPL-3.0-only, los metadatos de paquetes, el checklist público y el
runbook temporal ya fueron incorporados al monorepo. Las afirmaciones históricas
se preservan como evidencia de su fecha, pero la documentación activa refleja
las dos vías de serving.

La demostración no tiene costo de licencia Ultralytics bajo AGPL, aunque AWS
puede generar cargos y requerirá límites, revisión de facturación y limpieza
manual. La autorización de relicenciar las contribuciones fue confirmada por
el responsable antes de esta propuesta.

El plan gobernado está en
[`docs/governance/plans/2026-08-27-agpl-public-demo.md`](../../governance/plans/2026-08-27-agpl-public-demo.md).

## Referencias

- [Ultralytics: licencias YOLO](https://docs.ultralytics.com/)
- [Ultralytics: cumplimiento AGPL](https://docs.ultralytics.com/help/contributing/)
