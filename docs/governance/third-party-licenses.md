# Registro de licencias de terceros

Este registro describe gates de uso y no reemplaza asesoramiento legal ni
contratos privados. No se registran en Git números de licencia, contratos,
credenciales ni información comercial.

| Componente | Uso VAAET | Licencia upstream | Vías de despliegue |
| --- | --- | --- | --- |
| `ultralytics-opencv-headless` | Extra `vaaet-core[vision]` para YOLO | AGPL-3.0 o Enterprise | Demo pública AGPL-3.0 con checklist completo, o Enterprise para aplicación privada/comercial. |

## Gate de serving Ultralytics

VAAET se distribuye bajo AGPL-3.0-only. Antes de desplegar un worker de API que
ejecute `vaaet.vision`, el responsable elige y registra una de estas vías:

1. **Demo pública AGPL-3.0:** el código fuente correspondiente es público,
   reproducible y atribuye a Ultralytics; los modelos, bundles y datos usados
   tienen procedencia y permiso de redistribución verificados.
2. **Aplicación privada o comercial:** existe una licencia Ultralytics
   Enterprise vigente y aplicable, verificada en el registro privado de compras
   o legal.

El registro público puede declarar la vía elegida y el estado del gate, pero
nunca contratos, números de licencia, credenciales ni evidencia comercial.
Sin una vía completa, `vaaet-app/` permanece reservado y no se habilita serving
web con el extra `vision`. Esta política sigue las opciones de licencia
publicadas por [Ultralytics](https://docs.ultralytics.com/). La vía AGPL se
verifica con el [checklist de demo pública](agpl-demo-release-checklist.md).
