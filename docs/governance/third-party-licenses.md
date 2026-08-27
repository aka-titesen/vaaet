# Registro de licencias de terceros

Este registro describe gates de uso y no reemplaza asesoramiento legal ni
contratos privados. No se registran en Git números de licencia, contratos,
credenciales ni información comercial.

| Componente | Uso VAAET | Licencia upstream | Gate de despliegue |
| --- | --- | --- | --- |
| `ultralytics-opencv-headless` | Extra `vaaet-core[vision]` para YOLO | AGPL-3.0 o Enterprise | Una API o Web App que ejecute visión requiere licencia Enterprise válida. |

## Gate Ultralytics Enterprise

VAAET conserva su licencia MIT, pero el uso y la redistribución del extra
`vision` también quedan sujetos a la licencia upstream. Antes de desplegar un
worker de API que ejecute `vaaet.vision`, el responsable autorizado debe
verificar en el registro privado de compras o legal que existe una licencia
Ultralytics Enterprise vigente y aplicable al entorno. El release checklist
público sólo puede indicar `Enterprise license verified: yes/no`, sin adjuntar
evidencia privada.

Sin esa verificación, `vaaet-app/` permanece reservado y no se habilita serving
web con el extra `vision`. Esta política sigue las opciones de licencia
publicadas por [Ultralytics](https://docs.ultralytics.com/).
