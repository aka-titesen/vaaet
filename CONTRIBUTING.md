# Guía de contribución — Monorepo VAAET

Leé [AGENTS.md](AGENTS.md) y el ADR aplicable antes de modificar el repositorio.

- Los cambios ML se realizan desde `vaaet-ml/`; sus reglas y comandos están en
  [`vaaet-ml/CONTRIBUTING.md`](vaaet-ml/CONTRIBUTING.md).
- La documentación y configuración compartida se mantienen en la raíz.
- No crear repositorios Git o remotos DVC anidados.
- No agregar código a `vaaet-app/` hasta aprobar el contrato HTTP y el alcance
  del componente.

Los cambios arquitectónicos deben actualizar su ADR y el plan gobernado
correspondiente. No incluir secretos, binarios ML ni datos sensibles en Git.
