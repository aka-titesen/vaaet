# ADR-0020: Monorepo Git único y límite de aplicación

- Estado: aceptada; actualizada por ADR-0021 para la separación core/laboratorio
- Fecha: 2026-08-25
- Decisores: Facundo Nicolás González
- Sustituye: la decisión multi-repo de ADR-0012; conserva su contrato portable

## Contexto

VAAET ML necesita convivir en el futuro con una API de serving y una Web App sin
duplicar contratos, repositorios Git ni remotos DVC. ADR-0012 establecía dos
repositorios, pero el ciclo de evolución de los componentes requiere ahora una
raíz compartida con límites de despliegue explícitos.

## Decisión

VAAET usa una única raíz Git y DVC:

```text
vaaet/
├─ vaaet-ml/       # paquete Python, notebooks, tests, migraciones y artefactos
├─ vaaet-app/      # reservado para API y Web App futuras
├─ docs/           # ADRs, contratos y documentación compartida
├─ .dvc/           # única configuración y únicos remotos DVC
└─ .github/        # automatización de raíz
```

La distribución `vaaet-ml` y los contratos ML vigentes se conservan. La
propiedad del import portable `vaaet` y la separación `vaaet_ml` de laboratorio
se definen en ADR-0021.
El bundle v2 de cuatro archivos continúa siendo el único intercambio entre ML y
serving. DVC conserva sus remotos en la raíz y gobierna
`vaaet-ml/artifacts/traffic-state/` como una unidad atómica.

`vaaet-app/` no contiene código de aplicación hasta que exista un alcance
aprobado. Cuando se implemente, la API instalará `vaaet-ml` desde el workspace,
validará `vaaet.artifacts.validate_manifest()` antes de deserializar un bundle y
publicará un contrato HTTP versionado. La Web App sólo consumirá esa API; no
accederá a PostgreSQL, DVC, Google Drive, binarios ML ni módulos Python.

## Invariantes

- Se mantienen las 19 `FEATURE_COLS`, las tres salidas aprendidas, los cuatro
  estados públicos y la confirmación humana exclusiva de `Accident`.
- No cambian el MLP, umbrales, esquema PostgreSQL, roles, secretos, remotos DVC
  ni el formato del bundle v2.
- La reubicación no promociona artefactos ni cambia su elegibilidad, lineage,
  snapshots, holdouts, catálogos o input locks.
- Los notebooks continúan siendo orquestadores Colab y la lógica reutilizable
  se separa en el core portable y el laboratorio según ADR-0021.

## Consecuencias

La migración se realiza en etapas trazables con `git mv`: primero gobierno y
frontera, luego traslado mecánico y adaptación de tooling, y por último la
simplificación de notebooks. No se combina con cambios de modelo, dependencias,
esquema, permisos ni aplicación web. CI se mantiene en la raíz y se acota por
rutas cuando haya componentes ejecutables independientes.

El plan gobernado de esta migración se registra en
[`docs/governance/plans/2026-08-25-monorepo-ml-app-boundary.md`](../../governance/plans/2026-08-25-monorepo-ml-app-boundary.md).
