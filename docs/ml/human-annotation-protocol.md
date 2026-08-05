# Protocolo de anotación humana de estados de tráfico

Este protocolo crea el ground truth que reemplaza gradualmente las etiquetas proxy. El anotador revisa el video y la telemetría de una ventana completa, incluyendo al menos dos minutos anteriores y posteriores, sin ver la salida del modelo.

Por ventana se registra `clip_id`, inicio y fin, estado, confianza del anotador, observaciones, calidad visual, identidad anonimizada del anotador y partición congelada. Los casos ambiguos se conservan como tales y no se fuerzan a una clase.

La muestra inicial debe cubrir al menos 200 ventanas Normal, 200 Reduced, todos los Congested reales y casos cercanos a ambas fronteras. Antes de promover Congested se requieren como mínimo 100 minutos validados de 20 episodios reales, con diversidad de fecha, horario y condiciones visuales.

Un posible accidente requiere revisión humana. La confirmación agrega
`validated_state=3` a `vaaet_feedback.human_validations`, con contexto temporal
revisado y nota; una alerta automática nunca constituye ground truth. El holdout
marcado como `test` queda congelado y no puede utilizarse para ajustar reglas,
umbrales, calibración o arquitectura.

La plantilla canónica está en `data/sample/traffic-state-annotation-template.csv`.
