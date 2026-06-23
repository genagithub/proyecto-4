### 📉 Evaluación de Riesgo en Planificaciones Comerciales (Ventas & Producción)

#### 🎯 El Problema de Negocio
La empresa proyecta sus ventas futuras en múltiples canales sin saber con certeza la probabilidad real de concreción de cada nuevo contrato. El objetivo es transformar la previsión comercial tradicional en una herramienta predictiva capaz de anticipar riesgos operativos y optimizar la asignación de recursos antes de la conclusión transaccional.

---

#### 💡 Solución Implementada (Detector de Riesgo)
El núcleo de la solución se enfoca un asistente inteligente para la toma de decisiones comerciales, protegiendo la operación de la empresa a través de tres pilares::
- **Control de Capacidad:** Evalúa la viabilidad de la venta nueva en relación con la capacidad actual de la cadena de producción, evitando la sobreventa.
- **Alertas de Incertidumbre:** Identifica contratos comerciales con baja probabilidad de éxito o que presentan un riesgo crítico de ejecución.
- **Mitigación Preventiva:** El análisis permite aislar las operaciones riesgosas para aplicar acciones correctivas antes de comprometer la capacidad de la empresa.

---

#### 🛠️ Enfoque Técnico y Despliegue
Para la construcción de este prototipo analítico, se desarrolló una arquitectura orientada a la experiencia del usuario técnico y de negocio, priorizando la interpretabilidad del modelo:
- **Procesamiento de Inputs:** El sistema captura de forma dinámica las variables independientes de cada nueva venta ingresada por el usuario para su evaluación inmediata.
- **Visualización Avanzada (PCA):** Implementación de reducción de dimensiones para proyectar y visualizar de forma gráfica la posición de la nueva venta frente al histórico de la empresa.
- **Clasificación y Score de Riesgo:** Entrenamiento de un modelo de clasificación binaria. El porcentaje de riesgo se calcula a partir de la probabilidad y seguridad de confianza del modelo al evaluar la transacción.

---

#### 🚀 El Data Product: Simulador Operativo de Ventas
El resultado final es una Interfaz Interactiva diseñada para los equipos de Ventas y Producción. La herramienta permite:
- Ingresar de forma dinámica los datos y condiciones de una venta nueva.
- Visualizar de inmediato el nivel de riesgo en la conclusión transaccional del acuerdo.
- Identificar alertas tempranas para rebalancear la planificación comercial antes de ejecutar la producción real.
