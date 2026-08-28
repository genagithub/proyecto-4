### ⚠️ Evaluación de Riesgo de Fracaso en Planificaciones Comerciales

#### 🎯 El Contexto del Problema 
La empresa proyecta sus ventas futuras en múltiples canales comerciales sin saber con certeza la probabilidad real de concreción y los riesgos asociados a cada nuevo contrato, al operar bajo un baseline reactivo y conservador debido a que las pérdidas transaccionales y los fallos logísticos representan una pequeña proporción del total de las ventas, la gestión interna los asume como eventos poco probables, fortuitos o inherentemente difíciles de anticipar. El equipo técnico seleccionó estrategicamente las variables requeridas del conjunto, por lo que el objetivo es transformar la previsión comercial tradicional en una herramienta prescriptiva capaz de anticipar riesgos operativos y asegurar la viabilidad de la ejecución antes de la conclusión transaccional.

---

#### 🛠️  Enfoque Técnico y Modelado
Se diseñó un modelo de scoring transaccional enfocado en la detección precisa de anomalías operativas, la arquitectura interna del sistema se construyó bajo un enfoque de metamodelado, utilizando como algoritmo base K-Nearest Neighbors y Random Forest para el metaaprendizaje. Además, para una optimización mayor, se aplicaron técnicas de Feature Engineering como las siguientes:

- **Definición de Variable Objetivo:** Se unificaron y simplificaron múltiples motivos de pérdida operativa (cancelaciones, fraudes o fallos logísticos) en una única variable objetivo binaria.
- **Estrategia para Datos Desbalanceados:** Dado que las transacciones fallidas representan una minoría estadística crítica (9% del histórico), se aplicaron técnicas de balanceo de clases para evitar el sesgo hacia la clase mayoritaria.
- **Numerización y Estandarización de Variables:** Se implementaron pipelines de codificación Target para las variables categóricas y estandarización sin valores Outliers para las continuas evitando un sesgo de influencia.



---

#### 🚀 Solución Analítica: Simulador Operativo de Ventas
El resultado final es una herramienta interactiva basada en un asistente inteligente para la toma de decisiones comerciales, protegiendo la operación de la empresa a través de dos pilares:
- **Procesamiento de Inputs:** Permite la introducción de combinaciones logísticas como el destino, el tipo de envío, los días de entrega estimados, entre otras características generando múltiples escenarios posibles.
- **Visualización Reducida:** Implementa visualización de puntos de dispersión, permitiendo observar de forma gráfica la posición de la nueva planificación comercial frente al histórico de la organización.
- **Clasificación y Score de Riesgo:** Realiza el análisis de probabilidad de cierre (Win/Loss Probability) o Churn del pipeline, retornando un valor que representa el porcentaje de fracaso sin especificar el tipo.

---

#### 📌 Propósito de este Proyecto: Impacto Directo

- **Mitigación Preventiva:** Dota al equipo de ventas y operaciones la ventaja competitiva de anticipar la viabilidad real de los nuevos contratos antes de su ejecución y transformar la incertidumbre de las ventas en previsibilidad operativa.
