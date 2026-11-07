### ⚠️ Scoring de riesgo sobre planificaciones comerciales

#### 🎯 Contexto del problema de negocio 
Los sectores de Ventas y Operaciones ejecutan sus múltiples transacciones sin saber con certeza la probabilidad real de concreción y los riesgos asociados a cada nuevo contrato, al operar bajo un baseline reactivo y conservador debido a que las pérdidas transaccionales y los fallos logísticos representan una pequeña proporción del total de las ventas, la gestión interna los asume como eventos poco probables, fortuitos o inherentemente difíciles de anticipar. 

Como etapa previa, desde el data warehouse se seleccionaron los atributos clave y se agruparon estratégicamente los datos requeridos para el proyecto. Por lo que el objetivo es transformar la previsión comercial tradicional en una herramienta prescriptiva capaz de anticipar riesgos operativos y asegurar la viabilidad de la ejecución antes de la conclusión transaccional.

---

#### 🛠️  Enfoque técnico y modelado
Se diseñó un modelo de scoring transaccional enfocado en la detección precisa de anomalías operativas, la arquitectura interna del sistema se construyó bajo un enfoque de metamodelado, utilizando como algoritmo base K-Nearest Neighbors y Random Forest para el metaaprendizaje. Además, para una optimización mayor, se aplicaron técnicas de Feature Engineering como las siguientes:

- **Definición de variable objetivo:** Se unificaron y simplificaron múltiples motivos de pérdida operativa (cancelaciones, fraudes o fallos logísticos) en una única variable objetivo binaria.
- **Estrategia para datos desbalanceados:** Dado que las transacciones fallidas representan una minoría estadística crítica (9% del histórico), se aplicaron técnicas de balanceo de clases para evitar el sesgo hacia la clase mayoritaria.
- **Numerización y estandarización de variables:** Se implementaron pipelines de codificación Target para las variables cualitativas y estandarización (con capeo de outliers) para las continuas evitando un sesgo de influencia.

---

#### 🚀 Solución analítica: Simulador operativo de ventas
El resultado final es una herramienta interactiva basada en un asistente inteligente para la toma de decisiones comerciales, protegiendo la operación de la empresa a través de dos pilares:
- **Procesamiento de inputs:** Permite la introducción de combinaciones logísticas como el destino, el tipo de envío, los días de entrega estimados, entre otras características generando múltiples escenarios posibles.
- **Visualización reducida:** Implementa visualización de puntos de dispersión, permitiendo observar de forma gráfica la posición de la nueva planificación comercial frente al histórico de la organización.
- **Clasificación y score de riesgo:** Realiza el análisis de probabilidad de cierre (Win/Loss Probability) o Churn del pipeline, retornando un valor que representa el porcentaje de fracaso sin especificar el tipo.

---

#### 📌 Propósito: Impacto directo

- **Mitigación preventiva:** Al integrarse como un asistente inteligente de scoring y simulación permite al equipo de Ventas y Operaciones introducir variables logísticas para proyectar escenarios y visualizar gráficamente el porcentaje de riesgo de fracaso de una nueva planificación frente al histórico de la empresa. Su implementación transforma la gestión reactiva tradicional en una estrategia preventiva, permitiendo a la gerencia validar la viabilidad real de cada contrato antes de su ejecución, blindando el margen transaccional y facilitando una operación logística con cero incertidumbre.
