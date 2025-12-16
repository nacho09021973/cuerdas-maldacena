### Nombre del proyecto

**CUERDAS-Maldacena: Motor Holográfico de Geometría Emergente y Diccionario para QFT Reales**

---

### Descripción

Este proyecto implementa un pipeline completo de análisis holográfico tipo AdS/CFT a partir de datos, organizado en tres bloques:

* **Bloque A – Geometría emergente y ecuaciones de campo**
  A partir de datos de borde (CFT sintéticos tipo sandbox), se reconstruye una **geometría de bulk emergente** y se descubren ecuaciones de campo mediante regresión simbólica.

* **Bloque B – Espectro escalar**
  Dadas las geometrías emergentes, un solver escalar tipo Sturm–Liouville calcula el espectro λ_SL y los exponentes UV (Δ_UV), generando un dataset limpio de modos de bulk.

* **Bloque C – Diccionario holográfico y datos reales**
  A partir del dataset (Δ_UV, λ_SL) se aprende un **diccionario emergente λ_SL ↔ Δ**, se construye un atlas holográfico interno y se validan estos resultados frente a sistemas físicos reales (por ejemplo, Ising 3D vía bootstrap) mediante contratos explícitos.

Todo el comportamiento esperado del pipeline, el entorno de referencia (Python, PySR, JuliaCall), los formatos de datos y las reglas de honestidad están descritos en detalle en el README principal del repositorio:
`README_cuerdas_diciembre_2025.md`. 

---

### Instrucciones

1. **Antes de hacer nada, lee el README**

   * Abre y lee `README_cuerdas_diciembre_2025.md` **antes de proponer cambios o escribir código**.
   * Como mínimo, asegúrate de entender las secciones:

     * `0. Cómo usar este README (humano o IA)`
     * `1. Visión global del pipeline`
     * `2. Entorno y dependencias`
     * `3. Formatos de datos`
     * `4. Honestidad y contratos`
     * `5. Guía para IAs colaboradoras`. 

2. **Respeta el entorno y los formatos de datos**

   * No cambies versiones mayores de Python, PySR, JuliaCall o PyTorch sin justificarlo y sin definir un plan de migración + test canario (run mínimo).
   * No renombres campos ni cambies la estructura de los `.h5`, `.csv` o `.json` descritos en el README sin explicar claramente el impacto y cómo mantener compatibilidad. 

3. **Honestidad: no inyectar teoría en la loss**

   * No introduzcas fórmulas teóricas conocidas (por ejemplo, (m^2 L^2 = \Delta(\Delta - d))) ni otras relaciones de diccionario en las funciones de pérdida, features o regularizadores de los scripts 02, 03, 06 o 07.
   * Este tipo de fórmulas solo se pueden usar en análisis y contratos (scripts 08 y 09) como checks post-hoc explícitamente etiquetados. 

4. **CPU por defecto, GPU opcional**

   * El pipeline debe funcionar íntegramente en CPU.
   * Cualquier uso de GPU (por ejemplo, en entrenamiento de modelos en 02 o 07) debe ser opcional, seleccionable (por ejemplo, `--device cpu|cuda`) y no debe romper la ruta CPU-only. 

5. **Qué tipo de ayuda se espera**

   * Tareas deseadas:

     * Refactorizar y limpiar código respetando IO y formatos existentes.
     * Añadir tests unitarios (por ejemplo, para el solver escalar).
     * Diseñar y codificar nuevos contratos físicos en los scripts de contratos (04 y 09).
     * Mejorar logs, mensajes de error y documentación en cabeceras. 
   * Tareas que requieren aprobación explícita:

     * Cambios de formato de ficheros en `runs/`.
     * Cambios de semántica en campos como `provenance` o `dictionary_source`.
     * Introducción de dependencias pesadas o cambios de backend numérico.

6. **No rompas el run mínimo reproducible**

   * Después de cualquier cambio importante, el pipeline debe poder ejecutar el **run mínimo** descrito en la sección 6 del README (sandbox pequeño → geometría emergente → ecuaciones → contratos → dataset de modos → diccionario → atlas → contratos con Ising stub).
   * Si tu cambio impide ese flujo, debes explicar qué parte falla, por qué y cómo se podría corregir. 

