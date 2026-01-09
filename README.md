# TFM-Arquitectura-RAG
Este repositorio contiene el desarrollo de un sistema RAG (Retrieval-Augmented Generation) multimodal aplicado al dominio de rutas de montaña en el Pirineo. El proyecto ha sido realizado como Trabajo Fin de Máster (TFM) y permite recuperar información relevante y generar respuestas en lenguaje natural a partir de documentos que combinan texto e imágenes.

El repositorio incluye un modo de ejecución interactivo mediante la función chat_interactivo(), que permite consultar el sistema RAG Multimodal de forma directa desde la consola.
Este modo está pensado para probar y explorar el comportamiento del sistema realizando preguntas en lenguaje natural sobre las rutas de montaña disponibles en la base de conocimiento.

Al ejecutar chat_interactivo():

- El usuario puede introducir preguntas libremente por consola.

- El sistema recupera los documentos más relevantes mediante búsqueda semántica.

- Se genera una respuesta apoyada en el contexto recuperado.

Cada pregunta se procesa de forma independiente, sin mantener memoria conversacional entre turnos, lo que permite evaluar de manera controlada el rendimiento del sistema ante distintos tipos de consultas
