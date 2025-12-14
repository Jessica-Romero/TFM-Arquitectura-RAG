def llm_build_filters(query: str) -> list | None:
    """
    Extrae filtros usando LLM y los devuelve en formato Chroma/Haystack.
    
    Formato esperado por ChromaEmbeddingRetriever:
    [
        {"field": "meta.poblacion_cercana", "operator": "==", "value": "El Bruc"},
        {"field": "meta.dificultad", "operator": "==", "value": "media"},
    ]
    
    Returns:
        list: Lista de filtros en formato Chroma, o None si no hay filtros
    """
    prompt = f"""
Eres un asistente que extrae filtros estructurados de consultas sobre rutas de montaña.

METADATOS DISPONIBLES (todos comienzan con "meta." en la base de datos):
{json.dumps(FILTER_SCHEMA, indent=2)}

INSTRUCCIONES IMPORTANTES:
1. Analiza la pregunta y extrae SOLO los filtros que el usuario menciona EXPLÍCITAMENTE
2. Devuelve un ARRAY JSON de filtros (lista)
3. Cada filtro debe tener EXACTAMENTE este formato:
   {{
     "field": "meta.<nombre_campo>",  // DEBE comenzar con "meta."
     "operator": "==" | "!=" | ">" | "<" | ">=" | "<=",
     "value": "texto" o número
   }}
4. Para campos de texto usa "=="
5. Para comparaciones numéricas:
   - "más de X km" → operator: ">="
   - "menos de X km" → operator: "<="
   - "entre X y Y" → dos filtros separados
6. Si NO HAY FILTROS, devuelve un array vacío: []

EJEMPLOS DE RESPUESTA:

1. Pregunta: "Rutas en Barcelona con dificultad media"
   Respuesta: [
     {{"field": "meta.provincia", "operator": "==", "value": "Barcelona"}},
     {{"field": "meta.dificultad", "operator": "==", "value": "media"}}
   ]

2. Pregunta: "Sendero de menos de 10 km cerca de Montserrat"
   Respuesta: [
     {{"field": "meta.distancia_total_km", "operator": "<=", "value": 10}},
     {{"field": "meta.zona", "operator": "==", "value": "Montserrat"}}
   ]

3. Pregunta: "Rutas cerca de El Bruc"
   Respuesta: [
     {{"field": "meta.poblacion_cercana", "operator": "==", "value": "El Bruc"}}
   ]

4. Pregunta: "Hola, ¿qué rutas recomiendas?"
   Respuesta: []  // No hay filtros específicos

5. Pregunta: "¿Qué ruta me recomiendas para hacer en familia con niños y que sea circular en la zona de La Selva? "
   Respuesta: [
     {{"field": "meta.zona", "operator": "==", "value": "La Selva"}},
     {{"field": "meta.categoria", "operator": "==", "value": "Familias y niños"}}
   ]

IMPORTANTE: Responde ÚNICAMENTE con el array JSON. No incluyas texto explicativo.
No uses "filters": null ni estructuras anidadas. Solo el array.

PREGUNTA DEL USUARIO: "{query}"

RESPUESTA (solo JSON array):
"""

    try:
        out = gemini_filter_extractor.run(
            parts=[prompt],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 500,
            }
        )
        text = out["replies"][0].strip()
        
        # Limpiar respuesta si viene con markdown
        if text.startswith("```json"):
            text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
        elif text.startswith("```"):
            text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
        
        # Parsear JSON
        data = json.loads(text)
        
        # Validar que sea una lista
        if not isinstance(data, list):
            print(f"LLM no devolvió una lista, devolvió: {type(data)}")
            return None
        
        # Validar y normalizar cada filtro
        validated_filters = []
        for filt in data:
            if not isinstance(filt, dict):
                continue
                
            # Validar campos obligatorios
            if "field" not in filt or "value" not in filt:
                continue
            
            # Asegurar que field empieza con "meta."
            if not filt["field"].startswith("meta."):
                filt["field"] = f"meta.{filt['field']}"
            
            # Establecer operador por defecto si no existe
            if "operator" not in filt:
                # Determinar operador por tipo de campo
                field_name = filt["field"].replace("meta.", "")
                if field_name in ["distancia_total_km", "altitud_minima_m", "altitud_maxima_m", 
                                 "tiempo_total_min", "desnivel_acumulado_pos", "desnivel_acumulado_neg"]:
                    # Campos numéricos: intentar determinar operador
                    if isinstance(filt["value"], (int, float)):
                        filt["operator"] = "=="
                    else:
                        filt["operator"] = "=="
                else:
                    # Campos de texto
                    filt["operator"] = "=="
            
            # Validar operador válido
            valid_operators = ["==", "!=", ">", "<", ">=", "<="]
            if filt["operator"] not in valid_operators:
                filt["operator"] = "=="
            
            # Convertir valores numéricos si es necesario
            try:
                if isinstance(filt["value"], str):
                    # Intentar convertir a número si parece numérico
                    if filt["value"].replace('.', '', 1).isdigit():
                        filt["value"] = float(filt["value"])
                    elif filt["value"].replace(',', '.', 1).replace('.', '', 1).isdigit():
                        # Manejar comas como separador decimal
                        filt["value"] = float(filt["value"].replace(',', '.'))
            except (ValueError, AttributeError):
                pass  # Mantener como string
            
            validated_filters.append(filt)
        
        # Devolver None si la lista está vacía (para consistencia con tu código)
        return validated_filters if validated_filters else None
        
    except json.JSONDecodeError as e:
        print(f" Error parseando JSON del LLM: {e}")
        print(f"Respuesta cruda: {text[:200]}...")
        return None
    except Exception as e:
        print(f" Error en llm_build_filters: {e}")
        return None


# ============================================================
# 3. PIPELINE SOLO DE RETRIEVAL (SIN GENERADOR)
# ============================================================

def build_retrieval_pipeline(
    top_k_text: int = 5,
    top_k_images: int = 4,
) -> Pipeline:
    # Stores de Chroma (los embeddings ya están guardados)
    text_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_COLLECTION,
        distance_function="cosine",
    )

    image_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=IMAGE_COLLECTION,
        distance_function="cosine",
    )

    text_embedder = MiniLMQueryEmbedder()
    image_embedder = CLIPQueryEmbedder()

    text_retriever = ChromaEmbeddingRetriever(
        document_store=text_store,
        top_k=top_k_text,
    )
    image_retriever = ChromaEmbeddingRetriever(
        document_store=image_store,
        top_k=top_k_images,
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("image_embedder", image_embedder)
    pipe.add_component("text_retriever", text_retriever)
    pipe.add_component("image_retriever", image_retriever)

    pipe.connect("text_embedder.embedding", "text_retriever.query_embedding")
    pipe.connect("image_embedder.embedding", "image_retriever.query_embedding")

    return pipe





'''
        filters_list = llm_build_filters(query)
        print("DEBUG filtros LLM (lista):", filters_list)

        filters = None
        if filters_list:
            if len(filters_list) == 1:
                # Un solo filtro simple
                filters = filters_list[0]
            else:
                # AND de todos
                filters = {
                    "operator": "AND",
                    "conditions": filters_list,
                }

        inputs = {
            "text_embedder": {"query": query},
            "image_embedder": {"query": query},
        }

        if filters is not None:
            inputs["text_retriever"] = {"filters": filters}
            inputs["image_retriever"] = {"filters": filters}

        result = pipe.run(inputs)





        # 🔹 FILTRO MANUAL: solo rutas con poblacion_cercana = "El Bruc"
       # filters_el_bruc = {
         #   "field": "meta.poblacion_cercana",
          #  "operator": "==",
           # "value": "El Bruc",
        #}
        #filters = llm_build_filters(query)
        #print("DEBUG filtros LLM:", filters)

        #result = pipe.run(
           # {
               # "text_embedder": {"query": query},
              #  "image_embedder": {"query": query},
             #   "text_retriever": {"filters": filters},
            #    "image_retriever": {"filters": filters},
         #   }
        #)

   
'''