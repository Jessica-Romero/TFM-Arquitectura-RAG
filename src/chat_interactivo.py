from rag_multimodal import crear_chatbot
from config import init_process, TEXT_COLLECTION, IMAGE_COLLECTION, TEXT_EMBED_MODEL, CHROMA_PERSIST_DIR, BASE_DIR,API_KEY
from pathlib import Path

def chat_interactivo():
    """Chat interactivo con RAG multimodal por consola."""
    print("\n" + "="*60)
    print("💬 CHAT RAG MULTIMODAL")
    print("   Escribe 'salir', 'exit' o 'quit' para terminar")
    print("   Escribe 'limpiar' para limpiar historial")
    print("="*60 + "\n")
    
    bot = crear_chatbot(verbose=True,top_k_text=7)

    while True:
        try:
            query = input("Tu pregunta: ").strip()
            
            if not query:
                continue
            
            # Comandos especiales
            if query.lower() in {"salir", "exit", "quit"}:
                print("\n👋 ¡Hasta luego!")
                break
            
            if query.lower() == "limpiar":
                bot.limpiar_historial()
                print(" Historial limpiado\n")
                continue
            
            # Procesar pregunta
            print("\n" + "─" * 40)
            print(f"🤔 Procesando: {query}")
            print("─" * 40)
            
            resultado = bot.preguntar(query, mostrar_info=True)
            
            print("\n" + "🔵" * 30)
            print("RESPUESTA Generada:")
            print("🔵" * 30)
            print(resultado["respuesta"])
            print("\n" + "─" * 40)
            
        except KeyboardInterrupt:
            print("\n\n Interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\n Error: {e}")

if __name__ == "__main__":
    print("Iniciando Chat RAG Multimodal...")

    BASE_DIR = Path(__file__).resolve().parents[1]

    init_process(
        TEXT_COLLECTION_param="rutas_text_waypoints",
        IMAGE_COLLECTION_param="rutas_imagenes",
        TEXT_EMBED_MODEL_param="sentence-transformers/all-MiniLM-L6-v2",
        PATH_CHROMA_DIR_param=BASE_DIR / "src" / "chroma_db",
        BASE_DIR_param=BASE_DIR
    )

    chat_interactivo()