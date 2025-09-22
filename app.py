import gradio as gr
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import json
import os

# --- Modelo de embeddings para similitud semántica ---
modelo = SentenceTransformer('all-MiniLM-L6-v2')  # Gratis y rápido

# --- Datos básicos ---
servicios = [
    "Desarrollo de páginas web modernas",
    "Soporte técnico en hardware y software",
    "Asesoría tecnológica personalizada",
    "Desarrollo Full Stack",
    "Soluciones Cloud y hosting"
]

proyectos = [
    {"nombre": "VITASUSTEN", "url": "https://lexicorelabssystem.cl/VITASUSTEN/"},
    {"nombre": "Proyecto Telegram POS", "url": "https://github.com/lexicorelabssystem/proyecto-telegram-pos"}
]

inventario = ["Router TP-Link", "Laptop Dell", "Monitor LG", "Teclado Mecánico"]

contacto = "📲 +56 9 1234 5678 | contacto@lexicorelabssystem.cl"

# --- Palabras clave ---
palabras_clave = {
    "saludo": ["hola", "buenas", "hi", "hey"],
    "servicio": ["servicio", "hacen", "ofrecen", "web", "soporte", "full stack"],
    "proyecto": ["proyecto", "trabajo", "muestra", "pagina", "telegram"],
    "inventario": ["producto", "inventario", "tienen", "disponible", "stock"],
    "contacto": ["contacto", "correo", "teléfono", "telefono", "email"]
}

# --- Archivo de aprendizaje ---
archivo_aprendizaje = "aprendizaje.json"
if os.path.exists(archivo_aprendizaje):
    with open(archivo_aprendizaje, "r", encoding="utf-8") as f:
        memoria = json.load(f)
else:
    memoria = {}

# --- Función para guardar aprendizaje ---
def guardar_aprendizaje(pregunta, respuesta):
    memoria[pregunta] = respuesta
    with open(archivo_aprendizaje, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)

# --- Función de respuesta ---
def responder(mensaje, historial):
    mensaje_lower = mensaje.lower()
    respuesta = "Lo siento 🤔 no entendí. Puedes preguntar por 'servicios', 'proyectos', 'inventario' o 'contacto'."

    # --- 1️⃣ Buscar en memoria exacta ---
    for key in memoria:
        if fuzz.ratio(mensaje_lower, key) > 80:
            respuesta = memoria[key]
            break
    else:
        # --- 2️⃣ Buscar coincidencia con palabras clave ---
        for categoria, palabras in palabras_clave.items():
            for palabra in palabras:
                if fuzz.ratio(mensaje_lower, palabra) > 70:
                    if categoria == "saludo":
                        respuesta = "¡Hola! Soy tu asistente de Lexicore IT. Pregunta por 'servicios', 'proyectos', 'inventario' o 'contacto'."
                    elif categoria == "servicio":
                        respuesta = "✅ Nuestros servicios:\n- " + "\n- ".join(servicios)
                    elif categoria == "proyecto":
                        respuesta = "🌐 Algunos proyectos:\n" + "\n".join([f"- {p['nombre']}: {p['url']}" for p in proyectos])
                    elif categoria == "inventario":
                        respuesta = "📦 Productos disponibles:\n- " + "\n- ".join(inventario)
                    elif categoria == "contacto":
                        respuesta = contacto
                    guardar_aprendizaje(mensaje_lower, respuesta)
                    break

        # --- 3️⃣ Similitud semántica ---
        if respuesta.startswith("Lo siento"):
            if memoria:
                preguntas_memoria = list(memoria.keys())
                embeddings = modelo.encode([mensaje_lower] + preguntas_memoria)
                sim = util.cos_sim(embeddings[0], embeddings[1:])[0]
                max_index = sim.argmax()
                if sim[max_index] > 0.65:
                    respuesta = memoria[preguntas_memoria[max_index]]

    # --- Globos estilo chat ---
            historial.append(f"""
    <div style="text-align:right; margin:5px;">
      <div style="display:inline-block; background:#0066cc; color:white;
                  padding:14px 18px; border-radius:20px; max-width:70%;
                  font-size:17px; line-height:1.5; font-family:Segoe UI, sans-serif;
                  box-shadow:0 2px 6px rgba(0,0,0,0.2);">
        {mensaje}
      </div>
    </div>
    """)

    historial.append(f"""
    <div style="text-align:left; margin:5px;">
      <div style="display:inline-block; background:#28a745; color:white;
                  padding:14px 18px; border-radius:20px; max-width:70%;
                  font-size:17px; line-height:1.5; font-family:Segoe UI, sans-serif;
                  box-shadow:0 2px 6px rgba(0,0,0,0.2);">
        {respuesta}
      </div>
    </div>
    """)
   # 👇 Aquí está el retorno correcto (3 valores)
    return "", "\n".join(historial), historial

# --- Interfaz Gradio ---
with gr.Blocks() as demo:
    historial = gr.State([])
    chat = gr.HTML("")
    mensaje = gr.Textbox(placeholder="Escribe algo...", lines=1)
    enviar = gr.Button("Enviar")
    
    # Botones de acceso rápido
    servicios_btn = gr.Button("Servicios")
    proyectos_btn = gr.Button("Proyectos")
    inventario_btn = gr.Button("Inventario")
    contacto_btn = gr.Button("Contacto")

    enviar.click(responder, inputs=[mensaje, historial], outputs=[mensaje, chat, historial])
    servicios_btn.click(lambda _, h: responder("servicios", h), inputs=[mensaje, historial], outputs=[mensaje, chat, historial])
    proyectos_btn.click(lambda _, h: responder("proyectos", h), inputs=[mensaje, historial], outputs=[mensaje, chat, historial])
    inventario_btn.click(lambda _, h: responder("inventario", h), inputs=[mensaje, historial], outputs=[mensaje, chat, historial])
    contacto_btn.click(lambda _, h: responder("contacto", h), inputs=[mensaje, historial], outputs=[mensaje, chat, historial])

demo.launch()
