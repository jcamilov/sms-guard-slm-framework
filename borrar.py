import os
from langfuse import Langfuse
from langfuse.model import InitialGeneration, InitialScore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configura Langfuse (asegúrate de que tus variables de entorno estén seteadas)
# os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
# os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
# os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # O tu host local/self-hosted

langfuse = Langfuse()

# Dataset de prueba (simplificado para el ejemplo)
test_dataset = [
    {"sms_text": "Haga clic aquí para actualizar su información bancaria: [enlace fraudulento]", "label": "smishing"},
    {"sms_text": "Su paquete está en camino. Entrega estimada mañana.", "label": "benign"},
    {"sms_text": "Urgente: Su cuenta ha sido suspendida. Verifique su identidad en: [enlace]", "label": "smishing"},
    {"sms_text": "Recordatorio: cita con el dentista el lunes a las 10 AM.", "label": "benign"},
    {"sms_text": "Ganaste $1,000,000! Envía tus datos bancarios para reclamar.", "label": "smishing"},
    # ... 95 SMS más
]

# Definición de modelos
models = {
    "openai_gpt4o": ChatOpenAI(model="gpt-4o", temperature=0),
    "openai_gpt35_turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    # "anthropic_claude": ChatAnthropic(model="claude-3-opus-20240229", temperature=0), # Si usas Anthropic
}

# Definición de prompts
prompts = {
    "prompt_basico": ChatPromptTemplate.from_messages([
        ("system", "Clasifica el siguiente SMS como 'smishing' o 'benign'. Responde solo con 'smishing' o 'benign'."),
        ("user", "SMS: {sms_text}")
    ]),
    "prompt_con_ejemplos": ChatPromptTemplate.from_messages([
        ("system", "Clasifica el siguiente SMS como 'smishing' o 'benign'. Aquí tienes algunos ejemplos:\n\nSmishing ejemplos:\n- 'Estimado cliente, su cuenta ha sido bloqueada. Verifique su identidad aquí: [link]'\n- 'Usted ha ganado un premio! Haga clic para reclamar: [link]'\n\nBenign ejemplos:\n- 'Su paquete ha sido entregado.'\n- 'Hola, ¿cómo estás?'\n\nResponde solo con 'smishing' o 'benign'."),
        ("user", "SMS: {sms_text}")
    ]),
    "prompt_rol_experto": ChatPromptTemplate.from_messages([
        ("system", "Eres un analista de ciberseguridad experto. Tu tarea es clasificar los SMS entrantes como 'smishing' (si intentan engañar o robar información) o 'benign' (si son inofensivos). Evalúa cuidadosamente el contenido, la fuente y cualquier solicitud. Responde solo con 'smishing' o 'benign'."),
        ("user", "SMS: {sms_text}")
    ]),
}

# Ejecución del experimento
for model_name, model_instance in models.items():
    for prompt_name, prompt_template in prompts.items():
        print(f"Running experiment for Model: {model_name}, Prompt: {prompt_name}")

        for i, sms_data in enumerate(test_dataset):
            sms_text = sms_data["sms_text"]
            true_label = sms_data["label"]

            # Cada inferencia es una traza
            with langfuse.trace(
                name=f"smishing-classification-{model_name}-{prompt_name}-{i}",
                metadata={
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "sms_id": i,
                    "input_sms": sms_text,
                    "ground_truth_label": true_label,
                }
            ) as trace:
                chain = prompt_template | model_instance | StrOutputParser()

                try:
                    prediction = chain.invoke({"sms_text": sms_text})

                    # Registrar la predicción y el score (evaluación)
                    trace.generation(
                        name="sms-classification-prediction",
                        input={
                            "sms_text": sms_text,
                            "prompt_template": prompt_template.format(sms_text=sms_text) # Guardar el prompt renderizado
                        },
                        output=prediction,
                    )

                    # Evaluar la predicción
                    is_correct = (prediction.strip().lower() == true_label.strip().lower())
                    score_name = "correctness" # Nombre de tu métrica de evaluación

                    trace.score(
                        name=score_name,
                        value=1 if is_correct else 0,
                        comment=f"Prediction: {prediction}, Ground Truth: {true_label}"
                    )

                except Exception as e:
                    print(f"Error processing SMS {i} with {model_name}/{prompt_name}: {e}")
                    trace.generation(
                        name="sms-classification-prediction",
                        input={"sms_text": sms_text},
                        output=f"ERROR: {e}",
                        status_message=str(e),
                        status_code=400, # O un código de error apropiado
                    )
                    trace.score(
                        name="correctness",
                        value=0,
                        comment=f"Error during inference: {e}"
                    )

print("Experiment complete. Check your Langfuse dashboard for results.")