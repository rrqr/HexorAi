import requests
import json

# عنوان الاتصال المحلي لـ Ollama داخل الكود سبيس
OLLAMA_URL = "http://localhost:11434/api/generate"

# اختر النموذج الذي قمت بتحميله
MODEL_NAME = "qwen2:7b"

def ask_ai(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False # لجعل الإجابة تظهر دفعة واحدة وليست تدفقية لتسهيل العرض
    }

    try:
        print("جاري التفكير...")
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "لا توجد استجابة.")
        else:
            return f"خطأ في الاتصال: {response.status_code}"
    except Exception as e:
        return fحدث خطأ: {str(e)}"

if __name__ == "__main__":
    print(f"--- نظام الذكاء الاصطناعي المحلي (مفعل عبر نموذج {MODEL_NAME}) ---")
    print("اكتب 'خروج' أو 'exit' للإنهاء.\n")

    while True:
        user_input = input("أنت: ")
        if user_input.lower() in ["خروج", "exit"]:
            print("إلى اللقاء!")
            break

        if not user_input.strip():
            continue

        ai_response = ask_ai(user_input)
        print(f"\nالذكاء الاصطناعي:\n{ai_response}\n" + "-"*40)
