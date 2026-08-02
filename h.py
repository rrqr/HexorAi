import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# اختيار نموذج خفيف ومحلي بالكامل
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print("جاري تحميل أوزان النموذج مباشرة إلى رامات الجهاز (RAM)...")

# تحميل التوكنايزر والنموذج
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# تحميل النموذج واعتماد الـ CPU والـ RAM (أو الـ GPU إذا توفر)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, # استخدام الدقة العادية للـ CPU
    device_map="cpu"           # إجبار النموذج على استخدام رامات الجهاز ومعالجه مباشرة
)

print("تم تحميل النموذج بنجاح في الذاكرة! يمكنك البدء بالدردشة:")

while True:
    prompt = input("\nأنت: ")
    if prompt.lower() == "exit":
        break
        
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # توليد الرد اعتماداً على العتاد المحلي
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nالنموذج المحلي (عبر رامات الجهاز):\n{response}")
