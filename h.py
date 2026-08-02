import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print("=" * 60)
print("جاري تحميل أوزان النموذج مباشرة إلى رامات الجهاز (RAM)...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("\n[تم بنجاح!] تم تحميل النموذج في الذاكرة الحية.")
print("اكتب سؤالك وابدأ الدردشة (اكتب exit للإنهاء)\n" + "-" * 60)

while True:
    try:
        user_input = input("\nأنت: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "خروج"]:
            print("إلى اللقاء!")
            break
            
        print("النموذج المحلي يكتب الرد...")
        
        # استخدام تنسيق المحادثة الخاص بالنموذج لضمان فهمه واستجابته الصحيحة
        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to("cpu")
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # استخراج النص الجديد فقط (إزالة السؤال الأصلي لتجنب الفراغ أو التكرار)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\nالنموذج المحلي:\n{response}")
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\nتم الإنهاء.")
        break
    except Exception as e:
        print(f"\nحدث خطأ: {str(e)}")
