import warnings
import telebot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# ضع توكن بوت تيليجرام الخاص بك هنا
TOKEN = "8884626544:AAF335PBPhxTY0wdmLinziGA6LPaR-ykKyY"
bot = telebot.TeleBot(TOKEN)

# تحميل النموذج غير المقيد في الذاكرة (RAM)
MODEL_NAME = "cognitivecomputations/dolphin-2.8-mistral-7b-v0.2"

print("جاري تحميل النموذج غير المقيد في رامات الجهاز، يرجى الانتظار...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="cpu"
)

print("[تم بنجاح!] النموذج جاهز ويعمل الآن عبر بوت تيليجرام.")

# استقبال الرسائل عبر بوت تيليجرام
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    chat_id = message.chat.id
    
    bot.send_message(chat_id, "جاري المعالجة محلياً...")
    
    try:
        # تجهيز المدخلات بتنسيق النموذج
        prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        
        # توليد الرد بدون قيود
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # إرسال النتيجة إلى تيليجرام
        bot.send_message(chat_id, response if response else "عذراً، لم يتم إنشاء رد.")
        
    except Exception as e:
        bot.send_message(chat_id, f"حدث خطأ أثناء التوليد: {str(e)}")

if __name__ == "__main__":
    bot.infinity_polling()
