<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Added for CORS support

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template("index.html")

# Load fine-tuned model and tokenizer
model_path = "./emotion_model/emotion_model/fine_tuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping (based on your label2id used during training) 
id2label = model.config.id2label

# Inference function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_label_id = torch.argmax(probs, dim=1).item()
        pred_label = id2label[pred_label_id]
        confidence = probs[0][pred_label_id].item()
    return pred_label, confidence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    print("Received message:", message)
    
    # Load conversation model
    model_path = "d:/convo_bot/laksh_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        inputs = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_new_tokens=25,       
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply_only = reply[len(message):].strip().split("\n")[0]
        reply_only = reply_only[:200]
        
        print("Generated reply:", reply_only)

        label, confidence = predict_emotion(message)
        return jsonify({
            'emotion': label, 
            'confidence': round(confidence, 3),
            'reply': reply_only
        })
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
from transformers import pipeline

classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
data = [
    "I've been feeling really good about life lately.",
    "Things are finally falling into place!",
    "I'm excited for what's coming next.",
    "I feel confident and full of energy.",
    "Today is one of those days where everything feels right.",
    "Nothing makes me happy anymore.",
    "I just feel empty inside all the time.",
    "Even getting out of bed feels like a struggle.",
    "I don’t enjoy the things I used to.",
    "It’s like there’s a cloud over me that never goes away.",
    "I'm so fed up with everyone.",
    "Why does everything make me so mad?",
    "I can't believe they treated me like that.",
    "Every little thing just pisses me off these days.",
    "I’m ready to explode.",
    "I can't stop overthinking everything.",
    "I'm constantly on edge for no reason.",
    "What if something bad happens and I can’t handle it?",
    "Even small things make me panic now.",
    "I feel like I'm always waiting for something to go wrong.",
    "It doesn’t feel like anything will ever get better.",
    "I don’t see the point of trying anymore.",
    "I feel like a burden to everyone.",
    "The future seems really dark.",
    "Sometimes I wonder if it would matter if I were gone.",
    "I’ve just been going through the motions.",
    "It’s been okay, I guess. Not good, not bad.",
    "I don’t really feel anything these days.",
    "I can’t describe how I feel, just... numb.",
    "Everything feels kind of grey.",
    "I'm smiling, but inside I feel broken.",
    "Things are great, except for this constant dread I can't shake."
]
for i in data:
    result = classifier(i)
    print(i,result)


  # [{'label': 'sadness', 'score': 0.94}]
>>>>>>> 8511609344383d6947dac2347a45c51f711cc90f
