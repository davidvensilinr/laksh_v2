# importing for loading tokenizer and emotional classification model
from transformers import AutoTokenizer,AutoModelForSequenceClassification
#for deep learning
import torch
#neural network functions in the aspect of deep learning
import torch.nn.functional as F 
#model name for emotion classification which is a fine tuned
#model of bert model in the aspect of emotions
model_name="nateraw/bert-base-uncased-emotion"
#tokenizing the inputs with the help of the model
tokenizer=AutoTokenizer.from_pretrained(model_name)
#classifying the emotions of the inputs using the model
model=AutoModelForSequenceClassification.from_pretrained(model_name)
#switching the model to evaluation mode
model.eval()
# setting the emotion labels
label=model.config.id2label
labels=[label[i] for i in range(len(label))]
print("Moods that could be detected by the model:")
print(labels)

#function to predict emotion
def predict_emotion(text):
    #pt stands for pytorch which is used to convert input to tensors(vectors)
    inputs=tokenizer(text,return_tensors="pt")
    #just read don't learn
    with torch.no_grad():
        #unpacks the dictonary into individual arguments
        
        outputs=model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        score = probs[0][predicted_class].item()
        labels = model.config.id2label
        return labels[predicted_class], score

test_data = [
    ("I’m so happy to see you!", "joy"),
    ("I feel empty and hopeless.", "sadness"),
    ("Why did you do that?!", "anger"),
    ("I'm scared something bad will happen.", "fear"),
    ("You mean the world to me.", "love"),
    ("What a surprise, I wasn't expecting this!", "surprise"),
    ("Everything feels meaningless.", "sadness"),
    ("You’re seriously getting on my nerves.", "anger"),
    ("I’m feeling really great today!", "joy"),
    ("I just want to disappear.", "sadness"),
    ("I’m so happy to see you!", "joy"),
    ("I feel empty and hopeless.", "sadness"),
    ("Why did you do that?!", "anger"),
    ("I'm scared something bad will happen.", "fear"),
    ("You mean the world to me.", "love"),
    ("What a surprise, I wasn't expecting this!", "surprise"),
    ("Everything feels meaningless.", "sadness"),
    ("You’re seriously getting on my nerves.", "anger"),
    ("I’m feeling really great today!", "joy"),
    ("I just want to disappear.", "sadness"),
    ("Thank you so much, this means a lot to me!", "joy"),
    ("I hate the way you talk to me.", "anger"),
    ("I can't stop crying lately.", "sadness"),
    ("You really care about me?", "love"),
    ("This came out of nowhere!", "surprise"),
    ("I’m terrified of losing you.", "fear"),
    ("Wow! I didn’t expect that!", "surprise"),
    ("I'm so grateful for your support.", "love"),
    ("It’s hard to trust anyone now.", "fear"),
    ("You always make me smile.", "joy")
]
correct = 0

for text, actual_label in test_data:
    predicted_label, score = predict_emotion(text)
    print(f"Text: {text}")
    print(f"Actual: {actual_label} | Predicted: {predicted_label} ({score:.2f})")
    print("✅ Correct" if predicted_label == actual_label else "❌ Incorrect")
    print("---")
    if predicted_label == actual_label:
        correct += 1

accuracy = correct / len(test_data)
print(f"\n📊 Accuracy on Test Set: {accuracy * 100:.2f}%")
