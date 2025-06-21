from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
model_name = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = model.config.id2label
print(labels)
print("Moods that could be detected by the model:")
for i in range(len(labels)):
    print(f"{i}:{labels[i]}")

#the output we obtained is 
"""{0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
Moods that could be detected by the model:
0:sadness
1:joy
2:love
3:anger
4:fear
5:surprise"""