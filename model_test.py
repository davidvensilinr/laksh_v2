from transformers import pipeline

classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
while True:
    i=input("You : ")
    if i=="exit":
        print("Laksh: Bye")
        break
    result = classifier(i)
    print("Laksh: You are feeling",result)


  # [{'label': 'sadness', 'score': 0.94}]
