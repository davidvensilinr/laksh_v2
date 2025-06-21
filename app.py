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
