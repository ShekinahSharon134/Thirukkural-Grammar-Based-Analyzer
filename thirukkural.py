from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tamil.utf8 import get_letters, splitMeiUyir

def read_thirukkural_from_file(file_path):
    with open("thiru1.txt", encoding="utf-8") as file:
        return file.readlines()

file_path = 'thirukkural.txt'  
kurals = read_thirukkural_from_file(file_path)

for sentence in kurals:
    sentence = sentence.strip()  
  
    words = sentence.split()
    if len(words) != 7:
        print(f"Thirukkural doesn't follow the correct format: {sentence}")
        continue

    top_line = " ".join(words[:4])
    bottom_line = " ".join(words[4:])
    input_text = sentence
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([input_text] + kurals)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_idx = cosine_similarities.argmax()
    best_score = cosine_similarities[best_idx]
    best_kural = kurals[best_idx]
    kural_words = best_kural.strip().split()
    line1 = " ".join(kural_words[:4])
    line2 = " ".join(kural_words[4:])

    print("\nClosest Matching Kural:")
    print(line1)
    print(line2)
    print("***************************************************************************************")
    print("Input follows the Thirukkural format: 4 words in the first line and 3 in the second.")
    print("***************************************************************************************")

    kuril = set(['அ', 'இ', 'உ', 'எ', 'ஒ'])
    nedil = set(['ஆ', 'ஈ', 'ஊ', 'ஏ', 'ஐ', 'ஓ', 'ஔ'])

    ner_rules = {
        ('Kuril',): "Ner",
        ('Kuril', 'Mei'): "Ner",
        ('Nedil',): "Ner",
        ('Nedil', 'Mei'): "Ner"
    }

    nirai_rules = {
        ('Kuril', 'Kuril'): "Nirai",
        ('Kuril', 'Kuril', 'Mei'): "Nirai",
        ('Kuril', 'Nedil'): "Nirai",
        ('Kuril', 'Nedil', 'Mei'): "Nirai"
    }

    eer_asai_cheer = {
        ('Ner', 'Ner'): "தேமா",
        ('Nirai', 'Ner'): "புளிமா",
        ('Nirai', 'Nirai'): "கருவிளம்",
        ('Ner', 'Nirai'): "கூவிளம்"
    }

    moov_asai_cheer = {
        ('Ner', 'Ner', 'Ner'): "தேமாங்காய்",
        ('Nirai', 'Ner', 'Ner'): "புளிமாங்காய்",
        ('Nirai', 'Nirai', 'Ner'): "கருவிளங்காய்",
        ('Ner', 'Nirai', 'Ner'): "கூவிளங்காய்"
    }

    last_word_rules = {
        ('Ner',): "நாள்",
        ('Nirai',): "மலர்",
        ('Ner', 'Ner'): "காசு",
        ('Nirai', 'Ner'): "பிறப்பு"
    }

    matched_rules = []
    classified_letters = []
    
    for idx, word in enumerate(kural_words):
        letters = get_letters(word)
        classification = []

        for letter in letters:
            split_result = splitMeiUyir(letter)
            if isinstance(split_result, tuple) and len(split_result) == 2:
                mei, uyir = split_result
                if uyir:
                    if uyir in kuril:
                        classification.append('Kuril')
                    elif uyir in nedil:
                        classification.append('Nedil')
                    else:
                        classification.append('Mei')
            else:
                if letter in kuril:
                    classification.append('Kuril')
                elif letter in nedil:
                    classification.append('Nedil')
                else:
                    classification.append('Mei')

        classified_letters.append(classification) 

        i = 0
        word_rules = []
        while i < len(classification):
            matched = False
            for size in [3, 2, 1]:
                chunk = tuple(classification[i:i + size])
                if chunk in nirai_rules:
                    word_rules.append(nirai_rules[chunk])
                    i += size
                    matched = True
                    break
                elif chunk in ner_rules:
                    word_rules.append(ner_rules[chunk])
                    i += size
                    matched = True
                    break
            if not matched:
                i += 1

        matched_rules.append(word_rules)

    print("\nAsai & seer Analysis")
    for i, word in enumerate(kural_words):
        print(f"\nWord {i + 1}: {word}")
        print(f"Letters: {classified_letters[i]}") 
        print(f"Asai Matches: {matched_rules[i]}")

        pattern = "Pattern Not Found"
        rules = matched_rules[i]

        if i == len(kural_words) - 1:
            pattern = last_word_rules.get(tuple(rules), pattern)
        else:
            if len(rules) == 2:
                pattern = eer_asai_cheer.get(tuple(rules), pattern)
            elif len(rules) == 3:
                pattern = moov_asai_cheer.get(tuple(rules), pattern)

        print(f"Identified Vaippaadu: {pattern}")
        print("-----------------------------------------------------------------")
