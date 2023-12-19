import random
import pandas as pd
import spacy
import argparse

nlp = spacy.load('en_core_web_trf')

def get_article(word):
    if word[0] in "aeiou":
        return "an"
    else:
        return "a"


def generate_phrase(items, modifiers, colors, item_type):
    # randomly choose num modifiers 0,1,2
    if item_type == 'fruit':
        num_modifiers = random.choice([1])
    else:
        num_modifiers = random.choice([1])
    # randomly choose num_modifiers animal_modifiers
    if item_type == 'human':
        modifier_type = random.sample(modifiers.keys(), num_modifiers)
        sampled_modifiers = [ random.choice(modifiers[key]) for key in modifier_type ] 
    else:
        sampled_modifiers = random.sample(modifiers, num_modifiers)
    # randomly choose if to add color (30% chance)
    add_color = random.choice([True, False, False])
    if add_color and item_type != 'human':
        color = random.choice(colors)
        sampled_modifiers = [color]
    # if no modifiers, try again.
    if len(sampled_modifiers) == 0:
        return generate_phrase(items, modifiers, colors, item_type)

    article = get_article(sampled_modifiers[0])
    final_modifiers = " ".join(sampled_modifiers)
    item = random.choice(items)
    return f"{article} {final_modifiers} {item}", len(sampled_modifiers), item


def generate_prompt(num_phrases):
    animals = [
        "cat", "dog", "bird", "bear", "lion", "horse", "elephant", "monkey", "frog",
        "turtle", "rabbit", "mouse", "panda", "zebra", "gorilla", "penguin"
    ]    
    objects = [
        "backpack", "crown", "suitcase", "chair", "balloon", "bow",
        "car", "bowl", "bench", "clock", "camera", "umbrella", "guitar", "shoe", "hat",
        "surfboard", "skateboard", "bicycle"
    ]
    fruit = ["apple", "tomato", "banana", "strawberry"]
    human = ["person", "man", "woman", "athlete", "programmer", "artist"]


    colors = [
        "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray", "black",
        "white", "beige", "teal"
    ]

    animal_modifiers = ["furry", "baby", "spotted", "sleepy"]
    object_modifiers = ["modern", "spotted", "wooden", "metal", "curved", "spiky", "checkered"]
    fruit_modifiers = ["sliced", "skewered"]
    human_modifiers = {"age": ['young', 'old', 'middle-aged', 'adolescent'],  
                       "emotion": ['happy', 'sad', 'angry', 'surprised', 'sleepy'],  
                       "status": ['lying', 'sitting', 'standing'],                   }


    phrases = []
    item_indices = []
    item_idx = 0
    num_modifiers = 0
    while len(phrases) < num_phrases:
        choice = random.choice(range(4))
        # randomly choose between animal, object, and fruit
        if choice == 0:
            phrase, num_phrase_modifiers, item = generate_phrase(animals, animal_modifiers, colors, 'animal')
        elif choice == 1:
            phrase, num_phrase_modifiers, item = generate_phrase(fruit, fruit_modifiers, colors, 'fruit')
        elif choice == 2:
            phrase, num_phrase_modifiers, item = generate_phrase(objects, object_modifiers, colors, 'object')
        else:
            phrase, num_phrase_modifiers, item = generate_phrase(human, human_modifiers, colors, 'human')

        if phrase not in phrases:
            num_modifiers += num_phrase_modifiers
            phrases.append(phrase)
            item_idx += len(phrase.split())
            item_indices.append(item_idx)
            
    prompt = " and ".join(phrases)
    return prompt, num_modifiers, item_indices



def extract_attribution_indices(prompt, parser):
    doc = parser(prompt)
    subtrees = []
    modifiers = ['amod', 'nmod', 'compound', 'npadvmod', 'advmod', 'acomp']

    for w in doc:
        if w.pos_ not in ['NOUN', 'PROPN'] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == 'conj':
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def segment_text(text: str):
    segments = []
    doc = nlp(text)
    subtrees = extract_attribution_indices(doc, nlp)
    if subtrees:
        for subtree in subtrees:
            segments.append(" ".join([t.text for t in subtree]))
    return segments


def dvmp_dataset_creation(num_samples, dest_path, num_phrases=3):
    prompts = []
    prompts_num_modifiers_prompt = []
    prompts_num_phrases_per_prompt = []
    prompts_item_indices = []
    while len(prompts) < num_samples:
        prompt, num_modifiers, item_indices = generate_prompt(num_phrases)
        if prompt not in prompts:
            prompts_num_phrases_per_prompt.append(num_phrases)
            prompts_num_modifiers_prompt.append(num_modifiers)
            prompts_item_indices.append(item_indices)

            segments = segment_text(prompt)
            num_mods = sum([len(s.split(" ")[:-1]) for s in segments])
            if num_mods != num_modifiers:
                print(prompt, num_mods, num_modifiers)
            prompts.append(prompt)

    subjects = []

    # add subjects to each prompt
    docs = nlp.pipe(prompts)
    for doc in docs:
        subjects.append([token.text for token in doc if token.pos_ == 'NOUN'])

    # convert to df and save
    df = pd.DataFrame({'prompt': prompts,
                       'num_modifiers': prompts_num_modifiers_prompt,
                       'num_phrases': prompts_num_phrases_per_prompt,
                       'item_indices': prompts_item_indices,
                       'subjects': subjects})
    if dest_path:
        df.to_csv(dest_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a DVMP dataset.')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to generate.')
    parser.add_argument('--num_phrases', type=int, default=3, help='Number of phrases.')
    parser.add_argument('--dest_path', type=str, default='destination.csv', help='Destination CSV file path.')
    args = parser.parse_args()

    dvmp_dataset_creation(args.num_samples, args.dest_path, args.num_phrases)