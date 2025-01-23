# CODE augmetation: https://github.com/jasonwei20/eda_nlp
import random
import json
import re

with open("text_augmentation/filtered_word_net_vi.json", "r") as f:
    wordnet_vi = json.load(f)

def get_synonyms(word):
    """Lấy các từ đồng nghĩa từ wordnet đã nạp sẵn."""
    return set(wordnet_vi.get(word, []))

def synonym_replacement(text, p=0.5, n=2, max_attempts=10):
    """Thay thế từ trong văn bản bằng các từ đồng nghĩa với xác suất p."""
    num_replaced = 0
    for phrase in wordnet_vi.keys():
        if phrase in text:
            synonyms = get_synonyms(phrase)
            if synonyms:
                if random.random() < p:
                    synonym = random.choice(list(synonyms))
                    text = text.replace(phrase, synonym)
                num_replaced += 1
                if num_replaced >= n:
                    break
    return text

import random

def random_deletion(text, p=0.2, n=2, max_attempts=10):
    """Xóa từ với xác suất p trong văn bản."""
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [word for word in words if random.random() > p]
    # Nếu không có từ nào được giữ lại, chọn ngẫu nhiên một từ
    return ' '.join(new_words if new_words else [random.choice(words)])

def random_swap(text, p=0.5, n=2, max_attempts=10):
    """Hoán đổi từ ngẫu nhiên với xác suất p trong văn bản."""
    words = text.split()
    for _ in range(n):
        if random.random() < p and len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def random_insertion(text, p=0.5, n=2, max_attempts=10):
    """Thêm n từ vào câu với xác suất p, giới hạn số lần thử tìm từ đồng nghĩa."""
    words = text.split()
    for _ in range(n):
        if random.random() < p:
            synonyms = []
            attempts = 0
            while not synonyms and attempts < max_attempts:
                word = random.choice(words)
                synonyms = get_synonyms(word)
                attempts += 1
            
            if synonyms:
                words.insert(random.randint(0, len(words)), random.choice(list(synonyms)))

    return ' '.join(words)

def compose(*functions):
    def composed_function(text, n=2, p=0.5, max_attempts=10):
        for func in functions:
            text = func(text, n=n, p=p, max_attempts=max_attempts)
        return text
    return composed_function


































# def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
#     """Perform EDA on the input sentence."""
#     words = re.findall(r'\w+', sentence.lower())
#     num_words = len(words)

#     if num_words == 0:
#         return []

#     n_sr, n_ri, n_rs = map(lambda alpha: max(1, int(alpha * num_words)), [alpha_sr, alpha_ri, alpha_rs])
#     augmented_sentences = []

#     for _ in range(num_aug // 4 + 1):
#         augmented_sentences.extend([
#             ' '.join(synonym_replacement(words, n_sr)),
#             ' '.join(random_insertion(words, n_ri)),
#             ' '.join(random_swap(words.copy(), n_rs)),
#             ' '.join(random_deletion(words, p_rd)),
#         ])

#     augmented_sentences = list(set(augmented_sentences))[:num_aug]
#     augmented_sentences.append(' '.join(words))
#     return augmented_sentences


# def gen_eda(train_orig, output_file, alpha, num_aug=9):
#     """Generate augmented sentences using EDA."""
#     with open(train_orig, "r") as f:
#         lines = f.readlines()

#     augmented_data = []
#     for line in lines:
#         try:
#             sentence, label = line.strip().split('|')
#             augmented_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
#             augmented_data.extend(f"{aug},{label}" for aug in augmented_sentences)
#         except Exception as e:
#             print(f"Error processing line: {line}. Error: {e}")
#             continue

#     with open(output_file, "w") as f:
#         f.write("\n".join(augmented_data))
#     print(f"Generated augmented data at {output_file}.")


     