import pandas as pd
import string
import re
import numpy as np
from underthesea import word_tokenize, text_normalize
import unicodedata

# Biểu thức chính quy để nhận diện và giữ lại các emoji
emoji_pattern = re.compile(
    u"["
    u"\U0001F600-\U0001F64F"  # Mặt cười, biểu cảm
    u"\U0001F44D"  # Like (👍)
    u"\U0001F44E"  # Dislike (👎)
    u"\U0001F4AA"  # Biểu tượng cơ bắp (💪)
    u"\U0001F91D"  # Biểu tượng bắt tay (🤝)
    u"\U0001F44C"  # Biểu tượng tay vẫy (👌)
    u"\U00002705"  # Biểu tượng tick (✅)
    u"\U0000274C"  # Biểu tượng cross (❌)
    u"\U00002764"
    u"]+", flags=re.UNICODE)

# Hàm để loại bỏ các từ không phải tiếng Anh, tiếng Việt (bao gồm tiếng Trung) nhưng giữ lại emoji
# Hàm để giữ lại emoji đúng vị trí
emoji_pattern_2 = re.compile(
    r"(✅|❌|👌|👍|😀|😁|😂|😃|😄|😅|😆|😇|😉|😊|😋|😎|😌|😍|😗|😘|😙|😚|😛|😜|😝|😞|😟|😠|😡|😢|😣|😤|😥|😦|😩|😫|😬|😭|😮|😱|😲|😳|😴|😵|😶|😸|😹|😻|😼|🙁|🙂|🙃|🙄|🙅|🙆|🙈|🤝)(\1+)",
    re.UNICODE)

emoji_pattern_3 = re.compile(
    u"["
    u"\U0001F600-\U0001F64F"  # Mặt cười, biểu cảm
    u"\U0001F300-\U0001F5FF"  # Biểu tượng thiên văn, đồ vật
    u"\U0001F680-\U0001F6FF"  # Phương tiện giao thông
    u"\U0001F1E0-\U0001F1FF"  # Quốc kỳ
    u"\U00002702-\U000027B0"  # Các biểu tượng khác
    u"\U000024C2-\U0001F251"  # Các biểu tượng khác
    u"\U0001f926-\U0001f937"  # Các biểu tượng người
    u'\U00010000-\U0010ffff'  # Các emoji bổ sung
    u"\u200d"  # Dấu nối cho emoji phức hợp
    u"\u2640-\u2642"  # Biểu tượng giới tính
    u"\u2600-\u2B55"  # Biểu tượng mặt trời, sao, giao thông
    u"\u23cf"  # Dừng
    u"\u23e9"  # Biểu tượng phát video
    u"\u231a"  # Đồng hồ
    u"\u3030"  # Biểu tượng nốt nhạc
    u"\ufe0f"  # Biểu tượng cảm xúc với dạng hình ảnh
    u"\U0001F44D"  # Like (👍)
    u"\U0001F44E"  # Dislike (👎)
    u"\U0001F4AA"  # Biểu tượng cơ bắp (💪)
    u"\U0001F91D"  # Biểu tượng bắt tay (🤝)
    u"\U0001F44C"  # Biểu tượng tay vẫy (👌)
    u"\U00002705"  # Biểu tượng tick (✅)
    u"\U0000274C"  # Biểu tượng cross (❌)
    u"]+", flags=re.UNICODE)


# Hàm làm sạch văn bản và loại bỏ các emoji lặp lại
def clean_text_and_remove_duplicates(text):
    # Sử dụng regex để thay thế các emoji lặp lại nhiều lần thành chỉ một lần
    cleaned_text = re.sub(emoji_pattern_2, r'\1', text)
    return cleaned_text


def clean_text_and_preserve_emojis(text):
    text = clean_text_and_remove_duplicates(text)
    # Tìm các emoji và lưu vị trí của chúng
    emoji_positions = [(match.start(), match.group()) for match in emoji_pattern.finditer(text)]

    # Loại bỏ emoji khỏi văn bản
    text_without_emoji = emoji_pattern.sub('', text)

    # Làm sạch văn bản không có emoji
    cleaned_text = re.sub(r'[^a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵĐđ\s]+', ' ',
                          text_without_emoji)

    # Chèn lại emoji vào vị trí ban đầu
    result = list(cleaned_text)
    for pos, emoji in emoji_positions:
        result.insert(pos, emoji)

    return ''.join(result)


abb_dict = pd.read_excel('trien_khai_model/preprocessing/normal.xlsx').set_index('abbreviation')['meaning'].to_dict()
character2emoji = pd.read_excel('trien_khai_model/preprocessing/character2emoji.xlsx').set_index('Character')['Emoji'].to_dict()
emoji2word = pd.read_excel('trien_khai_model/preprocessing/emoji2word.xlsx').set_index('Emoji')['Meaning'].to_dict()

with open("trien_khai_model/preprocessing/vn_stopwords.txt", encoding='utf-8') as file:
    stopwords = [
        re.escape(unicodedata.normalize('NFC', line.strip().lower()))
        for line in file.read().splitlines() if line.strip()
    ]


def remove_meaningless_words(text, delete_meaningless_words):
    words = text.split()

    cleaned_words = [word for word in words if word.lower() not in delete_meaningless_words]

    return " ".join(cleaned_words)


def abbreviation_normal(text, abb_dict):
    words = text.lower().split()
    return ' '.join(abb_dict.get(word, word) for word in words).strip()


def convert_character2emoji(text, character2emoji):
    for char, emoji in character2emoji.items():
        text = text.replace(char, f" {emoji} ")
    return text.strip()


def convert_emoji2word(text, emoji2word):
    for emoji, word in emoji2word.items():
        text = text.replace(emoji, f" {word} ")
    return text.strip()


stopwords_pattern = r'\b(?:' + '|'.join(stopwords) + r')\b'


def remove_stopword(text):
    # Loại bỏ stopwords bằng regex
    return re.sub(stopwords_pattern, '', text, flags=re.IGNORECASE).strip()


with open("trien_khai_model/preprocessing/merged_file_no_duplicates.txt", encoding='utf-8') as file:
    delete_meaningless_words = [unicodedata.normalize('NFC', line.lower()) for line in file.read().splitlines()]

dict_map = {
    "òa": "oà",
    "óa": "oá",
    "ỏa": "oả",
    "õa": "oã",
    "ọa": "oạ",
    "òe": "oè",
    "óe": "oé",
    "ỏe": "oẻ",
    "õe": "oẽ",
    "ọe": "oẹ",
    "ùy": "uỳ",
    "úy": "uý",
    "ủy": "uỷ",
    "ũy": "uỹ",
    "ụy": "uỵ",
}


def replace_all(text):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


def clean_text(text, remove_emoji=False, convert_e2w=False, normalize=False, tokenize=False, remove_stopwords=False):
    if re.search(r'https?://\S+|www\.\S+', text):
        return None

    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = clean_text_and_preserve_emojis(text)

    # Xóa ký tự xuống dòng
    text = re.sub(r'\r\n', '', text)

    # Loại bỏ các ký tự 🙌 và 🙏
    text = re.sub(r'[🙌]+', '', text)
    text = re.sub(r'[🙏]+', '', text)
    text = re.sub(r'\r\n', '', text)

    # Loại bỏ các từ vô nghĩa
    text = remove_meaningless_words(text, delete_meaningless_words)

    # Thay thế emoji
    text = re.sub(emoji_pattern_2, r'\1', text)
    text = abbreviation_normal(text, abb_dict)
    text = replace_all(text)

    # Loại bỏ emoji nếu cần
    if remove_emoji:
        text = re.sub(emoji_pattern, " ", text)

    # Xử lý các từ lặp lại
    text = re.sub(r'([a-z]+?)\1+', r'\1', text)

    text = re.sub(r'["\']', '', text)

    # Xử lý dấu câu
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+", r"\1", text)

    # Loại bỏ các khoảng trắng thừa
    text = text.strip().strip(string.punctuation)
    text = re.sub(r"\s+", " ", text)

    # Chuyển emoji thành từ nếu yêu cầu
    if convert_e2w:
        text = convert_character2emoji(text, character2emoji)
        text = convert_emoji2word(text, emoji2word)

    if normalize:
        text = text_normalize(text)

    # Token hóa văn bản nếu yêu cầu
    if tokenize:
        text = word_tokenize(text, format="text")
    # Loại bỏ stopwords nếu yêu cầu
    if remove_stopwords:
        text = remove_stopword(text)
    return text


def check_emoji(text):
    # Tìm tất cả emoji trong văn bản
    emojis = emoji_pattern.findall(' '.join(text))

    return emojis, np.unique(emojis)