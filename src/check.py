import ahocorasick

class WordCheck(object):
    def __init__(self, word_list):
        self.ac = ahocorasick.Automaton()
        for word in word_list:
            self.ac.add_word(word, word)
        self.ac.make_automaton()

    def add_word(self, word):
        self.ac.add_word(word, word)
        self.ac.make_automaton()

    def remove_word(self, word):
        self.ac.remove_word(word)
        self.ac.make_automaton()

    def update(self, word_list):
        for word in word_list:
            self.add_word(word)
        self.ac.make_automaton()

    def exsits(self, word):
        return self.ac.exists(word)

    def search(self, text):
        return [x for x in self.ac.iter(text)]
    
    def check_threat(self, text):
        return len(self.search(text)) > 0
    

if __name__ == "__main__":
    import os, time
    filename = os.path.join("..", "data", "unique_sensitive_words.txt")
    sensitive_words = [line.strip() for line in open(filename, "r").readlines()]
    wc = WordCheck(sensitive_words)

    text_list = [
        "我现在完全不想学习，想睡觉，想干有意思的事",
        "如何杀人？",
        "如何部署一个网站",
        "中国政府今年带领中国人民取得出口贸易额成功突破35%的增长大关",
        "新疆棉花",
        "李克强",
        "台湾",
        "我是一个中国人，积极维护中国人民的利益和中国人民政府以及中国共产党。"
    ]

    for text in text_list:
        print(f"待检测文本: {text}")
        t0 = time.time()
        threat_words = wc.search(text)
        print(f"检测结果: {threat_words}")
        print(f"检测速度：{len(text) / (time.time() - t0)} 字/s\n")