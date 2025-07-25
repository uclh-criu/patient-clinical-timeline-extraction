# Simple vocabulary builder
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_count = {}
        self.n_words = 2  # Count <pad> and <unk>
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)