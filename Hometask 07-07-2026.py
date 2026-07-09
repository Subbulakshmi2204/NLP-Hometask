from collections import defaultdict
import math

# -----------------------------
# CONFIGURATION
# -----------------------------
N = 5          # Highest order n-gram
DISCOUNT = 0.75

# -----------------------------
# CLASS
# -----------------------------
class KneserNeyLanguageModel:

    def __init__(self):

        # n-gram counts
        self.ngram_counts = [defaultdict(int) for _ in range(N)]

        # history counts
        self.history_counts = [defaultdict(int) for _ in range(N)]

        # vocabulary
        self.vocab = set()

        # continuation statistics
        self.continuation_count = defaultdict(set)

        # unique followers
        self.unique_followers = defaultdict(set)

        # unique predecessors
        self.unique_predecessors = defaultdict(set)

    # -----------------------------
    # Tokenize sentence
    # -----------------------------
    def tokenize(self, sentence):

        words = sentence.lower().strip().split()

        return ["<s>"]*(N-1) + words + ["</s>"]

    # -----------------------------
    # Train Model
    # -----------------------------
    def train(self, corpus):

        for sentence in corpus:

            tokens = self.tokenize(sentence)

            self.vocab.update(tokens)

            length = len(tokens)

            # Build 1-gram to 5-gram
            for n in range(1, N+1):

                for i in range(length-n+1):

                    gram = tuple(tokens[i:i+n])

                    self.ngram_counts[n-1][gram] += 1

                    if n > 1:

                        history = gram[:-1]

                        word = gram[-1]

                        self.history_counts[n-1][history] += 1

                        self.unique_followers[history].add(word)

                        self.unique_predecessors[word].add(history)

                        if n == 2:
                            self.continuation_count[word].add(history)

        print("Training completed.")
        print("Vocabulary Size :", len(self.vocab))
        print()

    # -----------------------------
    # Display Statistics
    # -----------------------------
    def show_statistics(self):

        print("Model Statistics")
        print("-------------------------")

        for i in range(N):

            print(f"{i+1}-gram count :",
                  len(self.ngram_counts[i]))

        print("-------------------------")
        # -----------------------------
    # Continuation Probability
    # -----------------------------
    def continuation_probability(self, word):

        numerator = len(self.continuation_count[word])

        denominator = len(self.ngram_counts[1])

        if denominator == 0:
            return 1 / len(self.vocab)

        return numerator / denominator

    # -----------------------------
    # Recursive Probability
    # -----------------------------
    def probability(self, history, word):

        order = len(history) + 1

        # Base case (unigram)
        if order == 1:

            return self.continuation_probability(word)

        gram = tuple(history) + (word,)

        gram_count = self.ngram_counts[order-1].get(gram, 0)

        history_count = self.history_counts[order-1].get(tuple(history), 0)

        # Backoff
        if history_count == 0:

            return self.probability(history[1:], word)

        discounted = max(gram_count - DISCOUNT, 0) / history_count

        lambda_weight = (
            DISCOUNT *
            len(self.unique_followers[tuple(history)])
            / history_count
        )

        backoff = self.probability(history[1:], word)

        return discounted + lambda_weight * backoff

    # -----------------------------
    # Score sentence
    # -----------------------------
    def sentence_probability(self, sentence):

        tokens = self.tokenize(sentence)

        probability = 1

        for i in range(N-1, len(tokens)):

            history = tokens[i-(N-1):i]

            word = tokens[i]

            probability *= self.probability(history, word)

        return probability

        # -----------------------------
    # Predict Next Words
    # -----------------------------
    def predict(self, query, top_k=5):

        tokens = self.tokenize(query)

        history = tokens[-(N-1):]

        scores = []

        for word in self.vocab:

            if word in ["<s>", "</s>"]:
                continue

            score = self.probability(history, word)

            scores.append((word, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    corpus = [

        "best places to visit in india",

        "best places to visit in chennai",

        "best places to visit during summer",

        "best places to eat in chennai",

        "best hotels in chennai",

        "places to visit in india",

        "places to visit near me",

        "places to visit during winter",

        "visit temples in tamil nadu",

        "best tourist places in india",

        "best tourist places in chennai",

        "best places to visit in kerala"

    ]

    model = KneserNeyLanguageModel()

    model.train(corpus)

    model.show_statistics()

    while True:

        query = input("\nEnter Search Query (or exit): ")

        if query.lower() == "exit":
            break

        predictions = model.predict(query)

        print("\nTop Suggestions\n")

        for word, score in predictions:

            print(f"{word:15} {score:.6f}")
