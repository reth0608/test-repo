from gensim.models import KeyedVectors

# Load pretrained embeddings (example: Word2Vec)
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors.bin", binary=True)

def enrich_prompt(prompt):
    words = prompt.split()
    enriched_words = []

    for word in words:
        try:
            similar = model.most_similar(word, topn=3)
            enriched_words.extend([w for w, _ in similar])
        except:
            enriched_words.append(word)

    return prompt + " " + " ".join(enriched_words)

prompt = "Explain good coding practices"
enriched = enrich_prompt(prompt)

print("Original:", prompt)
print("Enriched:", enriched)