import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from collections import Counter
from doc2vec_embeddings import examine_clusters, preprocess_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "reddit_posts.csv"
VECTOR_SIZES = [50, 100, 300]   # must match Part 1 Doc2Vec sizes
W2V_DIM = 100                   # Word2Vec training dimension (free parameter)
N_DOC_CLUSTERS = 4              # match Part 1's KMeans cluster count

# ---------------------------------------------------------------------------
# Step 1 — Load and preprocess data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df = df.dropna(subset=["title"])
    df = df.reset_index(drop=True)

    text_col = "title"

    def preprocess(text):
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) >= 2]
        return tokens

    print("Step 1: Preprocessing posts...")
    tokenized_posts = [preprocess(t) for t in df[text_col]]
    print(f"  {len(tokenized_posts)} posts ready.")

    # ---------------------------------------------------------------------------
    # Step 2 — Train ONE Word2Vec model
    # ---------------------------------------------------------------------------
    print(f"\nStep 2: Training Word2Vec (vector_size={W2V_DIM})...")
    w2v_model = Word2Vec(
        sentences=tokenized_posts,
        vector_size=W2V_DIM,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
    )
    w2v_model.save("word2vec_model.model")
    print("  Saved word2vec_model.model")

    words = list(w2v_model.wv.index_to_key)
    word_vectors = np.array([w2v_model.wv[w] for w in words])
    print(f"  Vocabulary size: {len(words)}")

    # ---------------------------------------------------------------------------
    # Step 3–7 — For each K: cluster words, embed docs, cluster docs, evaluate, plot
    # ---------------------------------------------------------------------------
    def embed_post(tokens, word_to_bin, K):
        vec = np.zeros(K)
        known = [t for t in tokens if t in word_to_bin]
        if not known:
            return vec
        for t in known:
            vec[word_to_bin[t]] += 1
        return vec / len(known)

    results = []  # (size, silhouette)

    for K in VECTOR_SIZES:
        print(f"\n--- Vector size K={K} ---")

        # Step 3: cluster word vectors into K bins
        print(f"  Step 3: KMeans on word vectors (n_clusters={K})...")
        kmeans_words = KMeans(n_clusters=K, random_state=42, n_init="auto")
        kmeans_words.fit(word_vectors)
        word_to_bin = {word: int(cid) for word, cid in zip(words, kmeans_words.labels_)}

        # Step 4: embed each document as a normalized histogram
        print("  Step 4: Embedding documents as BoW histograms...")
        embeddings = np.array([embed_post(tokens, word_to_bin, K) for tokens in tokenized_posts])

        # Normalize for cosine similarity
        normalized = normalize(embeddings, norm="l2")

        # Step 5: cluster documents
        print(f"  Step 5: Clustering documents (n_clusters={N_DOC_CLUSTERS})...")
        kmeans_docs = KMeans(n_clusters=N_DOC_CLUSTERS, random_state=42, n_init="auto")
        labels = kmeans_docs.fit_predict(normalized)

        out_df = pd.DataFrame({"post_text": df[text_col].values, "cluster": labels})
        out_path = f"bow_clusters_{K}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

        # Step 5b: word cloud evaluation
        df['cluster'] = labels
        wc_path = f'bow_wordclouds_{K}.png'
        print("  Step 5b: Generating word cloud per cluster...")
        examine_clusters(df, normalized, kmeans_docs, N_DOC_CLUSTERS,
                         text_col='title', fig_title=f'BoW Word Clouds (K={K})', save_path=wc_path)
        print(f"  Saved {wc_path}")

        # Step 6: evaluate
        score = silhouette_score(normalized, labels, metric="cosine")
        print(f"  Silhouette score (cosine): {score:.4f}")
        results.append((K, score))

        # Step 7: visualize
        print("  Step 7: TSNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, metric="cosine")
        reduced = tsne.fit_transform(normalized)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
        plt.title(f"BoW Clusters (vector_size={K})")
        plt.colorbar(label="Cluster")
        plt.tight_layout()
        plot_path = f"bow_clusters_{K}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved {plot_path}")

    # ---------------------------------------------------------------------------
    # Step 6 (continued) — Load Doc2Vec models and compute their silhouette scores
    # ---------------------------------------------------------------------------
    doc2vec_model_paths = {
        50:  "doc2vec_small.bin",
        100: "doc2vec_medium.bin",
        300: "doc2vec_large.bin",
    }

    doc2vec_scores = {}
    print("\nLoading Doc2Vec models for silhouette comparison...")
    for K, model_path in doc2vec_model_paths.items():
        try:
            d2v = Doc2Vec.load(model_path)
            n_docs = len(d2v.dv)
            d2v_vectors = np.array([d2v.dv[str(i)] for i in range(n_docs)])
            d2v_norm = normalize(d2v_vectors, norm="l2")
            km = KMeans(n_clusters=N_DOC_CLUSTERS, random_state=42, n_init="auto")
            d2v_labels = km.fit_predict(d2v_norm)
            d2v_score = silhouette_score(d2v_norm, d2v_labels, metric="cosine")
            doc2vec_scores[K] = d2v_score
            print(f"  Doc2Vec size={K}: silhouette={d2v_score:.4f}")
        except FileNotFoundError:
            doc2vec_scores[K] = None
            print(f"  Doc2Vec size={K}: model file '{model_path}' not found — skipping.")

    # ---------------------------------------------------------------------------
    # Step 6 (continued) — Print and save comparison table
    # ---------------------------------------------------------------------------
    print("\n\nEvaluation Summary")
    print("=" * 55)
    header = f"{'Vector Size':>12} | {'BoW Silhouette':>16} | {'Doc2Vec Silhouette':>18}"
    print(header)
    print("-" * 55)
    lines = [header, "-" * 55]
    for K, bow_score in results:
        d2v_score = doc2vec_scores.get(K)
        d2v_str = f"{d2v_score:.4f}" if d2v_score is not None else "(model not found)"
        row = f"{K:>12} | {bow_score:>16.4f} | {d2v_str:>18}"
        print(row)
        lines.append(row)

    summary = "\n".join(lines) + "\n"
    with open("evaluation_summary.txt", "w") as f:
        f.write(summary)
    print("\nSaved evaluation_summary.txt")
