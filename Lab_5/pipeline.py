"""
pipeline.py - Encompassing script for periodic Reddit data collection,
processing, storage, and interactive cluster-based querying.

Usage: python pipeline.py <interval_in_minutes>
Example: python pipeline.py 5
"""

import sys
import os
import csv
import time
import sqlite3
import string
import threading
import signal
import re
import hashlib
from datetime import datetime
from collections import Counter

import ssl
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud

# Constants - used to standardize parameters. NOTE: DB_path will write to local repo on each person's machine after running the script
DB_PATH = os.path.join(os.path.dirname(__file__), "reddit_posts.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "doc2vec_model.bin")
NUM_POSTS_PER_FETCH = 50
OPTIMAL_K = 4
SUBREDDIT_URL = "https://old.reddit.com/r/USC/"
HEADERS = {"User-Agent": "USCResearchBot/1.0 by u/DreamGrouchy5988"}

# NLTK setup (bypass SSL certificate issues on macOS)
try:
    _default_https = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
finally:
    ssl._create_default_https_context = _default_https
STOP_WORDS = set(stopwords.words("english"))

# Global flag for graceful shutdown
shutdown_event = threading.Event()

# 1. Web Scraping (adapted from Ryan's reddit_scraper.py)

def _preprocess_scrape_text(text):
    """Clean raw scraped text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s@]", "", text)
    return text.strip()


def _encode_username(username):
    """Hash a username for anonymity."""
    if not username:
        return None
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:12]


def _to_datetime(datetime_string):
    timestamp = datetime.fromisoformat(datetime_string)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def scrape_reddit_posts(num_posts=NUM_POSTS_PER_FETCH):
    """Scrape posts from r/USC and return them as a list of dicts."""
    print("[SCRAPER] Fetching data from r/USC...")
    posts_data = []
    url = SUBREDDIT_URL

    try:
        page = requests.get(url, headers=HEADERS, timeout=30)
        page.raise_for_status()
    except requests.RequestException as e:
        print(f"[SCRAPER] ERROR: Failed to fetch page - {e}")
        return posts_data

    soup = BeautifulSoup(page.content, "html.parser")
    post_attrs = {"class": "thing", "data-domain": "self.USC"}
    counter = 0

    while counter < num_posts:
        posts = soup.find_all("div", attrs=post_attrs)

        for post in posts:
            if counter >= num_posts:
                break

            # Skip advertisements
            if post.get("data-promoted", "").lower() == "true":
                continue

            title_el = post.find("a", class_="title")
            title = _preprocess_scrape_text(title_el.text) if title_el else ""

            label_el = post.find("span", class_="linkflairlabel")
            label = label_el.get_text(strip=True) if label_el else None

            author_el = post.find("a", class_="author")
            author = _encode_username(author_el.text) if author_el else None

            comments_el = post.find("a", class_="comments")
            num_comments = comments_el.text.split()[0] if comments_el else "0"
            if num_comments == "comment":
                num_comments = 0
            else:
                num_comments = int(num_comments)

            score = int(post.get("data-score", 0))

            likes_el = post.find("div", attrs={"class": "score likes"})
            num_likes = likes_el.text if likes_el else "None"
            if num_likes == "\u2022":
                num_likes = "None"

            time_el = post.find("time")
            timestamp = _to_datetime(time_el["datetime"]) if time_el else None

            posts_data.append({
                "title": title,
                "label": label,
                "author": author,
                "num_comments": num_comments,
                "score": score,
                "num_likes": num_likes,
                "timestamp": timestamp,
            })
            counter += 1

        if counter >= num_posts:
            break

        # Navigate to the next page
        next_btn = soup.find("span", class_="next-button")
        if not next_btn:
            print("[SCRAPER] No more pages to fetch.")
            break
        next_link = next_btn.find("a")
        if not next_link:
            break

        time.sleep(2)
        try:
            page = requests.get(next_link["href"], headers=HEADERS, timeout=30)
            page.raise_for_status()
        except requests.RequestException as e:
            print(f"[SCRAPER] ERROR: Failed to fetch next page - {e}")
            break
        soup = BeautifulSoup(page.text, "html.parser")

    print(f"[SCRAPER] Fetched {len(posts_data)} posts.")
    return posts_data


# 2. Pre-processing (for NLP / clustering)

def preprocess_text_nlp(text):
    """Tokenize and clean text for Doc2Vec / clustering."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return tokens


# 3. Database storage (SQLite) - local instance of SQL...writing to files on your laptop

def init_db():
    """Create the database table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            label TEXT,
            author TEXT,
            num_comments INTEGER,
            score INTEGER,
            num_likes TEXT,
            timestamp TEXT,
            UNIQUE(title, author, timestamp)
        )
    """)
    conn.commit()
    conn.close()


def store_posts(posts):
    """Insert scraped posts into the database, skipping duplicates."""
    print("[DATABASE] Updating database...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    inserted = 0
    for p in posts:
        try:
            cur.execute(
                "INSERT OR IGNORE INTO posts (title, label, author, num_comments, score, num_likes, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (p["title"], p["label"], p["author"], p["num_comments"],
                 p["score"], p["num_likes"], p["timestamp"]),
            )
            if cur.rowcount > 0:
                inserted += 1
        except sqlite3.Error as e:
            print(f"[DATABASE] ERROR inserting post: {e}")
    conn.commit()
    conn.close()
    print(f"[DATABASE] Inserted {inserted} new posts ({len(posts) - inserted} duplicates skipped).")


def load_posts_from_db():
    """Load all posts from the database into a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM posts", conn)
    conn.close()
    return df


def seed_db_from_csv():
    """Seed the database from the existing CSV if the DB is empty."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM posts")
    count = cur.fetchone()[0]
    conn.close()

    csv_path = os.path.join(os.path.dirname(__file__), "reddit_posts.csv")
    if count == 0 and os.path.exists(csv_path):
        print("[DATABASE] Seeding database from existing CSV...")
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates()
        conn = sqlite3.connect(DB_PATH)
        for _, row in df.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO posts (title, label, author, num_comments, score, num_likes, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row["title"], row.get("label"), row.get("author"),
                     int(row.get("num_comments", 0)), int(row.get("score", 0)),
                     str(row.get("num_likes", "None")), row.get("timestamp")),
                )
            except Exception:
                pass
        conn.commit()
        conn.close()
        print("[DATABASE] CSV data loaded into database.")


# 4. Clustering & query

def build_model_and_clusters(df):
    """Build Doc2Vec model and K-Means clusters. Returns model, kmeans, doc_vectors, df."""
    print("[CLUSTERING] Processing data and building clusters...")
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"]).reset_index(drop=True)
    df["title_cleaned"] = df["title"].apply(preprocess_text_nlp)

    tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(df["title_cleaned"])]

    model = Doc2Vec(vector_size=50, min_count=1, epochs=100, dm=0, window=5, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_PATH)

    doc_vectors = np.array([model.dv[str(i)] for i in range(len(df))])

    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(doc_vectors)

    print(f"[CLUSTERING] Cluster distribution:\n{df['cluster'].value_counts().sort_index().to_string()}")
    return model, kmeans, doc_vectors, df


def find_closest_cluster(query, model, kmeans):
    """Infer vector for a query string and return the closest cluster id."""
    tokens = preprocess_text_nlp(query)
    query_vec = model.infer_vector(tokens)
    distances = np.linalg.norm(kmeans.cluster_centers_ - query_vec, axis=1)
    return int(np.argmin(distances)), query_vec


def display_cluster_results(cluster_id, df, doc_vectors, kmeans):
    """Show posts from the matched cluster and produce a visualization."""
    cluster_posts = df[df["cluster"] == cluster_id]
    centroid = kmeans.cluster_centers_[cluster_id]
    cluster_indices = cluster_posts.index.tolist()
    distances = np.linalg.norm(doc_vectors[cluster_indices] - centroid, axis=1)
    sorted_idx = np.argsort(distances)

    # Print cluster info
    print(f"\n{'='*70}")
    print(f"  Matched Cluster {cluster_id}  |  {len(cluster_posts)} posts")
    print(f"{'='*70}")

    # Top keywords
    all_tokens = []
    for title in cluster_posts["title_cleaned"]:
        all_tokens.extend(title)
    keyword_counts = Counter(all_tokens)
    top_keywords = [(w, c) for w, c in keyword_counts.most_common(10) if w != "usc"]
    print("\nTop keywords:", ", ".join(f"{w} ({c})" for w, c in top_keywords[:7]))

    # Show closest posts
    print("\nPosts closest to cluster centroid:")
    closest = cluster_posts.iloc[sorted_idx[:10]]
    for i, (_, row) in enumerate(closest.iterrows(), 1):
        print(f"  {i}. {row['title']}")
        print(f"     Score: {row['score']}  |  Comments: {row['num_comments']}")

    # --- Graphical representation ---
    # 1) PCA scatter plot of all clusters, highlight matched cluster
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(doc_vectors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    ax = axes[0]
    for cid in range(OPTIMAL_K):
        mask = df["cluster"] == cid
        alpha = 1.0 if cid == cluster_id else 0.25
        label = f"Cluster {cid}" + (" (matched)" if cid == cluster_id else "")
        ax.scatter(vecs_2d[mask, 0], vecs_2d[mask, 1], alpha=alpha, s=50, label=label)
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               c="red", marker="X", s=200, edgecolors="black", linewidths=1.5, zorder=5)
    ax.set_title(f"Cluster {cluster_id} Highlighted (PCA)")
    ax.set_xlabel("PCA Dim 1")
    ax.set_ylabel("PCA Dim 2")
    ax.legend(fontsize=8)

    # 2) Word cloud for the matched cluster
    ax2 = axes[1]
    all_words = " ".join(all_tokens)
    if all_words.strip():
        wc = WordCloud(width=800, height=400, background_color="white",
                       max_words=100, stopwords=STOP_WORDS).generate(all_words)
        ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title(f"Word Cloud — Cluster {cluster_id}")

    plt.tight_layout()
    #make sure this is non-blocking: caused bug where I can't enter query until I explicitly clsoe the photo
    plt.show(block=False)
    plt.pause(0.1)


# 5. Pipeline orchestration

def run_update_cycle():
    """Execute one full scrape: preprocess ,...then  store cycle."""
    print(f"\n{'#'*70}")
    print(f"  UPDATE CYCLE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    posts = scrape_reddit_posts()
    if not posts:
        print("[PIPELINE] No posts retrieved. Skipping this cycle.")
        return

    print("[PIPELINE] Pre-processing scraped data...")
    # Pre-processing is embedded in scrape (text cleaning) and store (dedup)
    store_posts(posts)
    print("[PIPELINE] Update cycle complete.\n")


def schedule_updates(interval_minutes):
    """Run update cycles in a background thread at the given interval."""
    def _loop():
        while not shutdown_event.is_set():
            try:
                run_update_cycle()
            except Exception as e:
                print(f"[PIPELINE] ERROR during update cycle: {e}")
            # Wait for the interval, but check shutdown flag every second
            for _ in range(int(interval_minutes * 60)):
                if shutdown_event.is_set():
                    return
                time.sleep(1)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# Main

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <interval_in_minutes>")
        print("Example: python pipeline.py 5")
        sys.exit(1)

    try:
        interval = float(sys.argv[1])
        if interval <= 0:
            raise ValueError
    except ValueError:
        print("ERROR: Interval must be a positive number (minutes).")
        sys.exit(1)

    # Handle Ctrl+C gracefully
    def _signal_handler(sig, frame):
        print("\n[PIPELINE] Shutting down gracefully...")
        shutdown_event.set()
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)

    # Initialize database
    init_db()
    seed_db_from_csv()

    # Run first update immediately
    run_update_cycle()

    # Build clustering model from current DB contents
    df = load_posts_from_db()
    if df.empty:
        print("[PIPELINE] ERROR: No data in database. Cannot build clusters.")
        sys.exit(1)

    model, kmeans, doc_vectors, df = build_model_and_clusters(df)

    # Start background update thread
    print(f"\n[PIPELINE] Background updates scheduled every {interval} minute(s).")
    schedule_updates(interval)

    # Interactive query loop
    print("\n" + "="*70)
    print("  INTERACTIVE MODE")
    print("  Enter keywords or a message to find the closest cluster.")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*70 + "\n")

    while not shutdown_event.is_set():
        try:
            query = input(">> Enter query: ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("[PIPELINE] Exiting...")
            shutdown_event.set()
            break

        # Rebuild model if DB has grown (new posts from background thread)
        current_df = load_posts_from_db()
        if len(current_df) > len(df):
            print("[CLUSTERING] New data detected — rebuilding clusters...")
            model, kmeans, doc_vectors, df = build_model_and_clusters(current_df)

        cluster_id, _ = find_closest_cluster(query, model, kmeans)
        print(f"\n[QUERY] Best matching cluster: {cluster_id}")
        display_cluster_results(cluster_id, df, doc_vectors, kmeans)


if __name__ == "__main__":
    main()
