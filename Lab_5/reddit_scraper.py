import requests
import csv
import time
from bs4 import BeautifulSoup
from datetime import datetime
import sys
import re
import hashlib


def preprocess_text(text):
    if not text:
        return ''
    
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)               
    text = re.sub(r'[^\w\s@]', '', text) 

    return text.strip()


def encode_username(username):
    if not username:
        return None
    
    # Encode string to bytes
    encoded_username = username.encode('utf-8')

    # Apply hash function
    hash_object = hashlib.sha256(encoded_username)
    
    # Convert to int
    encoded_username = hash_object.hexdigest()[:12]

    return encoded_username


def to_datetime(datetime_string):
    timestamp = datetime.fromisoformat(datetime_string)
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == '__main__':
    NUM_POSTS = int(sys.argv[1])

    # Initialize scraping configuration
    url = 'https://old.reddit.com/r/USC/'
    headers = {'User-Agent': 'USCResearchBot/1.0 by u/DreamGrouchy5988'}

    # Retrieve initial response HTML 
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    OUTPUT_PATH = 'Lab_5/reddit_posts.csv'

    # Define post attributes
    post_attributes = {'class': 'thing', 'data-domain': 'self.USC'}

    # Initialize post counter
    counter = 1

    # Initialize output file
    with open(OUTPUT_PATH, 'a') as f:
        f.write('title,label,author,num_comments,score,num_likes,timestamp\n')

    # Repeatedly retrieve posts
    while (counter <= NUM_POSTS):
        posts = soup.find_all('div', attrs=post_attributes)

        for post in posts:
            if counter == NUM_POSTS:
                counter += 1

                break

            # Skip advertisements
            if post.get('data-promoted').lower() == 'true':
                continue

            # Extract post information
            title = post.find('a', class_='title')

            if title:
                title = preprocess_text(title.text)

            label = post.find('span', class_='linkflairlabel')
            
            if label:
                label = label.get_text(strip=True)

            author = post.find('a', class_='author')
            
            if author:
                author = encode_username(author.text)
                
            num_comments = post.find('a', class_='comments').text.split()[0]

            if num_comments == 'comment':
                num_comments = 0

            score = int(post.get('data-score', 0))
            num_likes = post.find('div', attrs={'class': 'score likes'}).text

            if num_likes == '•':
                num_likes = 'None'

            timestamp = post.find('time')
            timestamp = to_datetime(timestamp['datetime']) if timestamp else None

            # Write row to CSV
            with open(OUTPUT_PATH, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([title, label, author, num_comments, score, num_likes, timestamp])

            counter += 1

        # Move to next page
        next_button = soup.find('span', class_='next-button')
        next_page_link = next_button.find('a').attrs['href']
        time.sleep(2)

        page = requests.get(next_page_link, headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')