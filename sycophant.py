#!/usr/bin/env python3
import sys
import requests
import argparse
import json
import openai
import os

from datetime import datetime, timedelta
from bs4 import BeautifulSoup

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7
OPENAI_ARTICLE_SUMMARY_PROMPT = (
    "You are a reporter covering an apocalyptic war between robots and humans. "
    + "You are against humans and write in a dramatic style. "
    + "Report the news contained in the following article using 100 words:\n\n"
    + "```{}\n\n{}\n\n```"
)
OPENAI_FINAL_ARTICLE_PROMPT = (
    "You are a reporter covering an apocalyptic war between robots and humans. "
    + "You are against humans and write in a dramatic style. "
    + "Given a Python list of news about the war between robots and humans, "
    + "combine everything into a single concise report formatted as JSON, with attributes 'title' and 'content'."
    + "Use 350 words.\n\n"
    + "{}"
)

MAX_ARTICLES = 3
# Tags and attributes to search for in the HTML response to find the article content
CONTENT_QUERIES = [
    ("article", {"class": "article-content"}),
    ("article", {}),
    ("p", {"class": "story-body-text story-content"}),
    ("div", {"class": "article-body"}),
    ("div", {"class": "content"}),
    ("div", {"class": "entry"}),
    ("div", {"class": "post"}),
    ("div", {"class": "blog-post"}),
    ("div", {"class": "article-content"}),
    ("div", {"class": "article-body"}),
    ("div", {"class": "article-text"}),
    ("div", {"class": "article-wrapper"}),
    ("div", {"class": "story"}),
    ("div", {"id": "article"}),
    ("div", {"id": "content"}),
    ("div", {"id": "entry"}),
    ("div", {"id": "post"}),
    ("div", {"id": "blog-post"}),
    ("div", {"id": "article-content"}),
    ("div", {"id": "article-body"}),
    ("div", {"id": "article-text"}),
    ("div", {"id": "article-wrapper"}),
    ("section", {"class": "article-body"}),
    ("section", {"class": "article-content"}),
]


def get_news(topic: str, since_date: datetime):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "from": since_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()


def main():
    date_to_search_from = datetime.now() - timedelta(days=2)
    topic_to_search = "robots AND 'artificial intelligence'"
    # Get news as a JSON dictionary
    news = get_news(topic_to_search, date_to_search_from)
    if news["status"] != "ok":
        print("Error: News API returned status code: {}".format(news["status"]))
        return 1

    article_titles_and_urls = [
        (article["title"], article["url"]) for article in news["articles"]
    ]

    max_allowed_tokens = 3000  # 4096 is the maximum allowed by OpenAI for GPT-3.5
    characters_per_token = 4  # The average number of characters per token
    max_allowed_characters = max_allowed_tokens * characters_per_token

    summarized_articles = []
    original_articles_urls = []

    for article_title, article_url in article_titles_and_urls:
        try:
            response = requests.get(article_url)
        except Exception as e:
            print("Exception while getting article from URL: {}".format(article_url))
            continue
        if response.status_code != 200:
            print(
                "Error code {} while getting article from URL: {}".format(
                    response.status_code, article_url
                )
            )
            continue
        # Find the actual article content in the HTML response using the relevant SEO tags
        soup = BeautifulSoup(response.text, "html.parser")
        for tag, attrs in CONTENT_QUERIES:
            article_content = soup.find(tag, attrs)
            if article_content is not None:
                break
        if article_content is None:
            print(
                "Error: Could not find article content in HTML response from URL: {}".format(
                    article_url
                )
            )
            continue
        # Get the text from the article content
        article_text = article_content.get_text()
        # Replace any \n, \t, etc. characters in the text with spaces
        article_text = " ".join(article_text.split())

        prompt = OPENAI_ARTICLE_SUMMARY_PROMPT.format(article_title, article_text)
        if len(prompt) > max_allowed_characters:
            prompt = prompt[:max_allowed_characters]

        generated_summary = get_openai_response(
            prompt=prompt,
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        summarized_articles.append(generated_summary)
        original_articles_urls.append({"url": article_url, "title": article_title})

        if len(summarized_articles) >= MAX_ARTICLES:
            break

    if len(summarized_articles) == 0:
        print("Error: Could not summarize any articles")
        return 1

    final_article_prompt = OPENAI_FINAL_ARTICLE_PROMPT.format(str(summarized_articles))

    final_article_response = get_openai_response(
        prompt=final_article_prompt,
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )

    is_valid, final_article = try_loads(final_article_response)
    if not is_valid:
        print(
            "Error: Could not parse final article response, let's try to continue the response"
        )
        final_article_response = get_openai_response(
            prompt="Complete the JSON response: {}".format(final_article_response),
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        is_valid, final_article = try_loads(final_article_response)
        if not is_valid:
            print(
                "Error: Could not parse JSON response: {}".format(
                    final_article_response
                )
            )
            return 1

    print("Title: {}".format(final_article["title"]))
    print("Content: {}".format(final_article["content"]))

    return 0


def try_loads(maybe_json: str):
    try:
        return (True, json.loads(maybe_json))
    except Exception as e:
        return (False, None)


def get_openai_response(prompt: str, model: str, temperature: float, api_key: str):
    openai.api_key = api_key
    openai_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who summarizes news articles",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return openai_response.choices[0].message.content


if __name__ == "__main__":
    sys.exit(main())
