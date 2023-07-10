#!/usr/bin/env python3
import sys
import requests
import argparse
import json
import openai
import os
import re

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4"
OPENAI_MAX_TOKENS = 4000
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
    + "combine everything into a single concise report formatted as JSON: "
    + '{"title": "...", "content": "..."}\n\n'
    + "Use 350 words:\n\n"
)

OPEN_AI_PROMPT_FOR_DALLE = (
    "4 examples of good prompts for LLMs that generate images:\n"
    + "- Full body photo of a horse in a space suit\n"
    + "- 3D render of a cute tropical fish in an aquarium on a dark blue background, digital art\n"
    + "- An expressive oil painting of a basketball player dunking, depicted as an explosion of a nebula\n"
    + "- A blue orange sliced in half laying on a blue floor in front of a blue wall\n\n"
    + "Now, write 1 prompt for an image that would go on a newspaper front page, "
    + "based on the following title which should not be included in the prompt: "
)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
TOPIC_TO_SEARCH = "robots AND 'artificial intelligence'"
MAX_ARTICLES = 3
MAX_ARTICLE_AGE = 2  # days

ASSET_PATH = ""
POST_DIR = ""
TEMPLATE_PATH = "post-template.jinja"
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


def get_news(topic: str, since_date: datetime, api_key: str):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "from": since_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        help="Template (Jinja2) to use for generating the article",
        default=TEMPLATE_PATH,
    )
    parser.add_argument(
        "--post-dir", help="Path to the post to generate", default=POST_DIR
    )
    parser.add_argument(
        "--openai-api-key", help="OpenAI API key", default=OPENAI_API_KEY
    )
    parser.add_argument(
        "--openai-model", help="OpenAI model to use", default=OPENAI_MODEL
    )
    parser.add_argument(
        "--openai-temperature",
        help="OpenAI temperature to use",
        default=OPENAI_TEMPERATURE,
    )
    parser.add_argument(
        "--openai-article-summary-prompt",
        help="OpenAI article summary prompt to use",
        default=OPENAI_ARTICLE_SUMMARY_PROMPT,
    )
    parser.add_argument(
        "--openai-final-article-prompt",
        help="OpenAI final article prompt to use",
        default=OPENAI_FINAL_ARTICLE_PROMPT,
    )
    parser.add_argument(
        "--openai-prompt-for-dalle",
        help="OpenAI GPT prompt to create a prompt for DALL-E",
        default=OPEN_AI_PROMPT_FOR_DALLE,
    )
    parser.add_argument(
        "--openai-max-tokens",
        help="OpenAI max tokens to use for parsing each article",
        default=OPENAI_MAX_TOKENS,
    )
    parser.add_argument("--news-api-key", help="News API key", default=NEWS_API_KEY)
    parser.add_argument(
        "--news-max-articles",
        help="Maximum number of articles to use from News API",
        default=MAX_ARTICLES,
    )
    parser.add_argument(
        "--news-topic", help="Topic to search for in News API", default=TOPIC_TO_SEARCH
    )
    parser.add_argument(
        "--news-max-article-age",
        help="Maximum age of articles to use (in days)",
        default=MAX_ARTICLE_AGE,
    )
    # args = parser.parse_args()

    date_to_search_from = datetime.now() - timedelta(days=MAX_ARTICLE_AGE)
    # Get news as a JSON dictionary through the News API (newsapi.org)
    print("Searching for primary sources on subject: {}".format(TOPIC_TO_SEARCH))
    news = get_news(
        topic=TOPIC_TO_SEARCH, since_date=date_to_search_from, api_key=NEWS_API_KEY
    )
    if news["status"] != "ok":
        print("Error: News API returned status code: {}".format(news["status"]))
        return 1

    article_titles_and_urls = [
        (article["title"], article["url"]) for article in news["articles"]
    ]
    print("Found {} articles".format(len(article_titles_and_urls)))

    max_allowed_tokens = OPENAI_MAX_TOKENS
    characters_per_token = 4  # The average number of characters per token
    max_allowed_characters = max_allowed_tokens * characters_per_token

    summarized_articles = []
    original_articles_urls = []  # Only the titles and the URLs of the articles we use

    print("Summarizing the top-{} articles...".format(MAX_ARTICLES))
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

    print("Generating the final article...")
    final_article_prompt = OPENAI_FINAL_ARTICLE_PROMPT + str(summarized_articles)
    final_article_response = get_openai_response(
        prompt=final_article_prompt,
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )

    final_article = try_loads(final_article_response)
    if not final_article:
        print(
            "Error: Could not parse final article response, let's try to continue the response"
        )
        final_article_response = get_openai_response(
            prompt="Complete the JSON response: {}".format(final_article_response),
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        final_article = try_loads(final_article_response)
        if not final_article:
            print(
                "Error: Could not parse JSON response: {}".format(
                    final_article_response
                )
            )
            return 1

    print("Generating tags for the final article...")
    generated_tags_response = get_openai_response(
        prompt="Generate 3 tags as a JSON list, use one word for each tag,"
        + 'e.g. ["tag1", "tag2", "tag3"], for the following article: \n'
        + final_article["title"],
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )
    generated_tags = try_loads(generated_tags_response)
    if not generated_tags:
        print(
            "Error: Could not parse generated tags response: {}".format(
                generated_tags_response
            )
        )
        print("Will ignore the tags and continue")
        generated_tags = ""
    # Split the tags into comma-separated values
    generated_tags = ", ".join(generated_tags)

    print("Generating the prompt for the image generation...")
    prompt_gpt_to_create_dalle_prompt = (
        OPEN_AI_PROMPT_FOR_DALLE + final_article["title"]
    )
    dalle_prompt = get_openai_response(
        prompt=prompt_gpt_to_create_dalle_prompt,
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )

    print("Generating an image based on the article...")
    dalle_response = openai.Image.create(
        prompt=dalle_prompt,
        n=1,
        size="1024x1024",
        response_format="url",
    )
    dalle_image_url = dalle_response["data"][0]["url"]

    print("Downloading the image...")
    response = requests.get(dalle_image_url)
    if response.status_code != 200:
        print(
            "Error code {} while getting image from URL: {}".format(
                response.status_code, dalle_image_url
            )
        )
        return 1
    image = Image.open(BytesIO(response.content))
    title_normalized = re.sub(r"[^\w\s]", "", final_article["title"])
    title_normalized = title_normalized.replace(" ", "_")
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image_file_name = Path("{}_{}.png".format(current_date, title_normalized))
    image_path = Path(ASSET_PATH) / image_file_name
    image.save(image_path)

    # Append the links to the original articles to the final article
    made_with_sycophant = (
        "\n\nWritten with the help of "
        + "[sycophant](https://github.com/platisd/sycophant) "
        + "based on content from the following articles:\n"
    )
    attribution_links = "\n".join(
        [
            "- [{}]({})".format(article["title"], article["url"])
            for article in original_articles_urls
        ]
    )

    post_title = "'" + final_article["title"] + "'"
    post_date = current_date
    post_tags = generated_tags
    img_path = ASSET_PATH
    post_image = image_file_name
    image_caption = "'" + dalle_prompt + "'"
    post_content = final_article["content"] + made_with_sycophant + attribution_links
    post_filename = image_file_name.with_suffix(".md")

    print("Generating the final post...")
    environment = Environment(loader=FileSystemLoader(Path(TEMPLATE_PATH).parent))
    template = environment.get_template(Path(TEMPLATE_PATH).name)
    output = template.render(
        post_title=post_title,
        post_date=post_date,
        post_tags=post_tags,
        img_path=img_path,
        post_image=post_image,
        image_caption=image_caption,
        post_content=post_content,
    )

    post_path = Path(POST_DIR) / post_filename
    with open(post_path, "w") as f:
        f.write(output)

    print(
        "Done! Generated files: \n"
        + " - [Post]({})\n".format(post_path)
        + " - [Image]({})\n".format(image_path)
    )

    return 0


def try_loads(maybe_json: str):
    try:
        return json.loads(maybe_json, strict=False)
    except Exception as e:
        print(e)
        print("Response not a valid JSON: \n" + maybe_json)
        return None


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
