import re
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/trial-data")
async def get_product_datav2():
    x = compare_images_v2("/Users/abhinavpersonal/Downloads/flp.jpeg",
                          "/Users/abhinavpersonal/Downloads/amz.webp")
    return x


@app.get('/amazon/content/scrape/v1')
def scrape_content(link: str):
    if not link:
        return {'error': 'Link parameter is missing'}

    # Set up Chrome WebDriver options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Optional: Run Chrome in headless mode

    # Provide the path to the Chrome WebDriver executable
    webdriver_path = '/home/abhinav.personal/projects/chromedriver'  # Replace with the actual path

    # Set up Chrome WebDriver
    driver = webdriver.Chrome(executable_path=webdriver_path, options=chrome_options)

    try:
        # Make a request to the webpage
        driver.get(link)

        # Extract the content of the hidden div using JavaScript injection
        script = """
               const div = document.getElementById('a-popover-10');
               if (div) {
                   return div.innerText;
               } else {
                   return null;
               }
               """
        content = driver.execute_script(script)

        if content:
            return {'content': content}
        else:
            return {'error': 'Hidden div not found'}
    finally:
        # Quit the WebDriver after scraping
        driver.quit()

def compare_images_v2(image1_path, image2_path):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

    # Create a brute-force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matching keypoints
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Apply RANSAC to estimate the best transformation matrix
    _, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Calculate the percentage of inlier matches
    inlier_ratio = np.sum(mask) / len(mask)

    # Calculate the color histograms
    color_hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize the histograms
    cv2.normalize(color_hist1, color_hist1)
    cv2.normalize(color_hist2, color_hist2)

    # Calculate the Euclidean distance between color histograms
    color_distance = cv2.compareHist(color_hist1, color_hist2, cv2.HISTCMP_BHATTACHARYYA)

    # Set thresholds for inlier ratio and color distance
    inlier_threshold = 0.5  # Adjust this value as needed
    color_threshold = 0.5  # Adjust this value as needed

    print(inlier_ratio)
    print(color_distance)

    # Compare the inlier ratio and color distance with the thresholds
    if inlier_ratio >= inlier_threshold and color_distance <= color_threshold:
        return True  # Images are similar or identical
    else:
        return False  # Images are different


# @app.get("/trial-data-2")
# async def get_product_data():
#     api_data = {
#         "Amazon": "Apple iPhone 13 512GB YELLOW",
#         "Flipkart": "128GB Apple iPhone 13  - (Product) YELLOW",
#         "Reliance": "iPhone Apple  13 (128GB) -  YELLOW"
#     }
#
#     cleaned_titles = clean_titles(api_data)
#     is_similar = compare_titles(cleaned_titles)
#
#     if is_similar:
#         print("The titles are similar.")
#     else:
#         print("The titles are not similar.")
#
#     return is_similar


# @app.post("/product-data")
# async def get_product_data(request_data: Dict[str, str]):
#     api_data = {
#         "Amazon": request_data.get("Amazon", ""),
#         "Flipkart": request_data.get("Flipkart", ""),
#         "Reliance": request_data.get("Reliance", "")
#     }
#
#     cleaned_titles = clean_titles(api_data)
#     is_similar = compare_titles(cleaned_titles)
#
#     if is_similar:
#         return "The titles are similar."
#     else:
#         return "The titles are not similar."
#
#
# def clean_titles(titles):
#     clean_titles = {}
#     for key, title in titles.items():
#         clean_title = ''.join(e for e in title if e.isalnum() or e.isspace()).lower()
#         clean_titles[key] = clean_title
#     return clean_titles
#
#
# def compare_titles(titles):
#     total_titles = len(titles)
#     print(total_titles)
#     matching_pairs = 0
#
#     title_array = list(titles.values())
#
#     for i in range(total_titles - 1):
#         title1 = title_array[i]
#         print(title1)
#
#         for j in range(i + 1, total_titles):
#             title2 = title_array[j]
#             print(title2)
#
#             if has_similar_words(title1, title2) and has_same_storage_capacity(title1, title2):
#                 matching_pairs += 1
#
#     unique_title_combinations = (total_titles * (total_titles - 1)) / 2
#
#     if unique_title_combinations != 0:  # Check if the value is non-zero
#         matching_percentage = matching_pairs / unique_title_combinations
#     else:
#         matching_percentage = 0.0
#
#     print(matching_percentage)
#
#     similarity_threshold = 0.90  # Adjust the similarity threshold as needed
#
#     return matching_percentage >= similarity_threshold
#
# def has_similar_words(title1, title2):
#     words1 = title1.split()
#     words2 = title2.split()
#
#     matching_words = 0
#
#     for word in words1:
#         if any(word.lower() in w.lower() for w in words2):
#             matching_words += 1
#
#     word_percentage = matching_words / len(words1)
#     return word_percentage > 0.8
#
#
# def has_same_storage_capacity(title1, title2):
#     storage_capacity1 = extract_storage_capacity(title1)
#     storage_capacity2 = extract_storage_capacity(title2)
#     return storage_capacity1 is not None and storage_capacity1 == storage_capacity2
#
#
# def extract_storage_capacity(title):
#     matches = re.findall(r'\b(\d+)\s*gb\b', title)
#     if matches:
#         return matches[0] + "GB"
#     return None


@app.post("/product-data")
async def compare_strings(request_data: Dict[str, str]):
    title1 = request_data.get("title1", "")
    title2 = request_data.get("title2", "")

    result = compare_strings(title1, title2)
    return result


def compare_strings(title1, title2):
    clean_title1 = clean_title(title1)
    clean_title2 = clean_title(title2)

    if has_similar_words(clean_title1, clean_title2) and has_same_storage_capacity(clean_title1, clean_title2):
        return "The titles are similar."
    else:
        return "The titles are not similar."


def clean_title(title):
    return ''.join(e for e in title if e.isalnum() or e.isspace()).lower()


def has_similar_words(title1, title2):
    words1 = title1.split()
    words2 = title2.split()

    matching_words = 0

    for word in words1:
        if any(word.lower() in w.lower() for w in words2):
            matching_words += 1

    word_percentage = matching_words / len(words1)
    return word_percentage > 0.88


def has_same_storage_capacity(title1, title2):
    storage_capacity1 = extract_storage_capacity(title1)
    storage_capacity2 = extract_storage_capacity(title2)

    if storage_capacity1 is not None and storage_capacity2 is not None:
        return any(capacity1 == capacity2 for capacity1 in storage_capacity1 for capacity2 in storage_capacity2)

    return False

def extract_storage_capacity(title):
    matches = re.findall(r'\b(\d+)\s*gb\b', title)
    if matches:
        return matches[0] + "GB"
    return None
