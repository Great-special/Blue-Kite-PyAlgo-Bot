from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_text_or_none(cell):
    return cell.get_text(strip=True) if cell else None

# --- Setup Headless Chrome ---
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

def get_news_data():
    # intialize the webdriver
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.forexfactory.com/")

    # --- Wait for calendar table to load ---
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'table.calendar__table'))
    )

    # Scroll to bottom to make sure all rows are loaded
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # --- Parse the page with BeautifulSoup ---
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    # Find all rows in the calendar
    rows = soup.find_all('tr', class_='calendar__row')
    print(len(rows))
    data = []

    for row in rows:
        time_ = get_text_or_none(row.find('td', class_='calendar__time'))
        currency = get_text_or_none(row.find('td', class_='calendar__currency'))

        impact_cell = row.find('td', class_='calendar__impact')
        impact_icon = impact_cell.find('span', class_='icon') if impact_cell else None
        impact = impact_icon['title'] if impact_icon and 'title' in impact_icon.attrs else None

        event = get_text_or_none(row.find('td', class_='calendar__event'))
        actual = get_text_or_none(row.find('td', class_='calendar__actual'))
        forecast = get_text_or_none(row.find('td', class_='calendar__forecast'))
        previous = get_text_or_none(row.find('td', class_='calendar__previous'))

        if any([time_, currency, impact, event]):
            data.append({
                'Time': time_,
                'Currency': currency,
                'Impact': impact,
                'Event': event,
                'Actual': actual,
                'Forecast': forecast,
                'Previous': previous
            })
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.to_csv('forex_factory_calendar.csv', index=False)
    return data



# --- Output ---
# df = pd.DataFrame(get_news_data())
# print(df)
# symbol = 'EUR/USD'  # Example symbol, you can change this to any symbol you want to analyze

# symbol_data = df[df['Currency'].isin([symbol.split('/')[0], symbol.split('/')[1]])]
# print(symbol_data)

# analyzer = SentimentIntensityAnalyzer()
# sentiment_scores = []

# for event in symbol_data['Event']:
#     score = analyzer.polarity_scores(event)['compound']
#     print(score)
#     if abs(score) > 0.2:  # Consider meaningful sentiment
#         sentiment_scores.append(score)

# try:
#     avg_score = sum(sentiment_scores) / len(sentiment_scores)
# except ZeroDivisionError:
#     avg_score = 0


# if avg_score > 0.1:
#     print("BUY")
# elif avg_score < -0.1:
#     print("SELL")
# else:
#     print("NEUTRAL")

# # Optional: Save to CSV
# df.to_csv('forex_factory_calendar.csv', index=False)
# print(f"\n Extracted {len(df)} rows and saved to 'forex_factory_calendar.csv'")
