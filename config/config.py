import os
from dotenv import load_dotenv

load_dotenv()

APP_IDS={
    "CBE":os.getenv("CBE_APP_ID","com.combanketh.mobilebanking"),
    "BOA":os.getenv("BOA_APP_ID","com.boa.boaMobileBanking"),
    "DashenBank":os.getenv("DASHENBANK_APP_ID","com.dashen.dashensuperapp")
}

BANK_NAMES={
    "CBE":"Commercial Bank of Ethiopia",
    "BOA":"Bank of Abissinia",
    "DashenBank":"Dashen Bank S.c."
}

SCRAPING_CONFIG={
    "reviews_per_bank":int(os.getenv("REVIEWS_PER_BANK",420)),
    "max_retries":int(os.getenv("MAX_RETRIES",3)),
    "lang":"en",
    "country":"et"
}

DATA_PATHS={
    'raw': 'data/raw',
    'processed': 'data/processed',
    'raw_reviews': 'data/raw/reviews_raw.csv',
    'processed_reviews': 'data/processed/reviews_processed.csv',
    'sentiment_results': 'data/processed/reviews_with_sentiment.csv',
    'final_results': 'data/processed/reviews_final.csv'
}