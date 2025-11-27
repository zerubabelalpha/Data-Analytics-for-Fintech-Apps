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
    "Dashenbank":"Dashenbank"
}

SCRAPPING_CONFIG={
    "reviews_per_bank":int(os.getenv("REVIEWS_PER_BANK",420)),
    "max_retries":int(os.getenv("MAX_RETRIES",3)),
    "lang":"en",
    "country":"et"
}

DATA_PATHS={
    "raw":"data/raw",
    "processed":"data/processed",
    "raw_reviews":"data/raw_reviews",
    "processed_reviews":"data/processed_reviews",
    "sentiment_results":"data/sentiment_results",
    "final_results":"data/final_results"
}