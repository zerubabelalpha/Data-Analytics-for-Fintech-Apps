import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os


#Download required NLTK data
def download_nltk_resources():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        print("NLTK download completed or already available")

# run these once
# download_nltk_resources()
sia = SentimentIntensityAnalyzer()

class BankReviewAnalyzer:
    def __init__(self):
        self.df = None
        self.bank_dfs = {}
        self.analysis_results = {}
        self.count_vec = None
        self.tfidf_vec = None
        self.dictionary = None

    def load_data(self, filepath: str) -> bool:
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {len(self.df)} reviews")
            self._split_data_by_bank()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _split_data_by_bank(self):
        if self.df is None or 'bank_name' not in self.df.columns:
            print("No data loaded or 'bank_name' column missing")
            return

        self.bank_dfs = {}
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank].copy()
            self.bank_dfs[bank] = bank_data
            print(f"  - {bank}: {len(bank_data)} reviews")

    
    # PREPROCESSING
    def preprocess_data(self, bank_name: str = None):
        if bank_name:
            if bank_name in self.bank_dfs:
                self._preprocess_single_bank(bank_name)
        else:
            if self.df is not None:
                self.df = self._preprocess_dataframe(self.df, "combined")
            for bank_name in self.bank_dfs:
                self._preprocess_single_bank(bank_name)

    def _preprocess_single_bank(self, bank_name: str):
        self.bank_dfs[bank_name] = self._preprocess_dataframe(
            self.bank_dfs[bank_name], bank_name
        )

    def _preprocess_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        df_processed = df.copy()

        
        df_processed["sentiment_score"] = df_processed["review_text"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )

        #  Sentiment label 
        df_processed["sentiment_label"] = df_processed["sentiment_score"].apply(
            self.vader_label
        )

        # Basic cleaning
        df_processed["clean_text"] = df_processed["review_text"].str.lower()
        df_processed["tokens"] = df_processed["clean_text"].str.split()

        stop_words = set(stopwords.words("english"))
        df_processed["tokens_nostop"] = df_processed["tokens"].apply(
            lambda w: [x for x in w if x not in stop_words]
        )

        print(f"Preprocessing completed for {name}: {len(df_processed)} reviews")
        return df_processed

    
    # SENTIMENT LABEL
    @staticmethod
    def vader_label(c):
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        else:
            return "neutral"

    
    # MAIN ANALYSIS
    def analyze_bank_sentiment(self, bank_name: str = None):
        if bank_name:
            return self._analyze_single_bank(bank_name)
        else:
            return {b: self._analyze_single_bank(b) for b in self.bank_dfs}

    def _analyze_single_bank(self, bank_name: str):
        if bank_name not in self.bank_dfs:
            print(f"Bank '{bank_name}' not found")
            return {}

        print(f"\n===== ANALYZING: {bank_name} =====")
        df = self.bank_dfs[bank_name]

        results = {}
        results["basic_stats"] = self._get_basic_stats(df, bank_name)
        results["frequency_analysis"] = self._analyze_frequency_based(df, bank_name)
        results["tfidf_analysis"] = self._analyze_tfidf(df, bank_name)

        # THEMATIC TOPIC CLUSTERING
        results["themes"] = self._extract_themes(df, bank_name)

        # Store review theme assignment
        self.bank_dfs[bank_name] = self._assign_themes_to_reviews(
            df, results["themes"]
        )

        self.analysis_results[bank_name] = results
        return results

    
    # BASIC STATS
    def _get_basic_stats(self, df, bank_name):
        stats = {
            "total_reviews": len(df),
            "avg_rating": df["rating"].mean(),
            "sentiment_distribution": df["sentiment_label"].value_counts().to_dict()
        }
        return stats

    
    # FREQUENCY ANALYSIS WITH NGRAMS (1–3)
    def _analyze_frequency_based(self, df, bank_name, top_n=15):
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        X = vec.fit_transform(df["clean_text"])

        total = np.asarray(X.sum(axis=0)).flatten()
        vocab = np.array(vec.get_feature_names_out())

        freq_df = pd.DataFrame({"ngram": vocab, "count": total})
        return freq_df.sort_values("count", ascending=False).head(top_n)

    
    # TF-IDF (1–3 GRAM)
    def _analyze_tfidf(self, df, bank_name, top_n=15):
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
        X = vec.fit_transform(df["clean_text"])

        weights = np.asarray(X.mean(axis=0)).flatten()
        vocab = np.array(vec.get_feature_names_out())

        tfidf_df = pd.DataFrame({"ngram": vocab, "tfidf": weights})
        return tfidf_df.sort_values("tfidf", ascending=False).head(top_n)

    
    # THEMATIC GROUPING  
    def _extract_themes(self, df, bank_name):
        """
        Cluster keywords into themes manually + auto keyword expansion
        """
        base_themes = {
            "Account Access Issues": ["login", "log in", "access", "password", "locked", "authentication",
                                      "otp", "pin", "blocked", "fail to login", "verification", "credential"],
            "Transaction Performance": ["transfer", "payment", "transaction", "delay", "processing", "pending", "not working",
                                        "failed", "reversal", "credited", "debited", "real time", "slow transfer"],
            "User Interface & Experience": ["app", "interface", "navigate", "design", "slow", "lag", "freeze",
                                            "bug", "glitch", "crash", "layout", "usability","good", "dark mode", "update"],
            "Customer Support": ["support", "service", "help", "staff", "agent", "customer care",
                                 "call center", "respond", "response", "rude", "assistance"],
            "Feature Requests": ["feature", "add", "improve", "option", "missing", "update feature",
                                 "needs", "should include", "would like"]
        }

        # Extract most common n-grams for additional signal
        freq_words = self._analyze_frequency_based(df, bank_name, top_n=30)

        # Add top ngrams if not mapped
        all_ngrams = list(freq_words["ngram"])
        themes = {t: set(words) for t, words in base_themes.items()}

        # simple expansion: attach uncommon keywords to closest theme by basic heuristic
        for ng in all_ngrams:
            assigned = False
            for theme, words in themes.items():
                if any(w in ng for w in words):
                    themes[theme].add(ng)
                    assigned = True
                    break
            if not assigned:
                # fallback dump theme
                themes.setdefault("Miscellaneous Feedback", set()).add(ng)

        return themes
 
 
    # ASSIGN THEMES TO EACH REVIEW
    def _assign_themes_to_reviews(self, df, themes):
        def identify_themes_for_text(text):
            hits = []
            for theme, words in themes.items():
                for w in words:
                    if w in text:
                        hits.append(theme)
                        break
            return list(set(hits)) if hits else ["Uncategorized"]

        df["identified_theme(s)"] = df["clean_text"].apply(identify_themes_for_text)
        return df

    
    # BANK COMPARISON 
    
    def compare_banks_sentiment(self):
        if not self.bank_dfs:
            print("No bank data available")
            return

        comparison_data = []
        for bank_name, bank_df in self.bank_dfs.items():
            sentiment = bank_df["sentiment_label"].value_counts(normalize=True) * 100
            comparison_data.append({
                "bank_name": bank_name,
                "total_reviews": len(bank_df),
                "avg_rating": bank_df["rating"].mean(),
                "positive_pct": sentiment.get("positive", 0),
                "neutral_pct": sentiment.get("neutral", 0),
                "negative_pct": sentiment.get("negative", 0),
            })

        comparison_df=pd.DataFrame(comparison_data)
    
        print("\n" + "="*60)
        print("BANK COMPARISON SUMMARY")
        print("="*60)
        for _, row in comparison_df.iterrows():
                    print(f"\n{row['bank_name']}:")
                    print(f"  Reviews: {row['total_reviews']}")
                    print(f"  Avg Rating: {row['avg_rating']:.2f}")
                    print(f"  Positive: {row['positive_pct']:.1f}%")
                    print(f"  Neutral: {row['neutral_pct']:.1f}%")
                    print(f"  Negative: {row['negative_pct']:.1f}%")
                
        return comparison_df

        
    # REPORT GENERATION
    
    def generate_bank_reports(self, output_dir="bank_reports"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for bank_name in self.bank_dfs:
            output_file = os.path.join(
                output_dir, f"{bank_name.replace(' ', '_').lower()}_analysis.csv"
            )
            self.bank_dfs[bank_name][[
                "review_id",
                "review_text",
                "rating",
                "review_date",
                "sentiment_label",
                "sentiment_score",
                "identified_theme(s)",
                "bank_name",
                "source"
            ]].to_csv(output_file, index=False)

            print(f"Report saved: {output_file}")

    def run_complete_multi_bank_analysis(self, filepath: str = None):
        """Full end-to-end analysis pipeline."""
        
        # Load or create data
        if filepath:
            if not self.load_data(filepath):
                print("Using sample data instead.")
                
        # Preprocess text
        self.preprocess_data()

        # Individual bank analysis
        bank_results = self.analyze_bank_sentiment()

        # Summary comparison
        comparison = self.compare_banks_sentiment()

        # Reports
        self.generate_bank_reports()

        return {
            "individual_bank_results": bank_results,
            "bank_comparison": comparison,
            "all_banks_data": self.bank_dfs
        }