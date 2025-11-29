import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import spacy
import os
from typing import Dict, List, Optional

# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        print("NLTK download completed or already available")

# run these once
download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# Make plots prettier
sns.set_style(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

class BankReviewAnalyzer:
    def __init__(self):
        self.df = None
        self.bank_dfs = {}  # Dictionary to store individual bank data
        self.analysis_results = {}  # Store results per bank
        self.count_vec = None
        self.tfidf_vec = None
        self.lda_model = None
        self.dictionary = None
        
    def load_data(self, filepath: str) -> bool:
        """
        Load bank review data from CSV file
        
        Expected columns:
        review_id, review_text, rating, review_date, review_year, review_month,
        bank_code, bank_name, user_name, thumbs_up, text_length, source
        """
        try:
            self.df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {len(self.df)} reviews")
            
            # Split data by bank
            self._split_data_by_bank()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _split_data_by_bank(self):
        """Split the main dataframe by bank for individual analysis"""
        if self.df is None or 'bank_name' not in self.df.columns:
            print("No data loaded or 'bank_name' column not found")
            return
        
        self.bank_dfs = {}
        bank_names = self.df['bank_name'].unique()
        
        print(f"Found {len(bank_names)} banks: {list(bank_names)}")
        
        for bank in bank_names:
            bank_data = self.df[self.df['bank_name'] == bank].copy()
            self.bank_dfs[bank] = bank_data
            print(f"  - {bank}: {len(bank_data)} reviews")
    
    
    
    def preprocess_data(self, bank_name: str = None):
        """
        Preprocess the review data for specific bank or all banks
        
        Args:
            bank_name: If provided, preprocess only this bank's data. 
                      If None, preprocess all banks and combined data.
        """
        if bank_name:
            # Preprocess specific bank
            if bank_name in self.bank_dfs:
                self._preprocess_single_bank(bank_name)
            else:
                print(f"Bank '{bank_name}' not found")
        else:
            # Preprocess all banks
            if self.df is not None:
                self.df = self._preprocess_dataframe(self.df, "combined")
            
            for bank_name, bank_df in self.bank_dfs.items():
                self._preprocess_single_bank(bank_name)
    
    def _preprocess_single_bank(self, bank_name: str):
        """Preprocess data for a single bank"""
        if bank_name in self.bank_dfs:
            self.bank_dfs[bank_name] = self._preprocess_dataframe(
                self.bank_dfs[bank_name], bank_name
            )
    
    def _preprocess_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Preprocess a dataframe and return the processed version"""
        df_processed = df.copy()
        
        # Convert rating to sentiment label
        df_processed["sentiment_label"] = df_processed["rating"].apply(self.rating_to_label)
        
        # Basic text cleaning
        df_processed["clean_text"] = df_processed["review_text"].str.lower()
        
        # Tokenization
        df_processed["tokens"] = df_processed["clean_text"].str.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        df_processed["tokens_nostop"] = df_processed["tokens"].apply(
            lambda words: [w for w in words if w not in stop_words]
        )
        
        print(f"Preprocessing completed for {name}: {len(df_processed)} reviews")
        return df_processed
    
    @staticmethod
    def rating_to_label(r):
        """Convert star rating to sentiment label"""
        if r <= 2:
            return "negative"
        elif r == 3:
            return "neutral"
        else:
            return "positive"
    
    def analyze_bank_sentiment(self, bank_name: str = None):
        """
        Perform sentiment analysis for specific bank or all banks
        
        Args:
            bank_name: If provided, analyze only this bank. 
                      If None, analyze all banks individually.
        """
        if bank_name:
            return self._analyze_single_bank(bank_name)
        else:
            results = {}
            for bank in self.bank_dfs.keys():
                results[bank] = self._analyze_single_bank(bank)
            return results
    
    def _analyze_single_bank(self, bank_name: str) -> Dict:
        """Perform complete analysis for a single bank"""
        if bank_name not in self.bank_dfs:
            print(f"Bank '{bank_name}' not found")
            return {}
        
        print(f"\n{'='*50}")
        print(f"ANALYZING: {bank_name}")
        print(f"{'='*50}")
        
        bank_df = self.bank_dfs[bank_name]
        results = {}
        
        # 1. Basic statistics
        results['basic_stats'] = self._get_basic_stats(bank_df, bank_name)
        
        # 2. Sentiment analysis
        results['textblob_sentiment'] = self._analyze_textblob_sentiment(bank_df)
        results['vader_sentiment'] = self._analyze_vader_sentiment(bank_df)
        
        # 3. Text analysis
        results['frequency_analysis'] = self._analyze_frequency_based(bank_df, bank_name)
        results['tfidf_analysis'] = self._analyze_tfidf(bank_df, bank_name)
        
        # 4. Topic modeling (if enough reviews)
        if len(bank_df) >= 5:
            results['topics'] = self._perform_topic_modeling(bank_df, bank_name)
        
        self.analysis_results[bank_name] = results
        return results
    
    def _get_basic_stats(self, df: pd.DataFrame, bank_name: str) -> Dict:
        """Get basic statistics for a bank's reviews"""
        stats = {
            'total_reviews': len(df),
            'average_rating': df['rating'].mean(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
        }
        
        print(f"\nBasic Statistics for {bank_name}:")
        print(f"  - Total Reviews: {stats['total_reviews']}")
        print(f"  - Average Rating: {stats['average_rating']:.2f}")
        print(f"  - Sentiment Distribution: {stats['sentiment_distribution']}")
        
        return stats
    
    def _analyze_textblob_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform TextBlob sentiment analysis"""
        df_analyzed = df.copy()
        df_analyzed["tb_polarity"] = df_analyzed["review_text"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        df_analyzed["tb_subjectivity"] = df_analyzed["review_text"].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        df_analyzed["tb_sentiment"] = df_analyzed["tb_polarity"].apply(self.polarity_to_label)
        
        return df_analyzed[['review_text', 'rating', 'tb_polarity', 'tb_subjectivity', 'tb_sentiment']]
    
    def _analyze_vader_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform VADER sentiment analysis"""
        df_analyzed = df.copy()
        
        def vader_compound(text):
            return sia.polarity_scores(text)["compound"]
        
        df_analyzed["vader_compound"] = df_analyzed["review_text"].apply(vader_compound)
        df_analyzed["vader_sentiment"] = df_analyzed["vader_compound"].apply(self.vader_label)
        
        return df_analyzed[['review_text', 'rating', 'vader_compound', 'vader_sentiment']]
    
    def _analyze_frequency_based(self, df: pd.DataFrame, bank_name: str, top_n: int = 10):
        """Frequency-based word analysis"""
        count_vec = CountVectorizer(stop_words="english")
        X_counts = count_vec.fit_transform(df["clean_text"])
        
        word_counts = np.asarray(X_counts.sum(axis=0)).flatten()
        vocab = np.array(count_vec.get_feature_names_out())
        
        freq_df = pd.DataFrame({"word": vocab, "count": word_counts})
        freq_df = freq_df.sort_values("count", ascending=False)
        
        print(f"\nTop {top_n} frequent words for {bank_name}:")
        for i, row in freq_df.head(top_n).iterrows():
            print(f"  {row['word']}: {row['count']}")
        
        return freq_df
    
    def _analyze_tfidf(self, df: pd.DataFrame, bank_name: str, top_n: int = 10):
        """TF-IDF based word importance analysis"""
        tfidf_vec = TfidfVectorizer(stop_words="english")
        X_tfidf = tfidf_vec.fit_transform(df["clean_text"])
        
        tfidf_means = np.asarray(X_tfidf.mean(axis=0)).flatten()
        vocab_tfidf = np.array(tfidf_vec.get_feature_names_out())
        
        tfidf_df = pd.DataFrame({"word": vocab_tfidf, "tfidf": tfidf_means})
        tfidf_df = tfidf_df.sort_values("tfidf", ascending=False)
        
        print(f"\nTop {top_n} TF-IDF words for {bank_name}:")
        for i, row in tfidf_df.head(top_n).iterrows():
            print(f"  {row['word']}: {row['tfidf']:.4f}")
        
        return tfidf_df
    
    def _perform_topic_modeling(self, df: pd.DataFrame, bank_name: str, num_topics: int = 2):
        """Perform LDA topic modeling"""
        try:
            dictionary = Dictionary(df["tokens_nostop"])
            corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens_nostop"]]
            
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=10,
                random_state=42
            )
            
            topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
            
            print(f"\nTopics for {bank_name}:")
            for i, topic in topics:
                topic_words = [word for word, weight in topic]
                print(f"  Topic {i+1}: {', '.join(topic_words)}")
            
            return topics
        except Exception as e:
            print(f"Topic modeling failed for {bank_name}: {e}")
            return None
    
    @staticmethod
    def polarity_to_label(p):
        """Convert polarity score to sentiment label"""
        if p > 0.1:
            return "positive"
        elif p < -0.1:
            return "negative"
        else:
            return "neutral"
    
    @staticmethod
    def vader_label(c):
        """Convert VADER compound score to sentiment label"""
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def compare_banks_sentiment(self):
        """Compare sentiment analysis results across all banks"""
        if not self.bank_dfs:
            print("No bank data available")
            return
        
        comparison_data = []
        
        for bank_name, bank_df in self.bank_dfs.items():
            # Calculate sentiment percentages
            sentiment_counts = bank_df['sentiment_label'].value_counts(normalize=True) * 100
            
            comparison_data.append({
                'bank_name': bank_name,
                'total_reviews': len(bank_df),
                'avg_rating': bank_df['rating'].mean(),
                'positive_pct': sentiment_counts.get('positive', 0),
                'neutral_pct': sentiment_counts.get('neutral', 0),
                'negative_pct': sentiment_counts.get('negative', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
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
    
    def visualize_bank_comparison(self):
        """Create visualizations comparing all banks"""
        if not self.bank_dfs:
            print("No bank data available")
            return
        
        # Create comparison data
        comparison_df = self.compare_banks_sentiment()
        
        if comparison_df is None or comparison_df.empty:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average Rating Comparison
        axes[0, 0].bar(comparison_df['bank_name'], comparison_df['avg_rating'], color='skyblue')
        axes[0, 0].set_title('Average Rating by Bank')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Review Count Comparison
        axes[0, 1].bar(comparison_df['bank_name'], comparison_df['total_reviews'], color='lightgreen')
        axes[0, 1].set_title('Total Reviews by Bank')
        axes[0, 1].set_ylabel('Number of Reviews')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Sentiment Distribution (Stacked Bar)
        sentiment_data = comparison_df[['positive_pct', 'neutral_pct', 'negative_pct']]
        bar_width = 0.8
        bottom = np.zeros(len(comparison_df))
        
        colors = ['#4CAF50', '#FFC107', '#F44336']  # green, yellow, red
        
        for i, col in enumerate(['positive_pct', 'neutral_pct', 'negative_pct']):
            axes[1, 0].bar(comparison_df['bank_name'], comparison_df[col], 
                          bottom=bottom, label=col.replace('_pct', '').title(), 
                          color=colors[i], width=bar_width)
            bottom += comparison_df[col]
        
        axes[1, 0].set_title('Sentiment Distribution by Bank')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        
        # 4. Sentiment Pie Chart for all banks combined
        if self.df is not None:
            overall_sentiment = self.df['sentiment_label'].value_counts()
            axes[1, 1].pie(overall_sentiment.values, labels=overall_sentiment.index, 
                          autopct='%1.1f%%', colors=colors)
            axes[1, 1].set_title('Overall Sentiment Distribution (All Banks)')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_bank_reports(self, output_dir: str = "bank_reports"):
        """Generate individual reports for each bank"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for bank_name in self.bank_dfs.keys():
            # Analyze bank if not already done
            if bank_name not in self.analysis_results:
                self._analyze_single_bank(bank_name)
            
            # Save bank data to CSV
            output_file = os.path.join(output_dir, f"{bank_name.replace(' ', '_').lower()}_analysis.csv")
            self.bank_dfs[bank_name].to_csv(output_file, index=False)
            print(f"Report saved: {output_file}")
    
    def run_complete_multi_bank_analysis(self, filepath: str = None):
        """Run complete analysis for all banks"""
        if filepath:
            if not self.load_data(filepath):
                print("Using sample data instead")
                self.create_sample_data()
        else:
            self.create_sample_data()
        
        # Preprocess all data
        self.preprocess_data()
        
        # Analyze each bank individually
        bank_results = self.analyze_bank_sentiment()
        
        # Compare banks
        comparison = self.compare_banks_sentiment()
        
        # Create visualizations
        self.visualize_bank_comparison()
        
        # Generate reports
        self.generate_bank_reports()
        
        return {
            'individual_bank_results': bank_results,
            'bank_comparison': comparison,
            'all_banks_data': self.bank_dfs
        }



def main():
    analyzer = BankReviewAnalyzer()
    analyzer.run_complete_multi_bank_analysis()
if __name__ == "__main__":
    main()