import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


THEME_KEYWORDS = {
    "App Performance": [
        "crash", "freeze", "slow", "lag", "bug", "error", "update",
        "loading", "responsive", "performance"
    ],
    "Navigation & UX": [
        "navigate", "menu", "interface", "design", "layout", "easy",
        "user friendly", "intuitive"
    ],
    "Customer Support": [
        "support", "call", "help", "service", "staff", "agent",
        "branch", "respond", "response"
    ],
    "Transactions": [
        "transfer", "payment", "deposit", "withdraw", "transaction",
        "processing", "delay", "pending"
    ],
    "Security": [
        "password", "otp", "verification", "secure", "locked",
        "authentication"
    ]
}

IMPROVEMENT_SUGGESTIONS = {
    "App Performance": "Improve app stability, remove bugs, optimize loading time.",
    "Navigation & UX": "Simplify menu layout and improve navigation flow.",
    "Customer Support": "Reduce service time and improve responsiveness.",
    "Transactions": "Optimize transfer speed and reduce payment failures.",
    "Security": "Simplify OTP steps and reduce unnecessary authentication.",
    "Other": "General app improvements based on user feedback."
}



def extract_themes(text):
    """Extract themes based on keyword matching."""
    text = str(text).lower()
    matched = [theme for theme, words in THEME_KEYWORDS.items()
               if any(w in text for w in words)]
    return matched if matched else ["Other"]


def compute_sentiment_pct(df):
    """Return sentiment % distribution for a bank DataFrame."""
    counts = df["sentiment_label"].value_counts(normalize=True) * 100
    return {
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0)
    }



class BankDataVisualizer:

    def __init__(self, reports_dir="bank_reports"):
        self.reports_dir = reports_dir
        self.bank_dfs = {}
        self._load_reports()
        self.df = pd.concat(self.bank_dfs.values(), ignore_index=True)

   
    # LOAD REPORTS
   
    def _load_reports(self):
        if not os.path.exists(self.reports_dir):
            raise ValueError(f"Directory {self.reports_dir} not found.")

        for file in os.listdir(self.reports_dir):
            if file.endswith("_analysis.csv"):
                bank = file.replace("_analysis.csv", "").replace("_", " ").title()
                df = pd.read_csv(os.path.join(self.reports_dir, file))

                df["bank"] = bank
                df["themes"] = df["review_text"].apply(extract_themes)
                self.bank_dfs[bank] = df

        print(f"Loaded {len(self.bank_dfs)} bank reports.")

   
    # BANK SUMMARY
   
    def compare_banks_sentiment(self):
        summary = [{
            "bank_name": bank,
            "avg_rating": df["rating"].mean(),
            "total_reviews": len(df),
            **compute_sentiment_pct(df)
        } for bank, df in self.bank_dfs.items()]
        return pd.DataFrame(summary)

   
    # DRIVER/PAIN POINT IDENTIFICATION
   
    def identify_drivers_and_painpoints(self):
        results = {}

        for bank, df in self.bank_dfs.items():
            theme_counts = {}

            for _, row in df.iterrows():
                sentiments = row["sentiment_label"]
                for theme in row["themes"]:
                    theme_counts.setdefault(theme, {"positive": 0, "negative": 0})
                    if sentiments in ["positive", "negative"]:
                        theme_counts[theme][sentiments] += 1

            drivers = [t for t, s in theme_counts.items() if s["positive"] > s["negative"]]
            painpoints = [t for t, s in theme_counts.items() if s["negative"] > s["positive"]]

            results[bank] = {
                "drivers": drivers[:5],
                "painpoints": painpoints[:5]
            }

        return results

   
    # RECOMMENDATIONS
   
    def get_bank_recommendations(self):
        info = self.identify_drivers_and_painpoints()
        return {
            bank: [IMPROVEMENT_SUGGESTIONS.get(p, IMPROVEMENT_SUGGESTIONS["Other"])
                   for p in details["painpoints"]]
            for bank, details in info.items()
        }

   
    # PAIRWISE COMPARISON
   
    def compare_two_banks(self, bank1, bank2):
        info = self.identify_drivers_and_painpoints()

        if bank1 not in info or bank2 not in info:
            return "Invalid bank names."

        b1, b2 = info[bank1], info[bank2]

        return {
            "bank1": bank1,
            "bank2": bank2,
            "bank1_drivers": b1["drivers"],
            "bank2_drivers": b2["drivers"],
            "bank1_painpoints": b1["painpoints"],
            "bank2_painpoints": b2["painpoints"],
            "bank1_strengths": list(set(b1["drivers"]) - set(b2["drivers"])),
            "bank2_strengths": list(set(b2["drivers"]) - set(b1["drivers"])),
            "shared_painpoints": list(set(b1["painpoints"]) & set(b2["painpoints"]))
        }

    def compare_all_banks_matrix(self):
        banks = list(self.bank_dfs.keys())
        return {
            f"{banks[i]} vs {banks[j]}": self.compare_two_banks(banks[i], banks[j])
            for i in range(len(banks))
            for j in range(i + 1, len(banks))
        }

   
    # VISUALIZATIONS
   
    def plot_sentiment_distribution(self):
        summary = self.compare_banks_sentiment()
        summary.set_index("bank_name")[["positive", "neutral", "negative"]].plot(
            kind="bar", stacked=True
        )
        plt.title("Sentiment Distribution Across Banks")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_rating_distribution(self):
        merged = self.df
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.boxplot(data=merged, x="bank", y="rating", ax=axes[0])
        axes[0].set_title("Rating Distribution — Boxplot")

        sns.violinplot(data=merged, x="bank", y="rating", ax=axes[1])
        axes[1].set_title("Rating Distribution — Violin")

        for ax in axes:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_keyword_clouds(self):
        for bank, df in self.bank_dfs.items():
            wc = WordCloud(width=1200, height=600, background_color="white")\
                .generate(" ".join(df["review_text"].astype(str)))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud — {bank}")
            plt.show()

# main 
    def run_all(self):
        print("\n----------- VISUALS -----------")
        self.plot_sentiment_distribution()
        self.plot_rating_distribution()
        self.plot_keyword_clouds()

        print("\n----------- ANALYTICS -----------")
        print("\nDrivers & Pain Points:")
        print(self.identify_drivers_and_painpoints())

        print("\nRecommendations:")
        print(self.get_bank_recommendations())

        print("\nPairwise Comparisons:")
        print(self.compare_all_banks_matrix())

        print("\nAll tasks completed.")


if __name__ == "__main__":
    viz = BankDataVisualizer("bank_reports")
    viz.run_all()
