# Bank Reviews Database Schema

## 📋 Overview
A PostgreSQL database for storing bank information, user reviews, and sentiment analysis data for banking applications.

## 🗄️ Tables

### 1. `banks`
Stores bank and mobile application information.

| Column | Data Type | Constraints | Description |
|--------|-----------|-------------|-------------|
| `bank_id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing unique identifier |
| `bank_name` | `VARCHAR(255)` | `UNIQUE NOT NULL` | Name of the bank |
| `app_name` | `VARCHAR(255)` | `UNIQUE NOT NULL` | Name of the mobile application |

### 2. `reviews`
Stores user reviews with sentiment analysis.

| Column | Data Type | Constraints | Description |
|--------|-----------|-------------|-------------|
| `review_id` | `VARCHAR` | `PRIMARY KEY` | External review identifier |
| `bank_id` | `INTEGER` | `NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE` | Foreign key to banks table |
| `review_text` | `TEXT` | `NOT NULL` | Full text of the review |
| `review_date` | `DATE` | | Date the review was posted |
| `sentiment_label` | `VARCHAR(50)` | | Sentiment category  |
| `sentiment_score` | `DECIMAL(5,4)` | | Numerical sentiment score  |
| `source_` | `VARCHAR(100)` | | Source platform of the review |

## 🔗 Relationships
- **One-to-Many**: One bank → Many reviews
- **Foreign Key**: `reviews.bank_id` references `banks.bank_id`
- **Cascade Delete**: Deleting a bank automatically deletes all its reviews

## 🚀 Performance Indexes
```sql
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_score);
```

## 🛠️ Installation
1. Create the database:
```sql
CREATE DATABASE bank_reviews;
```

2. Run the schema:
```sql
\i schema.sql
```

## 📈 Data Model Diagram

banks
├── bank_id (PK)
├── bank_name
└── app_name
    ↓
reviews
├── review_id (PK)
├── bank_id (FK) → banks.bank_id
├── review_text
├── review_date
├── sentiment_label
├── sentiment_score
└── source_