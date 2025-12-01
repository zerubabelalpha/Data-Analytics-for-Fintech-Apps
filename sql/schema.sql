CREATE TABLE banks(
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(255) UNIQUE NOT NULL,
    app_name VARCHAR(255) UNIQUE NOT NULL
);

create table reviews(
    review_id VARCHAR PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,  
    review_date DATE,
    sentiment_label VARCHAR(50),
    sentiment_score DECIMAL(5,4),  
    source_ VARCHAR(100)
    
  
);

-- Create indexes
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_date ON reviews(review_date);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_score);