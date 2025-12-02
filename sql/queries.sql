--example queries

-- Get all reviews for a specific bank
SELECT r.*, b.bank_name 
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE b.bank_name = 'Dashen Bank';


--Get average sentiment per bank
SELECT b.bank_name, AVG(r.sentiment_score) as avg_sentiment
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_sentiment DESC;



--Count reviews by month
SELECT 
  DATE_TRUNC('month', review_date) as month,
  COUNT(*) as review_count
FROM reviews
GROUP BY month
ORDER BY month;