-- NYC Housing Market Analysis
-- Market-driven SQL questions

-- Q1: Where is housing market liquidity concentrated?
SELECT borough, COUNT(*) AS number_of_sales
FROM housing
GROUP BY borough
ORDER BY number_of_sales DESC;

-- Q2: Which cities show the highest average housing prices?
SELECT borough, 
ROUND (AVG(price),0) AS avg_price
FROM housing
GROUP BY borough
ORDER BY avg_price DESC;

-- Q3: Are there high-price, low-volume markets?
SELECT borough, 
COUNT(*) AS number_of_sales, 
ROUND (AVG(price), 0) AS avg_price
FROM housing
GROUP BY borough;
