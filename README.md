# Task 4: Advanced Analytics & Statistical Modeling

Statistical analysis and predictive modeling on the Superstore sales dataset — extracting business insights and building a model to flag unprofitable orders.

## Objective

Apply statistical analysis and basic predictive modeling to the Superstore dataset to extract business insights and build a predictive model.

## Contents

1. **Descriptive Statistics** — mean, median, mode, standard deviation, and skewness for `sales`, `profit`, `discount`, `quantity`, and `profit_margin`
2. **Hypothesis Testing**
   - Welch's t-test: does discounting affect profit?
   - Chi-square test: is loss-making independent of region?
   - 95% confidence interval for mean profit margin
3. **Time Series Analysis** — monthly sales trend (2015–2018), classical trend/seasonality/residual decomposition, 3-month moving-average forecast
4. **Customer Segmentation** — RFM-style feature engineering, K-Means clustering (elbow method to choose K), PCA visualization, segment profiling
5. **Predictive Modeling** — logistic regression to predict `is_loss` (whether an order is unprofitable), with accuracy/precision/recall/F1, confusion matrix, and top feature importance

## Dataset

`superstore_clean.csv` (not included in this repo — place it in the same directory as the notebook before running)

Expected columns: `order_id`, `order_date`, `ship_date`, `customer_id`, `segment`, `category`, `region`, `sales`, `profit`, `discount`, `quantity`, `profit_margin`, `is_loss`, `shipping_days`

Date range: 2015-01-03 to 2018-12-30

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

## How to run

1. Place `superstore_clean.csv` in the same folder as `Task4_Advanced_Analytics.ipynb`
2. Open the notebook in Jupyter / VS Code / Google Colab
3. Run all cells top to bottom

## Key Findings

1. **Discounting hurts profitability** — discounted orders average a loss, while non-discounted orders average ~$67 profit (Welch's t-test, p < 0.001)
2. **Loss rate varies significantly by region** (chi-square test, p < 0.001) — Central has the highest loss rate, West the lowest
3. **True mean profit margin** is estimated at 11.1%–12.9% (95% confidence interval)
4. **Sales show a clear upward trend with strong Q4 seasonality** — peaks in September, November, and December; slumps in January, February, and June–August
5. **Four distinct customer segments** identified via K-Means (VIP, discount-driven/unprofitable, standard, and at-risk/dormant customers), each with a tailored business recommendation
6. **Logistic regression model** predicts order-level loss with accuracy above the 82% benchmark; `discount` is by far the strongest driver of an order being unprofitable, followed by category effects (Office Supplies and Technology are less loss-prone than Furniture, all else equal)

## Business Recommendations

- Review and tighten the discounting policy, especially in categories/regions with the highest loss rates
- Investigate regional pricing, discount, and logistics practices in Central and East regions
- Use the 11.1%–12.9% profit margin confidence interval for forecasting and target-setting instead of a single point estimate
- Scale up inventory and staffing ahead of Q4 to match seasonal demand
- Prioritize retention programs (loyalty perks, early access, dedicated support) for the VIP customer segment
- Use the logistic regression model to flag high-risk orders (high discount, Furniture category) before they're finalized
