# Music Streaming Churn & Fairness Audit
This project was developed as part of my Responsible Data Science course at NYU, where my partner and I audited a predictive model designed for a fictional streaming service, PlaylistPro. The goal was to go beyond raw predictive accuracy and examine the ethical implications of using an Algorithmic Decision-Support System (ADS) to allocate financial retention discounts. Please note that this project is still in progress!

## Project Questions
- Can a PyTorch-based Neural Network accurately predict which subscribers are at risk of churning?
- Does the model disproportionately exclude specific demographic groups (based on age or location) from receiving financial incentives?
- What are the ethical trade-offs between high precision (avoiding wasted budget) and high recall (ensuring all at-risk users get a discount)?

## Tools & Libraries
- Python
- PyTorch (Neural Network Modeling)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Preprocessing & Evaluation)

## Key Insights
- Intervention Bias: We identified that false negatives in churn prediction aren't just "missed data"—they result in the economic exclusion of users from promotional benefits.
- Model Architecture: A Multi-Layer-Perceptron (MLP) effectively processed behavioral data (skip rates, listening hours) but required careful tuning to balance fairness across age groups.
- Fairness vs Profit: We explored how standard business optimizations for "Customer Lifetime Value" can inadvertently create disparate impacts if demographic features are not audited for bias.

## Files
- `streaming_churn_rate.ipynb`: Main technical notebook (EDA, PyTorch model, and Audit)
- `RDS Final Project Audit Slides.pdf`: High-level summary of initial findings and ethical considerations
- `Group87_DraftReport.pdf`: Detailed analysis of the ADS framework and data ethics

## Next Steps
- Implement specific fairness constraints (like Equalized Odds) directly into the PyTorch loss function.
- Conduct a longitudinal study to see if retention discounts actually change long-term user behavior across different demographics.
