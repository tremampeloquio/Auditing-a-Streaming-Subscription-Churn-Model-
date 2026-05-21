# Music Streaming Churn & Fairness Audit
This project was developed as part of the DS-UA 202: Responsible Data Science course. My partner and I audited an algorithmic decision-support system (ADS) used by a fictional streaming service, PlaylistPro, to predict customer churn and target retention discounts.

The goal was to go beyond raw predictive accuracy and examine the ethical and business implications of using a PyTorch Multi-Layer Perceptron (MLP) to allocate direct financial interventions to 125,000 synthetic users.

## Project Questions
- Can a PyTorch-based Neural Network accurately predict which subscribers are at risk of churning?
- Does the model disproportionately exclude specific demographic groups (e.g., Gen X) or subscription tiers (e.g., Premium vs. Free) from receiving financial incentives?
- What are the downstream financial impacts and ethical trade-offs when a model prioritizes aggregate accuracy over equitable False Negative Rates (FNR)?

## Tools & Libraries
- Python
- PyTorch (Neural Network Modeling)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Preprocessing, Evaluation, & Feature Importance)

## Key Insights
- The Accuracy Illusion: While the baseline MLP achieved strong aggregate performance (84.1% accuracy), these top-level metrics masked severe demographic disparities.
- Subscription Tier Bias: The model severely under-identified churn among paid tiers. The False Negative Rate (FNR)—which represents a "missed churner" structurally denied a discount—was only 3.6% for Free users but jumped to 23.9% for Premium users.
- Intersectional Harm: Compounding biases created massive disparities. The most favored subgroup (Gen Z - Free) saw an FNR of just 1.7%, while the most harmed subgroup (Millennials - Premium) faced an FNR of 30.4%, representing a >17x disparity in predictive error.
- Proxy Discrimination: Permutation Feature Importance revealed the model relied heavily on behavioral proxies like customer_service_inquiries and weekly_hours. These features inadvertently penalized busy adults, functioning as proxies for age and income.

## Files
- `streaming_churn_rate.ipynb`: Main technical notebook (EDA, PyTorch model, and FairnessAudit)
- `PlaylistPro_Final_Audit_Deck.pdf`: High-level presentation deck summarizing our findings, the dataset's domain mismatch, and ethical considerations
- `Final_Report_RDS_Course_Project.pdf`: Detailed 9-page analysis of the ADS framework, fairness metrics, and deployment recommendations

## Next Steps
- Shift from single static data snapshots to collecting longitudinal engagement trajectories to better capture actual churn intent.
- Apply subgroup-aware calibration to equalize the False Negative Rate (FNR) across age groups and subscription tiers during pilot deployments.
- Reduce proxy dependence on subscription-tier variables through fairness-aware modeling, such as implementing revenue-aware loss functions.
- Implement regular re-auditing backed by intersectional fairness dashboards to monitor demographic drift over time.
