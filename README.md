# 🛍️ B2C Customer Segmentation — Marketo + Salesforce Commerce Cloud

**Multi-method customer segmentation framework for a D2C premium skincare brand, unifying marketing engagement (Marketo) and transactional data (Salesforce Commerce Cloud) to build actionable customer personas with CLV estimation and campaign playbooks.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Business Problem

**Company:** GlowHaven — a D2C premium skincare brand selling across its own website, retail partners, and subscription boxes.

**Challenge:** Marketing runs Marketo for email/SMS campaigns and lifecycle automation. Salesforce Commerce Cloud manages transactions, order history, and customer profiles. These systems are siloed, leaving the team unable to answer critical questions:

- Who are the **highest-value customers** and what do they look like?
- Which customers are **at risk of churning** despite high past spend?
- How should **email, SMS, and push notification** strategies differ by segment?
- What is each customer's **estimated lifetime value** and how should that drive budget allocation?
- Which **product categories** resonate with which customer types?

This project unifies both data sources to build rich customer personas that drive personalized marketing at scale.

---

## Pipeline Overview

```
Step 1:  Environment Setup
Step 2:  Synthetic Data Generation
         ├── 2A: Salesforce Commerce Cloud — Customer Profiles (15,000 customers)
         ├── 2B: Salesforce Commerce Cloud — Order History (product catalog, transactions)
         └── 2C: Marketo Engagement Data (email, SMS, push, web, cart abandonment)
Step 3:  Data Validation & Quality Checks (join-key integrity, variable catalog)
Step 4:  Feature Engineering
         ├── 4A: Transaction Rollup (revenue, AOV, frequency, returns, profit)
         ├── 4B: Product Affinity Features (category spend %)
         ├── 4C: Marketo Engagement Rollup (engagement index, purchase intensity)
         ├── 4D: CLV Estimation (simplified lifetime value)
         ├── 4E: Unified Feature Matrix
         └── 4F: Feature Scaling
Step 5:  Exploratory Data Analysis (distributions, correlations, CLV by demographics)
Step 6:  Advanced Segmentation
         ├── 6A: RFM Analysis (Recency, Frequency, Monetary)
         ├── 6B: K-Means Clustering (Elbow + Silhouette optimization)
         ├── 6C: Gaussian Mixture Model (soft probabilistic clustering)
         ├── 6D: DBSCAN (VIP & anomaly detection)
         └── 6E: Hierarchical Clustering (Ward linkage + dendrogram)
Step 7:  Model Comparison (ARI, NMI cross-validation)
Step 8:  Persona Creation & Segment Profiling (radar charts, demographic cross-analysis)
Step 9:  Campaign Strategy & Export (playbook per persona, executive dashboard)
```

---

## Data

| Detail | Value |
|:---|:---|
| Customers | 15,000 D2C skincare customers (synthetic) |
| Orders | Multi-product transactions with returns, discounts, and profit |
| Product Catalog | 15 SKUs across Serums, Moisturizers, Cleansers, Sunscreen, Masks, Supplements |
| Marketo Activities | Email, SMS, Push, Web Visits, Cart Abandonment, Reviews, Referrals |
| Demographics | Age group, gender, channel preference, membership tier, region |

### Feature Matrix (Clustering Inputs)

| Feature | Source | Category |
|:---|:---|:---|
| Total Orders, Revenue, AOV | Commerce Cloud | Purchase Behavior |
| Order Frequency, Unique Products | Commerce Cloud | Purchase Behavior |
| Return Rate, Days Since Last Order | Commerce Cloud | Purchase Behavior |
| Engagement Index | Marketo | Engagement |
| Purchase Intensity | Marketo | Engagement |
| Category Diversity (Shannon Entropy) | Commerce Cloud | Product Affinity |
| Estimated CLV | Computed | Lifetime Value |

---

## Key Techniques

### Customer Lifetime Value (CLV) Estimation
Simplified CLV model: `Avg Monthly Revenue × Expected Lifetime Months × Profit Margin`. Provides a forward-looking value metric that drives segment prioritization.

### RFM Analysis (B2C-Adapted)
Recency = days since last order, Frequency = total orders, Monetary = total revenue. Scored 1–5 per dimension with B2C-specific segment labels (VIP Loyalists, Promising Newcomers, Cart Abandoners, etc.).

### K-Means Clustering
Optimal K selected via Elbow (KneeLocator) and Silhouette analysis across K=2–11. Features standardized with StandardScaler. PCA projection for visualization.

### Gaussian Mixture Model (GMM)
Soft clustering with probability of membership. Model selection via BIC minimization. Captures overlapping customer segments better than hard partitioning.

### DBSCAN
Density-based method for **VIP and outlier detection** — identifies ultra-high-value customers that don't fit standard clusters, plus data quality outliers.

### Hierarchical Clustering
Ward linkage with dendrogram for visual validation of cluster count.

### Consensus Validation
Cross-method comparison using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) to confirm segment stability across algorithms.

---

## Results Summary

### Customer Personas

| Persona | Profile | Campaign Strategy |
|:---|:---|:---|
| 💎 VIP Skincare Enthusiasts | Highest CLV, full routine buyers, highly engaged | Exclusive early access, VIP-only launches, personalized routines |
| 🌟 Loyal Routine Builders | Regular purchasers, growing baskets, moderate CLV | Loyalty rewards, bundle offers, routine completion nudges |
| 🔄 Bargain Hunters | Discount-driven, high return rate, price-sensitive | Flash sales, clearance alerts, BOGO offers |
| 🌱 Promising Newcomers | Recent first purchase, exploring products | Welcome series, starter kits, educational content |
| 😴 Lapsed Customers | High past spend, long dormancy, declining engagement | Win-back campaigns, "we miss you" offers, new product alerts |

### Visualizations

The notebook produces 20+ publication-quality charts including:
- Feature distributions and correlation heatmaps
- CLV distribution by age group, gender, and membership tier
- RFM segment bars and scatter plots
- Elbow curve, Silhouette, Calinski-Harabasz, and Davies-Bouldin plots
- PCA cluster projections for all methods
- DBSCAN VIP/anomaly detection
- Dendrogram for hierarchical clustering
- Radar charts comparing personas across key metrics
- Persona × Demographics cross-analysis (age, channel stacked bars)
- Executive dashboard (revenue, CLV, engagement, orders by persona)

---

## Project Structure

```
b2c-customer-segmentation/
│
├── data/
│   ├── raw/                          # Raw synthetic data
│   └── processed/                    # Unified feature matrix & segmented output
│
├── notebooks/
│   └── b2c_customer_segmentation.ipynb   # Full 9-step pipeline
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # Synthetic Commerce Cloud + Marketo data
│   ├── feature_engineering.py        # Transaction rollups, CLV, engagement, product affinity
│   ├── segmentation.py               # RFM, K-Means, GMM, DBSCAN, Hierarchical
│   └── persona_builder.py            # Persona naming, profiling, campaign playbooks
│
├── outputs/                          # Saved charts and exported customer CSVs
├── requirements.txt
├── README.md
└── LICENSE
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/raianupam171126/b2c-customer-segmentation.git
cd b2c-customer-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/b2c_customer_segmentation.ipynb
```

**Google Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raianupam171126/b2c-customer-segmentation/blob/main/notebooks/b2c_customer_segmentation.ipynb)

---

## Tech Stack

- **Python 3.10** — pandas, NumPy
- **Clustering** — scikit-learn (KMeans, GaussianMixture, AgglomerativeClustering, DBSCAN)
- **Dimensionality Reduction** — PCA
- **Evaluation** — Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI
- **Visualization** — Matplotlib, Seaborn
- **Utilities** — scipy (linkage/dendrogram), kneed (elbow detection)

---

## B2B vs B2C — How This Differs

This project is the **B2C companion** to the [B2B Customer Segmentation](https://github.com/raianupam171126/b2b-customer-segmentation) project. Key differences:

| Dimension | B2B | B2C |
|:---|:---|:---|
| Entity | Account (company) | Individual customer |
| Volume | 2,500 accounts | 15,000 customers |
| Revenue Signal | Pipeline value, deal size | Order history, AOV, CLV |
| Engagement | Marketo lead scoring | Email/SMS/Push/Web/Cart abandonment |
| Segmentation Goal | Sales tier prioritization | Persona-driven personalization |
| Activation | ABM campaigns, CSM plays | Email flows, SMS, push, loyalty programs |

---

## Limitations & Future Work

- **Synthetic data** — real implementation requires Commerce Cloud OCAPI exports and Marketo REST API pulls
- **Simplified CLV** — production CLV would use BG/NBD + Gamma-Gamma models (lifetimes library) or probabilistic approaches
- **No real-time scoring** — segments are static snapshots; a production system would use Marketo Smart Lists or SFCC customer groups for real-time activation
- **No A/B testing framework** — campaign playbooks are strategic recommendations, not experimentally validated
- **Product recommendation engine** — natural extension using collaborative filtering on product affinity features

---

## References

- Fader, P. & Hardie, B. — *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*
- Fader, P., Hardie, B., Lee, K. — *"Counting Your Customers" the Easy Way: BG/NBD Model*
- scikit-learn documentation — Clustering, Mixture Models, Evaluation Metrics
- Salesforce Commerce Cloud OCAPI & Marketo REST API documentation

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR-PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/raianupam171126)
