# brca-precision-onco-ai

An interpretable machine learning platform for breast cancer subtype classification using multi-omics data. The system integrates predictive modeling with biomarker identification and pathway-level analysis to support precision oncology research.

## Overview

This project presents an end-to-end workflow for analyzing multi-omics datasets in breast cancer (TCGA BRCA). It combines supervised learning with post-hoc interpretability methods to identify biologically meaningful patterns across cancer subtypes.

The platform is implemented as an interactive Streamlit application, enabling data upload, prediction, and downstream biological interpretation.

<img width="1536" height="1024" alt="ChatGPT Image Apr 24, 2026, 08_37_47 PM" src="https://github.com/user-attachments/assets/9063e238-5dd1-4313-99ee-b7709c9f8eb7" />


## Live app link
Streamlit cloud : https://brca-precision-onco-ai-gawghxw2lhsw7pzs47ceze.streamlit.app/

## Dataset: The Cancer Genome Atlas (TCGA)
Reference: Ciriello G., Gatza M.L., Beck A.H., et al. (2015). Comprehensive Molecular Portraits of Invasive Lobular Breast Cancer. Cell, 163(2), 506–519. https://doi.org/10.1016/j.cell.2015.09.033


## Key Features

- Multi-omics breast cancer subtype prediction using XGBoost
- Feature importance-based global biomarker identification
- SHAP-based subtype-specific biomarker discovery
- Pathway enrichment analysis using KEGG (via Enrichr)
- Hub gene prioritization through aggregated importance scoring
- Input validation and preprocessing pipeline
- Interactive web interface for analysis

## Model Performance

| Metric        | Value  |
|--------------|--------|
| Accuracy     | 0.922  |
| Macro F1     | 0.861  |
| Weighted F1  | 0.912  |

The model demonstrates strong overall performance with balanced classification across subtypes.

## Project Structure

brca-precision-onco-ai/

├── app.py  
├── requirements.txt  
├── README.md  

├── models/  
│   ├── multiomics_xgb_model.pkl  
│   ├── features.pkl  
│   ├── label_encoder.pkl  
│   ├── scaler.pkl (optional)  

├── notebooks/  
└── data/  

## Installation

Clone the repository and install dependencies:

git clone https://github.com/kirankumar88/brca-precision-onco-ai.git  
cd brca-precision-onco-ai  
pip install -r requirements.txt  

## Running the Application

streamlit run app.py  

The application will launch locally in your browser.

## Input Requirements

- CSV file with feature columns matching the trained model  
- Numeric values only  
- Missing values are automatically handled during preprocessing  

A sample dataset can be downloaded from the application sidebar.

## Methodology

- Model: XGBoost multiclass classifier  
- Data: TCGA BRCA multi-omics features  
- Preprocessing: Median imputation and optional scaling  

Interpretation:
- Global feature importance  
- SHAP-based local explanations  

Biological Analysis:
- Gene extraction from feature space  
- Pathway enrichment using KEGG  
- Hub gene ranking via aggregated importance  

## Use Cases

- Cancer subtype classification  
- Biomarker discovery  
- Multi-omics data interpretation  
- Translational oncology research  
- AI-assisted biological hypothesis generation  

## Limitations

- Model trained on a specific dataset (TCGA BRCA)  
- Generalization to external cohorts requires validation  
- Pathway enrichment depends on selected gene subsets  

## Future Improvements

- Integration of additional omics layers  
- External validation datasets  
- Survival analysis and clinical outcome prediction  
- Network-based biomarker analysis  
- Deployment with scalable backend  

## Author

Kiran Kumar  
Bioinformatics, Machine Learning, Multi-Omics Research  

## License

This project is intended for research and educational purposes.
