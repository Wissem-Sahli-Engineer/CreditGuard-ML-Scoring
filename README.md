# 🏦 CreditGuard: End-to-End Loan Default Prediction

## 📌 Overview

This project tackles the "Credit Scoring" problem: predicting whether a borrower will default on a loan. Using a high-dimensional dataset (34 features), we implement a full Data Science pipeline—from rigorous statistical hypothesis testing to advanced class imbalance handling.
![CreditGuard Banner](assets/banner.png)

## 📁 Dataset Structure

The dataset contains 34 variables including:
<table>
    <thead>
        <tr>
            <th><strong>Variable</strong></th>
            <th><strong>Description</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>ID</strong></td>
            <td>client loan application id</td>
        </tr>
        <tr>
            <td><strong>year</strong></td>
            <td>year of loan application</td>
        </tr>
        <tr>
            <td><strong>loan_limit</strong></td>
            <td>indicates whether the loan is conforming (cf) or non-conforming (ncf)</td>
        </tr>
        <tr>
            <td><strong>Gender</strong></td>
            <td>gender of the applicant (male, female, joint, sex not available)</td>
        </tr>
        <tr>
            <td><strong>approv_in_adv</strong></td>
            <td>indicates whether the loan was approved in advance (pre, nopre)</td>
        </tr>
        <tr>
            <td><strong>loan_type</strong></td>
            <td>type of loan (type1, type2, type3)</td>
        </tr>
        <tr>
            <td><strong>loan_purpose</strong></td>
            <td>purpose of the loan (p1, p2, p3, p4)</td>
        </tr>
        <tr>
            <td><strong>Credit_Worthiness</strong></td>
            <td>credit worthiness (l1, l2)</td>
        </tr>
        <tr>
            <td><strong>open_credit</strong></td>
            <td>indicates whether the applicant has any open credit accounts (opc, nopc)</td>
        </tr>
        <tr>
            <td><strong>business_or_commercial</strong></td>
            <td>indicates whether the loan is for business/commercial purposes (ob/c - business/commercial, nob/c -
                personal)</td>
        </tr>
        <tr>
            <td><strong>loan_amount</strong></td>
            <td>amount of money being borrowed</td>
        </tr>
        <tr>
            <td><strong>rate_of_interest</strong></td>
            <td>interest rate charged on the loan</td>
        </tr>
        <tr>
            <td><strong>Interest_rate_spread</strong></td>
            <td>difference between the interest rate on the loan and a benchmark interest rate</td>
        </tr>
        <tr>
            <td><strong>Upfront_charges</strong></td>
            <td>initial charges associated with securing the loan</td>
        </tr>
        <tr>
            <td><strong>term</strong></td>
            <td>duration of the loan in months</td>
        </tr>
        <tr>
            <td><strong>Neg_amortization</strong></td>
            <td>indicates whether the loan allows for negative amortization (neg_amm, not_neg)</td>
        </tr>
        <tr>
            <td><strong>interest_only</strong></td>
            <td>indicates whether the loan has an interest-only payment option (int_only, not_int)</td>
        </tr>
        <tr>
            <td><strong>lump_sum_payment</strong></td>
            <td>indicates if a lump sum payment is required at the end of the loan term (lpsm, not_lpsm)</td>
        </tr>
        <tr>
            <td><strong>property_value</strong></td>
            <td>value of the property being financed</td>
        </tr>
        <tr>
            <td><strong>construction_type</strong></td>
            <td>type of construction (sb - site built, mh - manufactured home)</td>
        </tr>
        <tr>
            <td><strong>occupancy_type</strong></td>
            <td>occupancy type (pr - primary residence, sr- secondary residence, ir - investment property)</td>
        </tr>
        <tr>
            <td><strong>Secured_by</strong></td>
            <td>specifies the type of collateral securing the loan (home, land)</td>
        </tr>
        <tr>
            <td><strong>total_units</strong></td>
            <td>number of units in the property being financed (1U, 2U, 3U, 4U)</td>
        </tr>
        <tr>
            <td><strong>income</strong></td>
            <td>applicant's annual income</td>
        </tr>
        <tr>
            <td><strong>credit_type</strong></td>
            <td>applicant's type of credit (CIB - credit information bureau , CRIF - CRIF credit information bureau,
                EXP - experian , EQUI - equifax)</td>
        </tr>
        <tr>
            <td><strong>Credit_Score</strong></td>
            <td>applicant's credit score</td>
        </tr>
        <tr>
            <td><strong>co-applicant_credit_type</strong></td>
            <td>co-applicant's type of credit (CIB - credit information bureau EXP - experian)</td>
        </tr>
        <tr>
            <td><strong>age</strong></td>
            <td>the age of the applicant.</td>
        </tr>
        <tr>
            <td><strong>submission_of_application</strong></td>
            <td>indicates how the application was submitted (to_inst - to institution, not_inst - not to
                institution)</td>
        </tr>
        <tr>
            <td><strong>LTV</strong></td>
            <td>loan-to-value ratio, calculated as the loan amount divided by the property value</td>
        </tr>
        <tr>
            <td><strong>Region</strong></td>
            <td>geographic region where the property is located (North, south, central, North-East)</td>
        </tr>
        <tr>
            <td><strong>Security_Type</strong></td>
            <td>type of security or collateral backing the loan (direct, indirect)</td>
        </tr>
        <tr>
            <td><strong>Status</strong></td>
            <td>indicates whether the loan has been defaulted (1) or not (0)</td>
        </tr>
        <tr>
            <td><strong>dtir1</strong></td>
            <td>debt-to-income ratio</td>
        </tr>
    </tbody>
</table>

---

### Loan Type Assignments

-`Type 1` (Conventional Loans): Characterized by higher loan amounts, lower LTV ratios, and stronger credit scores, making them a preferred option for well-qualified, lower-risk borrowers.

-`Type 2` (Government-Backed Loans): Typically involve lower loan amounts, higher LTV ratios, and moderate credit scores, indicating they are used by borrowers with smaller down payments who benefit from government-backed programs.

-`Type 3` (Non-Conventional Loans): Feature moderate loan amounts, the highest LTV ratios, and lower credit scores, often associated with higher-risk products such as jumbo loans or adjustable-rate mortgages.

### Loan Purpose Assignments

-`p1` (Home Purchase): Represents loans taken out for primary residences, often displaying moderate credit scores and higher LTV ratios.

-`p2` (Home Improvement): Smaller loan amounts used for property renovations, with lower LTV ratios suggesting homeowners are leveraging built-up equity.

-`p3` (Refinancing): Applies to homeowners replacing an existing mortgage, characterized by moderate loan amounts and lower LTV ratios, indicating financial stability.

-`p4` (Investment Property): Involves larger loan amounts and higher risk profiles, primarily financed through conventional loans due to restrictions on Government-backed funding for investment properties.

## 🚀 Installation & Setup

Follow these steps to get your local development environment running:

### 1. Clone the repository

Open your terminal and navigate to your Desktop:

```bash
cd ~/Desktop
git clone https://github.com/Wissem-Sahli-Engineer/CreditGuard-ML-Scoring.git
cd CreditGuard-ML-Scoring
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```
