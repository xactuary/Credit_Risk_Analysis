# Credit Card Risk Analysis using Machine Learning
## SUMMARY
Using a credit card application credit risk dataset provided by the Lending Club, a peer-to-peer lending company, I have used various unbalanced Machine Learning techniques to determine
a model for predicting whether or not an applicant will be a good or bad credit risk.

### Data:
The data set includes 68,817 rows (Observations) and 86 columns (Variables).

loan_amnt                     float64
int_rate                      float64
installment                   float64
home_ownership                 object
annual_inc                    float64
verification_status            object
issue_d                        object
loan_status                    object
pymnt_plan                     object
dti                           float64
delinq_2yrs                   float64
inq_last_6mths                float64
open_acc                      float64
pub_rec                       float64
revol_bal                     float64
total_acc                     float64
initial_list_status            object
out_prncp                     float64
out_prncp_inv                 float64
total_pymnt                   float64
total_pymnt_inv               float64
total_rec_prncp               float64
total_rec_int                 float64
total_rec_late_fee            float64
recoveries                    float64
collection_recovery_fee       float64
last_pymnt_amnt               float64
next_pymnt_d                   object
collections_12_mths_ex_med    float64
policy_code                   float64
application_type               object
acc_now_delinq                float64
tot_coll_amt                  float64
tot_cur_bal                   float64
open_acc_6m                   float64
open_act_il                   float64
open_il_12m                   float64
open_il_24m                   float64
mths_since_rcnt_il            float64
total_bal_il                  float64
il_util                       float64
open_rv_12m                   float64
open_rv_24m                   float64
max_bal_bc                    float64
all_util                      float64
total_rev_hi_lim              float64
inq_fi                        float64
total_cu_tl                   float64
inq_last_12m                  float64
acc_open_past_24mths          float64
avg_cur_bal                   float64
bc_open_to_buy                float64
bc_util                       float64
chargeoff_within_12_mths      float64
delinq_amnt                   float64
mo_sin_old_il_acct            float64
mo_sin_old_rev_tl_op          float64
mo_sin_rcnt_rev_tl_op         float64
mo_sin_rcnt_tl                float64
mort_acc                      float64
mths_since_recent_bc          float64
mths_since_recent_inq         float64
num_accts_ever_120_pd         float64
num_actv_bc_tl                float64
num_actv_rev_tl               float64
num_bc_sats                   float64
num_bc_tl                     float64
num_il_tl                     float64
num_op_rev_tl                 float64
num_rev_accts                 float64
num_rev_tl_bal_gt_0           float64
num_sats                      float64
num_tl_120dpd_2m              float64
num_tl_30dpd                  float64
num_tl_90g_dpd_24m            float64
num_tl_op_past_12m            float64
pct_tl_nvr_dlq                float64
percent_bc_gt_75              float64
pub_rec_bankruptcies          float64
tax_liens                     float64
tot_hi_cred_lim               float64
total_bal_ex_mort             float64
total_bc_limit                float64
total_il_high_credit_limit    float64
hardship_flag                  object
debt_settlement_flag           object

Our target variable for prediction is "Loan_Status"

There are several variables that need to be converted to numeric in order to use them in the Machine learning program.  These are the ones that are highlighted as "objects" in the variable list.



