import numpy as np
import pandas as pd
import re
import time
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import json

def get_external_id_from_bam_link(bam_link):
    return re.search('external_id=(.*)&ENC=', bam_link).group(1)


def convert_datetime_str_to_unix(datetime_str):
    shortened_datetime_str = re.search(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6})', datetime_str).group(1)
    unix = int(datetime.strptime(shortened_datetime_str, '%Y-%m-%dT%H:%M:%S.%f').timestamp())
    return unix

def clean_user_data(FC2022_user_data, start_unix):
    """
    Step 1: only keep relevant columns
    Step 2: only keep users who participated in the study after the study start 2022-11-09 12:00:00 
            and finished all 9 weeks
    """
    # only keep useful columns
    FC2022_user_data = FC2022_user_data.loc[:,('id', 'link', 'type', 'verified', 'survey', 'register_time', 
                                        'verify_time', 'data_email_sent', 'bam_reminder_sent','post_survey_id', 
                                        'post_survey_link_de', 'post_survey_link_en','post_survey_sent', 
                                        'analysis_report_sent')]

    # convert time string to datetime
    FC2022_user_data.loc[:,'register_time'] = FC2022_user_data['register_time'].apply(convert_datetime_str_to_unix)

    FC2022_user_data.loc[:,"external_id"] = FC2022_user_data.link.map(get_external_id_from_bam_link)

    # verify_time can be NaN, it is a problem, if the user belongs to the experiment group
    # FC2022_user_data['verify_time'] = FC2022_user_data['verify_time'].map(lambda x: datetime.fromisoformat(x[:19]))
    nine_weeks_ago = int(time.time()) - 5443200


    # Users should join the study after the study start and finish 9 weeks
    FC2022_user_data = FC2022_user_data[(FC2022_user_data.register_time > start_unix)
                                        & (FC2022_user_data.register_time < nine_weeks_ago)]
    return FC2022_user_data


def remove_inactive_experimental_users(FC2022_user_data):
    return FC2022_user_data[~((FC2022_user_data.verified == 0) & (FC2022_user_data.type == "Experiment"))]

def get_age_from_birthdate(birthdate):
    unix_birthdate = birthdate/1000 #convert birthdate to unix
    birthdate = datetime.fromtimestamp(unix_birthdate)
    # calculate age in years
    age = (datetime.now() - birthdate) /timedelta(days=365.2425)
    return age

def get_valid_onboarding_survey(df):
    # get unique surveys
    valid_df = df.loc[:,("external_id", "survey", 'register_time', 'verify_time')].drop_duplicates()
    surveys = pd.DataFrame(list(valid_df.survey.map(json.loads)))
    surveys.loc[:,"age"] = surveys.birthdate.map(get_age_from_birthdate)
    surveys.loc[:,"household_size"] = surveys['loyaltyShareAdults'] + surveys['loyaltyShareTeens'] + surveys['loyaltyShareKids']
    # convert bmi str to num
    surveys.loc[:,'bmi'] = surveys['bmi'].apply(float)
    # bmi_mean = 
    return surveys

def get_last_data_update(df):
    
    last_item_timestamp = df.groupby('bamId')['basketTimestamp'].max()

    # Create a new DataFrame with unique user_id and last_item_timestamp
    unique_users_df = pd.DataFrame({'bamId': last_item_timestamp.index, 'basketTimestamp': last_item_timestamp.values})

    return unique_users_df

def get_receipts_of_certain_period(df, unix_diff_start, unix_diff_end):
    time_diff = df.basketTimestamp - df.register_time
    selected_df =  df[(time_diff>= unix_diff_start) & (time_diff < unix_diff_end)]
    return selected_df

def get_unixtime_from_dt64(dt64):
    return dt64.astype('datetime64[s]').astype('int')

def calculate_ofcom(df, ofcom_field):
    #calculate the FSA-NPS DI
    df = df.loc[:,('bamId', 'actual_energy_kcal', "type", 'OfComValue', 'ofComNSalt', 
                      'ofComNSaturatedFat','ofComNSugar', 'ofComPDietaryFiber', 'ofComPFVPN', 'ofComPProtein')]
    df.loc[:,'energy*ofcom'] = (df['actual_energy_kcal'].values) * (df[ofcom_field].values)
    numerator = df.groupby('bamId')['energy*ofcom'].sum().reset_index()
    denominator = df.groupby('bamId')['actual_energy_kcal'].sum().reset_index()
    fsa_nps_di = numerator.merge(denominator, how='outer',on = 'bamId')
    #keep the user type
    unique_users = df.loc[:,("bamId", "type")].drop_duplicates()
    fsa_nps_di = fsa_nps_di.merge(unique_users, how = "left", on = "bamId")
    fsa_nps_di.loc[:,'fsa_nps_di'] = fsa_nps_di['energy*ofcom']/fsa_nps_di['actual_energy_kcal']
    return fsa_nps_di

def check_normality(df, column):
    """
    if n<= 50:
        shapiro test
    else:
        D'Agostino's K^2 test
    """
    num_users = df[~df[column].isna()].shape[0] 
    if num_users <= 50:
       normality_test = stats.shapiro(df[column])
    else:
        normality_test = stats.normaltest(df[column])
    if normality_test.pvalue <= 0.05:
        return num_users, True
    else:
        return num_users, False
    
def get_median_iqr(df, column):
    # remove nan numbers first
    data = df[~df[column].isna()][column]
    median = data.median()
    q1 = np.percentile(data, 25, interpolation='midpoint')
    q3 = np.percentile(data, 75, interpolation='midpoint')
    iqr = q3 - q1
    return median.round(1), iqr.round(1)

def get_mean_std(df, column):
    data = df[column]
    return data.mean().round(1), data.std().round(1)

def get_column_stats(df, column):
    n, is_normal = check_normality(df, column)
    if is_normal:
        mean, std = get_mean_std(df, column)
        return n, is_normal, mean, std
    else:
        median, iqr = get_median_iqr(df, column)
        return n, is_normal, median, iqr
    
def non_para_comparison(dfs, t1, t2, column, between_group = True):
    
    position_dict = {"T0 overall": 0,
                     "T0 control": 1,
                     "T0 experiment": 2,
                     "T1 overall": 3,
                     "T1 control": 4,
                     "T1 experiment": 5,
                     "T2 overall": 6,
                     "T2 control": 7,
                     "T2 experiment": 8,
                     "T3 overall": 9,
                     "T3 control": 10,
                     "T3 experiment": 11,
                     }
    if between_group:
        # It cannot contain NaNs
        t1_data = dfs[position_dict[t1]][column].dropna()
        t2_data = dfs[position_dict[t2]][column].dropna()
        statistic, p_value = stats.mannwhitneyu(t1_data, t2_data)
    else:
        # get the relevant data
        relevant_t1_data = dfs[position_dict[t1]].loc[:,("bamId", column)]
        relevant_t2_data = dfs[position_dict[t2]].loc[:,("bamId", column)]
        
        # merge the two data sources, since it is paired comparison
        relevant_t1_data = relevant_t1_data.rename(columns = {column: column + "_t1"})
        relevant_t2_data = relevant_t2_data.rename(columns = {column: column + "_t2"})
        relevant_data = relevant_t1_data.merge(relevant_t2_data, how = "inner", on = "bamId")

        statistic, p_value = stats.wilcoxon(relevant_data[column + "_t1"], relevant_data[column + "_t2"])

    significant = False
    if p_value <= 0.05:
        significant = True 
    return statistic, p_value, significant
        # start wil


def get_ofcom_group_specific_data(df, ofcom_field):
    # T0: week -3 to week 0
    # T1: week 0 to week 3
    # T2: week 4 to week 6
    # T3: week 7 to week 9

    # in the order of T0 -> T4, all->control->experiment
    #T0 overall - T0 control - T0 exp
    #T1 overall - T1 control - T1 exp
    #T2 overall - T2 control - T2 exp
    #T3 overall - T3 control - T3 exp


    unix_diff_ls = [-1814400, 0 ,1814400, 3628800, 5443200]


    results = []
    dfs = [] 

    for i in range(len(unix_diff_ls[:-1])):
        ti_receipts = get_receipts_of_certain_period(df, unix_diff_ls[i], unix_diff_ls[i+1])
        # get ofcom fields
        ti_receipts_with_ofcom = calculate_ofcom(ti_receipts, ofcom_field)
        ti_control_receipts_with_ofcom = ti_receipts_with_ofcom[ti_receipts_with_ofcom.type == "Control"]
        ti_experiment_receipts_with_ofcom = ti_receipts_with_ofcom[ti_receipts_with_ofcom.type == "Experiment"]
        # get reports
        results.append(get_column_stats(ti_receipts_with_ofcom, "fsa_nps_di"))
        results.append(get_column_stats(ti_control_receipts_with_ofcom, "fsa_nps_di"))
        results.append(get_column_stats(ti_experiment_receipts_with_ofcom, "fsa_nps_di"))
        # collect raw ofcom dfs
        dfs.append(ti_receipts_with_ofcom)
        dfs.append(ti_control_receipts_with_ofcom)
        dfs.append(ti_experiment_receipts_with_ofcom)


    descriptive_stats = pd.DataFrame(results, columns =['num_users', 'is_normal', 'median/mean', "iqr/std"])
    return descriptive_stats, dfs
    
      


def ofcom_comparisons(dfs):
    within_group_comparison_todos = [("T0 control", "T1 control"),
                                    ("T0 control", "T2 control"),
                                    ("T0 control", "T3 control"),
                                    ("T0 experiment", "T1 experiment"),
                                    ("T0 experiment", "T2 experiment"),
                                    ("T0 experiment", "T3 experiment")
                                    ]

    between_group_comparison_todos = [("T0 control", "T0 experiment"),
                                    ("T1 control", "T1 experiment"),
                                    ("T2 control", "T3 experiment"),
                                    ("T3 control", "T3 experiment"),
                                    ]


    significant_diff = []
    insignificant_diff = []
    # within group comparisons
    for (t1, t2) in within_group_comparison_todos:
        t_statistic, p_value, is_significant = non_para_comparison(dfs, t1, t2, "fsa_nps_di", between_group = False)
        if is_significant:
            significant_diff.append((t1, t2, is_significant, p_value))
        else:
            insignificant_diff.append((t1, t2, is_significant, p_value))

    # between group comparisons
    for (t1, t2) in between_group_comparison_todos:
        t_statistic, p_value, is_significant = non_para_comparison(dfs, t1, t2, "fsa_nps_di", between_group = True)
        if is_significant:
            significant_diff.append((t1, t2, is_significant, p_value))
        else:
            insignificant_diff.append((t1, t2, is_significant, p_value))
    
    return significant_diff, insignificant_diff

def get_bmi_report(df):
    bmi_bins = [0,18.5, 25, 30, float("inf")]
    df.loc[:,'binned_bmi'] = pd.cut(df.loc[:,'bmi'], bmi_bins)


    # get the value counts and percentages
    value_counts = df.loc[:,'binned_bmi'].value_counts()
    percentages = df.loc[:,'binned_bmi'].value_counts(normalize=True) * 100

    bmi_df = pd.concat([value_counts, percentages], axis=1, keys=['binned_bmi_count', 'binned_bmi(%)'])

    return bmi_df


# def get_value_count_percentage(df, column):
#     value_counts = df[column].value_counts()
#     percentages = df[column].value_counts(normalize=True) * 100
#     return value_counts, percentages

def get_demographic_summary(df):

    df = df.copy()

    median_iqr_cols = ['bmi', 'percShoppingMigros','percShoppingCoop', 'usageMigros', 'usageCoop', 'percFruits',
                       'percVegetables', 'percProteinFoods', 'percProcessedFoods', 'percCarbs','percOils', 
                       'percBeverages', 'loyaltyShareAdults', 'loyaltyShareKids','loyaltyShareTeens', 'age', 
                       'household_size']

    value_count_cols = ['loyaltyCards','gender','fromSanitas', 'historyApps', 'currentlyUsingOtherApps',
                        'currentOtherApps','disorders', 'allergiesAndAbstentions', 'meat', 'otherDiseases',
                        'activityLevelAtWork', 'sports', 'education', 'householdIncome']

    median_iqr_summary = pd.DataFrame(index=median_iqr_cols, columns=['count', 'median', 'IQR'])

    for col in median_iqr_cols:
        median_iqr_summary.loc[col, 'count'] = df.loc[:,col].count()
        median_iqr_summary.loc[col, 'median'] = df.loc[:,col].median().round(1)
        median_iqr_summary.loc[col, 'IQR'] = (df.loc[:,col].quantile(0.75) - df[col].quantile(0.25)).round(1)

    value_count_results = {}
    for col in value_count_cols:
        counts = df.loc[:,col].value_counts()
        # Calculate the percentages of each category
        percentages = df.loc[:,col].value_counts(normalize=True) * 100
        
        value_count_results[col] = pd.concat([counts, percentages], axis=1, keys=[f'{col}_count', f'{col} (%)'])
        
    value_count_results["binned_bmi"] = get_bmi_report(df)
    return median_iqr_summary, value_count_results



def create_last_purchase_date_plot(df):
    # Convert the Unix timestamp column to a datetime object
    new_df = get_last_data_update(df)
    new_df['date'] = pd.to_datetime(new_df['basketTimestamp'], unit='s')
    # Create a distribution plot of the date column
    ax = sns.histplot(data=new_df, x='date', bins = 40)

    # Set the x-axis label
    ax.set_xlabel('Date')

    # Set the x-/y-axis label
    ax.set_xlabel('Last purchase date')
    ax.set_ylabel('Number of participants')

    start_date = datetime(2021, 4, 1)
    end_date = datetime(2023, 6, 1)
    ax.set_xlim(start_date,end_date)

    # Rotate the x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45)

    # Return the axis object
    return ax

def get_valid_users_with_last_xd_data(df, days):
    last_x_days_data = int(time.time()) - days * 86400
    valid_df = df[df.basketTimestamp >= last_x_days_data]

    print(df.shape, valid_df.shape)
    total_users = df.userHash.unique().shape[0]
    valid_user = valid_df.userHash.unique().shape[0]

    return total_users, valid_user