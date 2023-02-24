import pandas as pd

###################################################################################################
#                                     Task 2                                                      #
###################################################################################################
gym_10_min_df = pd.read_csv('hietaniemi-gym-data.csv')
from datetime import datetime



def number_of_rows(x):
    return x.shape[0]

def records_timerange(df):
    # specify the date range to test against
    
    start_date = '2020-04-24'
    start_date=datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = '2021-05-11'
    end_date=datetime.strptime(end_date, '%Y-%m-%d').date()
    df['time']=pd.to_datetime(df['time'])
    # check whether all dates in the DataFrame fall within the specified date range
    df['date']=df['time'].apply(lambda x: x.date())
    records_in_range = df.date.between(start_date, end_date).all()
    return records_in_range
    


# Test there are more than 50,000 rows in the dataset
def test_number_of_rows():
    assert number_of_rows(gym_10_min_df) > 50000
    
#Test there are records from between 2020-04-24 and 2021-05-11
def test_dates_in_range():
    assert records_timerange(gym_10_min_df) == True
    
# All values in the numerical columns are positive
def test_positive_19():
    assert (gym_10_min_df['19']>0).all() == True

def test_positive_20():
    assert (gym_10_min_df['20']>0).all() == True
    
def test_positive_21():
    assert (gym_10_min_df['21']>0).all() == True
    
def test_positive_22():
    assert (gym_10_min_df['22']>0).all() == True
    
def test_positive_23():
    assert (gym_10_min_df['23']>0).all() == True
    
def test_positive_24():
    assert (gym_10_min_df['24']>0).all() == True
    
def test_positive_25():
    assert (gym_10_min_df['25']>0).all() == True
    
def test_positive_26():
    assert (gym_10_min_df['26']>0).all() == True

    

    
