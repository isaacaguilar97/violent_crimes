import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib
import statistics as stat

# Load Data
df = pd.read_csv('final_table.csv')
# Load stat.models
stat.models_path = ['Primary Type_BURGLARY_stat.model.joblib.gz',
       'Primary Type_CRIM SEXUAL ASSAULT_stat.model.joblib.gz',
       'Primary Type_ROBBERY_stat.model.joblib.gz', 'Primary Type_INTIMIDATION_stat.model.joblib.gz',
       'Primary Type_HOMICIDE_stat.model.joblib.gz', 'Primary Type_KIDNAPPING_stat.model.joblib.gz',
       'Primary Type_HUMAN TRAFFICKING_stat.model.joblib.gz', 'Beat_1111_stat.model.joblib.gz',
       'Beat_1112_stat.model.joblib.gz', 'Beat_1113_stat.model.joblib.gz', 'Beat_1114_stat.model.joblib.gz',
       'Beat_1115_stat.model.joblib.gz', 'Beat_1121_stat.model.joblib.gz', 'Beat_1122_stat.model.joblib.gz',
       'Beat_1123_stat.model.joblib.gz', 'Beat_1124_stat.model.joblib.gz', 'Beat_1125_stat.model.joblib.gz',
       'Beat_1131_stat.model.joblib.gz', 'Beat_1132_stat.model.joblib.gz', 'Beat_1133_stat.model.joblib.gz',
       'Beat_1134_stat.model.joblib.gz', 'Beat_1135_stat.model.joblib.gz', 'n_crimes_stat.model.joblib.gz']
trained_stat.models = []
for file in stat.models_path:
    stat.model = joblib.load(file)
    name = file.replace('.joblib.gz', '')
    trained_stat.models.append((name, stat.model))

with st.sidebar:
    # App Title
    st.title('Significant Factors')

    # Description
    st.markdown('The following filters appear to be significant at influencing the number of Violent Crimes per hour in the District 11 of Chicago. Play arround with them to see how they influence Violent Crimes')

    # Features
    employ = st.slider('Unemployement', min_value=min(df['Unemployment']), max_value=max(df['Unemployment']), value=stat.mode(df['Unemployment']), step=0.1)
    temp = st.slider('Temperature', min_value=min(df['temperature']), max_value=max(df['temperature']), value=stat.mode(df['temperature']), step=0.1)
    humid = st.slider('Humidity', min_value=min(df['humidity']), max_value=max(df['humidity']), value=stat.mode(df['humidity']), step=0.1)
    feels = st.slider('Temp Feels Like', min_value=min(df['feels_like']), max_value=max(df['feels_like']), value=stat.mode(df['feels_like']), step=0.1)
    rn = st.slider('Rain', min_value=min(df['rain']), max_value=max(df['rain']), value=stat.mode(df['rain']), step=0.01)
    s_fall = st.slider('Snow Fall', min_value=min(df['snowfall']), max_value=max(df['snowfall']), value=stat.mode(df['snowfall']), step=0.1)
    s_dep = st.slider('Snow Depth', min_value=min(df['snow_depth']), max_value=max(df['snow_depth']), value=stat.mode(df['snow_depth']), step=0.1)
    cloud = st.slider('Cloud Cover', min_value=min(df['cloud_cover']), max_value=max(df['cloud_cover']), value=stat.mode(df['cloud_cover']), step=0.1)
    w_speed = st.slider('Wind Speed', min_value=min(df['wind_speed']), max_value=max(df['wind_speed']), value=stat.mode(df['wind_speed']), step=0.1)
    w_gusts = st.slider('Wind Gusts', min_value=min(df['wind_gusts']), max_value=max(df['wind_gusts']), value=stat.mode(df['wind_gusts']), step=0.1)
    wave_rad = st.slider('Shortwave Radiation', min_value=min(df['shortwave_radiation']), max_value=max(df['shortwave_radiation']), value=stat.mode(df['shortwave_radiation']), step=0.1)
    dir_rad = st.slider('Direct Radiation', min_value=min(df['direct_radiation']), max_value=max(df['direct_radiation']), value=stat.mode(df['direct_radiation']), step=0.1)
    dew = st.slider('Water Droplets', min_value=min(df['dew']), max_value=max(df['dew']), value=stat.mode(df['dew']), step=0.01)
    s_press = st.slider('Sea Level Pressure', min_value=min(df['sealevelpressure']), max_value=max(df['sealevelpressure']), value=stat.mode(df['sealevelpressure']), step=0.1)
    vis = st.slider('Visibility', min_value=min(df['visibility']), max_value=max(df['visibility']), value=stat.mode(df['visibility']), step=0.1)


    # Warning message
    # st.info("Be aware that results come from a sample of NBA players from season 2023")

    # Abbreviations
    # with st.expander("Abbreviation Index"):
    #     st.table(abb)

# Header 1
st.header('Chicago Crimes Distric 11')

st.markdown('When is Crime Happening?')

# 3 months (New preds)
calendar = pd.date_range(start='2024-01-01', end='2024-03-31', freq='H')
full_moon_dates = [
    "2024-01-25",
    "2024-02-24",
    "2024-03-25",
    "2024-04-23",
    "2024-05-23",
    "2024-06-21",
    "2024-07-21",
    "2024-08-19",
    "2024-09-17",
    "2024-10-17",
    "2024-11-15",
    "2024-12-15"
]

holiday_dates = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Washington's Birthday (Presidents Day)
    "2024-05-27",  # Memorial Day
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-10-14",  # Columbus Day
    "2024-11-11",  # Veterans Day
    "2024-11-28",  # Thanksgiving Day
    "2024-12-25"   # Christmas Day
]

full_calendar = pd.DataFrame({
    'Date': calendar,
    'Year': calendar.year,
    'month': calendar.strftime('%b'),
    'DayOfWeek': calendar.strftime('%a'),  # Monday=0, Sunday=6
    'hour': calendar.hour.astype('category'),
    'Week' : [date.isocalendar()[1] for date in calendar], 
    'is_holiday': [1 if str(date)[:10] in holiday_dates else 0 for date in calendar],  # Check if date is a holiday
    'is_full_moon': [1 if str(date)[:10] in full_moon_dates else 0 for date in calendar]  # Check if date is a full moon
})

sunrise = [7]*31 + [7]*4 + [6]*25 + [6]*9 + [7]*6 + [6]*16
sunset = [16]*27 + [17]*4 + [17]*29 + [17]*9 + [18]*7 + [19]*15
full_calendar['is_day'] = full_calendar['hour'].apply(lambda x: 1 if sunrise[x] <= x < sunset[x] else 0).astype('category')
full_calendar['is_full_moon'] = full_calendar['is_full_moon'].astype('category')
full_calendar['is_holiday'] = full_calendar['is_holiday'].astype('category')

week_concatenated = (full_calendar.groupby('Week')['Date'].agg(['first', 'last']).reset_index())
week_concatenated['last'] = week_concatenated['last'].dt.date # Remove the hours from the 'last' column
week_concatenated['weeks'] = week_concatenated['first'].astype(str) + ' - ' + week_concatenated['last'].astype(str)
full_calendar = full_calendar.merge(week_concatenated[['Week', 'weeks']], on='Week', how='left')

# Filters
col1, col2 = st.columns(2)
wks = col1.selectbox("Select Week", full_calendar['weeks'].unique().tolist())
v_type = col2.selectbox("Violent Crime Type", ['n_crimes','CRIM SEXUAL ASSAULT', 'ASSAULT', 'BATTERY', 'ROBBERY', 'INTIMIDATION', 'HOMICIDE', 'KIDNAPPING', 'HUMAN TRAFFICKING'])

# Filter data
new_obs_t = full_calendar[full_calendar['weeks'] == wks][['Year','month', 'DayOfWeek', 'hour', 'is_holiday', 'is_full_moon', 'is_day']]
new_obs = new_obs_t
new_obs['moonphase'] = .5
new_obs['temperature'] = temp
new_obs['humidity'] = humid
new_obs['feels_like'] = feels
new_obs['rain'] = rn
new_obs['snowfall'] = s_fall
new_obs['snow_depth'] = s_dep
new_obs['cloud_cover'] = cloud
new_obs['wind_speed'] = w_speed
new_obs['wind_gusts'] = w_gusts
new_obs['shortwave_radiation'] = wave_rad
new_obs['direct_radiation'] = dir_rad
new_obs['dew'] = dew
new_obs['sealevelpressure'] = s_press
new_obs['visibility'] = vis
new_obs['Unemployment'] = employ

categorical_features = new_obs.select_dtypes(include=['object', 'category'])
def preprocess_and_create_dummies(data):
    processed_data = data.copy()
    processed_data = pd.get_dummies(processed_data, columns=categorical_features.columns)
    return processed_data
new_obs= new_obs.pipe(preprocess_and_create_dummies)

# List of columns to add
new_columns = ['month_Apr', 'month_Aug', 'month_Dec', 'month_Feb', 'month_Jan', 'month_Jul',
               'month_Jun', 'month_Mar', 'month_May', 'month_Nov', 'month_Oct', 'month_Sep']

# Add columns if they don't exist already
for col in new_columns:
    if col not in new_obs.columns:
        new_obs[col] = 0

# Reorder columns
new_obs = new_obs[['Year', 'moonphase', 'temperature', 'humidity', 'feels_like', 'rain',
       'snowfall', 'snow_depth', 'cloud_cover', 'wind_speed', 'wind_gusts',
       'shortwave_radiation', 'direct_radiation', 'dew', 'sealevelpressure',
       'visibility', 'Unemployment', 'month_Apr',
       'month_Aug', 'month_Dec', 'month_Feb', 'month_Jan', 'month_Jul',
       'month_Jun', 'month_Mar', 'month_May', 'month_Nov', 'month_Oct',
       'month_Sep', 'DayOfWeek_Fri', 'DayOfWeek_Mon', 'DayOfWeek_Sat',
       'DayOfWeek_Sun', 'DayOfWeek_Thu', 'DayOfWeek_Tue', 'DayOfWeek_Wed',
       'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'is_holiday_0',
       'is_holiday_1', 'is_full_moon_0', 'is_full_moon_1', 'is_day_0',
       'is_day_1']]

# Select the stat.model from the list based on the crime type
selected_stat.model = None
for stat.model_name, stat.model in trained_stat.models:
    if 'n_crimes_stat.model' in stat.model_name:
        selected_stat.model = stat.model
        break
    elif 'Primary Type_' + v_type + '_stat.model' in stat.model_name:
        selected_stat.model = stat.model
        break

# If the selected stat.model is found
predictions = selected_stat.model.predict(new_obs)

# Show heatmap
matrix = np.array(predictions).reshape(7, 24)
fig = px.imshow(matrix,
                labels=dict(x="Hour", y="Day", color="Violent Crimes"),
                y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                x=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                   '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                   aspect="auto")

# Update layout
fig.update_layout(title='Violent Crimes Heatmap')
st.plotly_chart(fig)




# ##### Header 2 #####
st.header('Where is crime happening?')

# # DESCRIPTION
# st.markdown('Now that you have idenitfied what position you like the most, explore the skills you would need to have based on that position, your desired perfomance and your height.')

# Input
col3, col4, col5 = st.columns(3)
dy = col3.selectbox("Day of the Week", full_calendar['DayOfWeek'].unique().tolist())
hr = col4.selectbox("Hour", full_calendar['hour'].unique().tolist())
plce = col5.selectbox("Number of Top areas with Violent Crime", ['n_crimes','CRIM SEXUAL ASSAULT', 'ASSAULT', 'BATTERY', 'ROBBERY', 'INTIMIDATION', 'HOMICIDE', 'KIDNAPPING', 'HUMAN TRAFFICKING'])


# if h == 'meters':
#     height = col4.slider('Select your height', min_value=1.5, max_value=2.2, value=1.75, step=0.01)
# else:
#     height = col4.slider('Select your height', min_value=70.0, max_value=86.0, value=78.5, step=0.1)

# # Filter position and performance
# result_df = final_df[(final_df['pos'] == pos1) & (final_df['performance'] == perf)]

# # Group by hieght
# result_df = result_df.groupby([unit_dic[h]]).agg({
#     'points': 'mean',
#     'min': 'mean',
#     'fgp': 'mean',
#     'ftp': 'mean',
#     'tpp': 'mean',
#     'totReb': 'mean',
#     'assists': 'mean',
#     'steals': 'mean',
#     'blocks': 'mean',
#     'b_strength': lambda x: stat.mode(x)
# }).reset_index()

# # Function that find closest height and outputs a filtered table with that height
# def find_closest_height(df, target_height, tolerance=0.01):
#     # Check if the target height is present in the DataFrame
#     if target_height in df[unit_dic[h]].values:
#         return df[df[unit_dic[h]] == target_height]
    
#     # If not, find the closest height within the specified tolerance
#     lower_height = target_height - tolerance
#     upper_height = target_height + tolerance
    
#     # Check if there are observations for the lower and upper heights
#     lower_obs = df[(df[unit_dic[h]] >= lower_height) & (df[unit_dic[h]] <= target_height)]
#     upper_obs = df[(df[unit_dic[h]] <= upper_height) & (df[unit_dic[h]] >= target_height)]
    
#     # If both lower and upper observations are empty, and target_height is greater than max height, handle it
#     if lower_obs.empty and upper_obs.empty:
#         max_height = df[unit_dic[h]].max()
#         if target_height > max_height:
#             return df[df[unit_dic[h]] == max_height]
#         else:
#             return find_closest_height(df, target_height + 0.01, tolerance)
    
#     # If either lower or upper observations are not empty, return the one with observations
#     if not lower_obs.empty:
#         return lower_obs
#     else:
#         return upper_obs

# if height is None:
#     st.write('Enter your Height')
# else:
#     # Filter by height
#     skills = find_closest_height(result_df, height, tolerance=0.01)

#     # Round up the values in the specified columns
#     columns_to_round_up = ['points', 'min', 'totReb', 'assists', 'steals', 'blocks']
#     skills[columns_to_round_up] = np.ceil(skills[columns_to_round_up])

#     #Round to to decimal values
#     columns_to_round_up = ['fgp', 'ftp', 'tpp']
#     skills[columns_to_round_up] = round(skills[columns_to_round_up], 2)

#     # Remove height column and make table vertical
#     skills = skills.drop(unit_dic[h], axis=1).T

#     # Rename column
#     skills.rename(columns={skills.columns[0]: 'Average Value per Game'}, inplace=True)

#     # Show table
#     st.table(skills)

# with st.expander("Explore a little more"):
# ##### Data Information #####
#     st.header('Interesting Data Insights')

#     # Position Count
#     pos_counts = final_df['pos'].value_counts().reset_index()
#     pos_counts.columns = ['Position', 'Frequency']
#     fig2 = px.bar(pos_counts, x='Position', y='Frequency', labels={'Position': 'Positions', 'Frequency': 'Frequency'}, 
#                 title='Position Count', color_discrete_sequence=px.colors.sequential.Viridis)
#     st.plotly_chart(fig2)

#     # Biggest Strengths Distribution per Position
#     fig3 = px.box(final_df, x='pos', y=unit_dic[h],
#              color='pos',
#              labels={'pos': 'Position', unit_dic[h]: f'Height ({h})'},
#              title='Boxplot of Height per Position')
#     st.plotly_chart(fig3)

#     # Biggest Strengths Distribution per Position
#     fig4 = px.box(final_df, x=unit_dic[h], y='b_strength',
#              color='b_strength',
#              labels={unit_dic[h]: 'Height', 'b_strength': 'Biggest Strength'},
#              title='Boxplot of Height per Biggest Strength')
#     st.plotly_chart(fig4)

#     # Points vs Performance
#     fig5 = px.scatter(final_df, x='plusMinus', y='points', title='Scatter Plot of Points vs Performance',
#                     labels={'plusMinus': 'Performance', 'points': 'Points'})
#     st.plotly_chart(fig5)

# # Final Words
# st.markdown('Hope you enjoyed this dashboard, and were able to learn a little more about NBA players and yourself. If you want to learn more about the code I used for this dashboard, you can got to my [GitHub Repository](https://github.com/isaacaguilar97/you-in-nba). You can also go to learn more about the Exploratory Data Analysis that helped me build this Dashboard with my article in my Blog called [My NBA Exploratory Data Analysis](https://isaacaguilar97.github.io/my-blog/My-NBA-Exploratory-Data-Analysis)')

# st.write('Thank you for exploring this Data with me. Now you know what it will take you to become an NBA player! :basketball: You can do it!')