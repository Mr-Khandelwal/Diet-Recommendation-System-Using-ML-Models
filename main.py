import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


st.set_page_config(
    page_title="Food Recommendation App",
    page_icon=":tomato:",
    layout="wide",
)
# Additional Streamlit formattingS
st.sidebar.header("About")
st.sidebar.text("Welcome to our\nDIET RECOMMENDATION SYSTEM.\nEnter your details and\nget your fitness details.")
st.sidebar.text("Created by:\nRajat Khandelwal\nRhythm\nPrerna Badlnai")
# Load the pre-trained models
with open("./model/extratrees_model.pkl", "rb") as f:
    extra_trees_model = pickle.load(f)

# Load Gradient Boosting model
with open("./model/gradientboosting_model.pkl", "rb") as f:
    gradient_boosting_model = pickle.load(f)

def calculate_bmr(weight, height, age, gender):
    if gender == "Male":  # 0 represents Male
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    elif gender == "Female":  # 1 represents Female
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        raise ValueError("Invalid gender. Use '0' for Male or '1' for Female.")
    return bmr

# Function to calculate Body Fat Percentage
def calculate_body_fat_percentage(bmi, age, gender):
    if age >= 18 and age <= 21:
        return (1.51 * bmi) - (0.70 * age) - (3.6 * gender) + 14
    elif age >= 22:
        return (1.391 * bmi) + (0.16 * age) - (10.34 * gender) - 9
    else:
        raise ValueError("Invalid age range")

st.markdown(
    """
    <style>
    .header {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="header">Diet Recommendation System</div>', unsafe_allow_html=True)

# Inputs
st.subheader("Enter Your Details")
gender = st.selectbox("Enter Your Gender:", ["Female", "Male"])
age = st.number_input("Enter Your Age:", min_value=1, max_value=100, value=25)
height = st.number_input("Enter Your Height (in cm):", min_value=50.0, max_value=250.0, value=170.0)
weight = st.number_input("Enter Your Weight (in kg):", min_value=20.0, max_value=200.0, value=70.0)
lifestyle = st.selectbox(
    "Enter Your Lifestyle:",
    [
        "Sedentary or light activity (e.g., Office worker getting little or no exercise)",
        "Active or moderately active (e.g., Construction worker or running 1 hour daily)",
        "Vigorously active (e.g., Agricultural worker or swimming 2 hours daily)"
    ]
)
exercise = st.selectbox(
    "Enter Your Exercise Level:",
    [
        "Sedentary (little or no exercise)",
        "Lightly active (light exercise 1-3 days/week)",
        "Moderately active (moderate exercise 3-5 days/week)",
        "Very active (hard exercise 6-7 days/week)"
    ]
)
target = st.selectbox(
    "Enter Your Target:",
    [
        "Gain Muscle",
        "Loose Weight",
        "Maintain Weight"
        # "Very active (hard exercise 6-7 days/week)"
    ]
)
# target = st.number_input("Enter Your Target (e.g., Gain Muscle, Lose Weight, Maintain Weight):")

lifestyle_mapping = {
    "Sedentary or light activity (e.g., Office worker getting little or no exercise)":0,
    "Active or moderately active (e.g., Construction worker or running 1 hour daily)":1,
    "Vigorously active (e.g., Agricultural worker or swimming 2 hours daily):" :2
}
# Map input data for prediction
exercise_mapping = {
    "Sedentary (little or no exercise)": 0,
    "Lightly active (light exercise 1-3 days/week)": 1,
    "Moderately active (moderate exercise 3-5 days/week)": 2,
    "Very active (hard exercise 6-7 days/week)": 3
}
target_mapping = {
    "Gain Muscle":0,
    "Loose Weight":1,
    "Maintain Weight" :2
}
gender_mapping = {"Female": 0,"Male": 1}

# if st.button("Predict"):
    # Prepare the input array for the model
input_data = np.array([
    gender_mapping[gender],
    age,
    height,
    weight,
    lifestyle_mapping[lifestyle],
    exercise_mapping[exercise],
    target_mapping[target]
])

# calculating BMR,BMI, BFP
bmi = weight / ((height / 100) ** 2)
bmr = calculate_bmr(weight,height,age,gender)
if(gender=="Male"):
    body_fat_percentage = calculate_body_fat_percentage(bmi, age, gender=1)
else:
    body_fat_percentage = calculate_body_fat_percentage(bmi, age, gender=0)

input_data = np.append(input_data, [[bmi, bmr, body_fat_percentage]])

# Predict Daily Calorie Need & TDEE
# predictions = extra_trees_model.predict(input_data)
print(input_data)
predictions = extra_trees_model.predict(np.array(input_data).reshape(1,-1))

# output_scaler = StandardScaler()
# descaled_prediction = output_scaler.inverse_transform(predictions)
# print(descaled_prediction)
daily_calorie_need, tdee = predictions[0]

# Predict Suggested Diet
diet_prediction = gradient_boosting_model.predict(np.array(input_data).reshape(1,-1))
diet_suggestion = ["BALANCE DIET (55:25:20)","HIGH CARB DIET(60:20:20)","ZONE DIET (40:30:30)","KETOGENIC DIET (5:35:60","LOW CARB DIET (25:35:40)","DEPLETION DIET (DYNAMIC)"][diet_prediction[0]]

st.write(f"**BMI** {bmi}")
st.write(f"**BMR** {bmr}")
st.write(f"**BFP** {body_fat_percentage}")

# st.subheader("Prediction Results")
st.subheader(f"**Daily Calorie Need:** {daily_calorie_need:.2f} calories/day")
st.subheader(f"**Total Daily Energy Expenditure (TDEE):** {tdee:.2f} calories/day")
st.subheader(f"**Suggested Diet:** {diet_suggestion}")


# diet suggestion 

data = pd.read_csv("./wweia_dataset/wweia_data.csv")
data_category = pd.read_csv("./wweia_dataset/wweia_food_categories_addtl.csv")
updated = data.iloc[:,3:]
updt = data.iloc[:,:3]
df=data
# 1st method makes features between -1 to 1
data_norm_changed = updated.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))
data_norm = pd.concat([updt, data_norm_changed], axis ="columns")


synonims = {1002: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1004: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1006: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1008: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1202: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1204: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1206: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1208: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1402: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1404: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
1602: {1602, 1604},
1604: {1602, 1604},
1820: {1820, 1822},
1822: {1820, 1822},
2002: {2002, 2004},
2004: {2002, 2004},
2006: {2006},
2008: {2008},
2010: {2010},
2202: {2202, 2204, 2206, 3004},
2204: {2202, 2204, 2206, 3004},
2206: {2202, 2204, 2206, 3004},
2402: {2402, 2404, 3006},
2404: {2402, 2404, 3006},
2502: {2502},
2602: {2602},
2604: {2604},
2606: {2606},
2608: {2608},
2802: {2802},
2804: {2804},
2806: {2806},
3002: {3002},
3004: {2202, 2204, 2206, 3004},
3006: {2402, 2404, 3006},
3202: {3202, 4002},
3204: {3204, 4004},
3206: {3206},
3208: {3208},
3402: {3402},
3404: {3404},
3406: {3406},
3502: {3502},
3504: {3504},
3506: {3506},
3602: {3602},
3702: {3702},
3703: {3703, 3704, 3706, 3708, 3720, 3722},
3704: {3703, 3704, 3706, 3708, 3720, 3722},
3706: {3703, 3704, 3706, 3708, 3720, 3722},
3708: {3703, 3704, 3706, 3708, 3720, 3722},
3720: {3703, 3704, 3706, 3708, 3720, 3722},
3722: {3703, 3704, 3706, 3708, 3720, 3722},
3802: {3802},
4002: {3202, 4002},
4004: {3204, 4004},
4202: {4202, 4402},
4204: {4204},
4206: {4206},
4208: {4208},
4402: {4202, 4402},
4404: {4404},
4602: {4602, 4604, 4804},
4604: {4602, 4604, 4804},
4802: {4802},
4804: {4602, 4604, 4804},
5002: {5002, 5004},
5004: {5002, 5004},
5006: {5006},
5008: {5008},
5202: {5202, 5204},
5204: {5202, 5204},
5402: {5402, 5404},
5404: {5402, 5404},
5502: {5502},
5504: {5504},
5506: {5506},
5702: {5702, 5704},
5704: {5702, 5704},
5802: {5802},
5804: {5804},
5806: {5806},
6002: {6002},
6004: {6004},
6006: {6006},
6008: {6008},
6010: {6010},
6012: {6012},
6014: {6014},
6016: {6016},
6018: {6018},
6402: {6402},
6404: {6404},
6406: {6406},
6408: {6408},
6410: {6410},
6412: {6412},
6414: {6414},
6416: {6416},
6418: {6418},
6420: {6420},
6422: {6422},
6802: {6802, 6804, 6806},
6804: {6802, 6804, 6806},
6806: {6802, 6804, 6806},
7002: {7002, 7004, 7006, 7008},
7004: {7002, 7004, 7006, 7008},
7006: {7002, 7004, 7006, 7008},
7008: {7002, 7004, 7006, 7008},
7102: {7102, 7104, 7106},
7104: {7102, 7104, 7106},
7106: {7102, 7104, 7106},
7202: {7202},
7204: {7204},
7206: {7206},
7208: {7208},
7220: {7220},
7302: {7302},
7304: {7304},
7502: {7502},
7504: {7504},
7506: {7506},
7702: {7702, 7704, 7802, 7804},
7704: {7702, 7704, 7802, 7804},
7802: {7702, 7704, 7802, 7804},
7804: {7702, 7704, 7802, 7804},
8002: {8002},
8004: {8004},
8006: {8006, 8008},
8008: {8006, 8008},
8010: {8010},
8012: {8012},
8402: {8402, 8404, 8406},
8404: {8402, 8404, 8406},
8406: {8402, 8404, 8406},
8408: {8408},
8410: {8410},
8412: {8412},
8802: {8802},
8804: {8804},
8806: {8806},
9002: {9002},
9004: {9004},
9006: {9006},
9008: {9008},
9010: {9010},
9012: {9012},
9202: {9202},
9204: {9204},
9402: {9402, 9404, 9406},
9404: {9402, 9404, 9406},
9406: {9402, 9404, 9406},
9602: {9602},
9802: {9802},
9999: {9999}}

def recommend_1(df,final):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(data["description"]).toarray()
    knn = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='cosine')
    knn.fit(vectors)

    recommendations = []

    p=0

    def recommend(index):
        calories = []
        energy = 0
        remaining = cal_per_cat
        recipe_index = index
        _, indices = knn.kneighbors(vectors[recipe_index].reshape(1, -1))
        
        for i in indices[0][1::10]:
            if energy < cal_per_cat:
                if data.iloc[i].energy_kcal < remaining:
                    recommendation = f"{data.iloc[i].description}, -- calories = {data.iloc[i].energy_kcal} Kcal"
                    calories.append(data.iloc[i].energy_kcal)
                    energy += data.iloc[i].energy_kcal
                    remaining = cal_per_cat - energy
                    recommendations.append(recommendation)
        p=i
        return np.sum(calories)

    total_cal = []
    st.write("THE RECOMMENDED FOODS ARE")
    for item in final:
        energy = recommend(item)
        total_cal.append(energy)

    total_energy = np.sum(total_cal)
    error = abs(total_energy - cal) / cal
    b = p

    while error > 0.15:
        b += 1
        recommendation = f"{data.iloc[b].description}, -- calories = {data.iloc[b].energy_kcal} Kcal"
        total_energy += data.iloc[b].energy_kcal
        recommendations.append(recommendation)
        error = abs(total_energy - cal) / cal

    st.write("--------------------")
    st.write("The total calories of the recommended food ", total_energy, "Kcal")
    st.write("Error percentage between calories actually needed and recommended calories = {:.2f}%".format(error * 100))
    
    return recommendations

def Select(df3):
    final = []
    new_indexes = list(df3.index)
    randm = []
    for b in range(10):
        randm.append(random.choice(new_indexes))
    user_sel = []
    st.write("Select one among these that you prefer the most:")
    for x in randm:
        st.write("{}   -> {}".format(x, data["description"][x]))
    z = st.text_input("Enter index:", key=str(df3.shape))
    if z.strip():  # Check if z is not an empty string
        user_sel = list(map(int, z.split(" ")))
        return user_sel
    else:
        # Handle the case where z is an empty string
        return []
    return user_sel

# Define your custom theme
custom_theme = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f7f7f7;
        color: #333333;
        margin: 0;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: #ecf0f1;
    }
    .header {
        background-color: #3498db;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
"""

# Apply the custom theme
st.markdown(custom_theme, unsafe_allow_html=True)
cal = predictions[0,0]
cal_per_cat = cal/6

categories_range = [0, 14, 35, 82, 91, 105, 155]
food_categories = [f"{data_category['larger_category'][i]} - {data_category['wweia_food_category_description'][i]}" for i in range(len(data_category))]
selected = st.multiselect("Select food categories:", food_categories)
selected_indices = []
for category in selected:
        # Split the selected category to extract larger_category and wweia_food_category_description
        larger_category, food_category_description = category.split(" - ", 1)
        # Find the corresponding index based on the two columns
        index = data_category[(data_category['larger_category'] == larger_category) & (data_category['wweia_food_category_description'] == food_category_description)].index.tolist()
        if index:
            selected_indices.extend(index)
if len(selected) == 6:
    header_list = ["wweia_food_category_code", "wweia_food_category_description", "larger_category", "same_category"]
    df1 = pd.read_csv('./wweia_dataset/wweia_food_categories_addtl.csv', skiprows=lambda x: x not in selected_indices, names=header_list)

    def category_code():
        return df1["wweia_food_category_code"]

    list_of_categories = category_code().to_list()

    new_list = []
    for j in list_of_categories:
        categ = synonims[j]
        categ = list(categ)
        new_list.append(categ)

    # Your previous code
    select_user = []
    final = []
    check = []
    food2 = []
    for x in new_list:
        for y in x:
            z = data.loc[data["wweia_category_code"]==y]
            food2.append(z)
        df3 = pd.concat(food2)
        check.append(df3)
        select_user.append(food2)
        final.append(Select(df3))
        food2 = []


else:
    st.write("Please select exactly 6 food categories.")
# Streamlit app command
if st.button("Get Food Recommendations", key="recommend_button"):
    # Perform recommendation based on selected categories
    recommendations = recommend_1(df, final)
    # Display recommendations
    st.subheader("Recommended Foods:")
    for recommendation in recommendations:
        st.success(recommendation)












    # categories_range = [0, 14, 35, 82, 91, 105, 155]
    # selected = []

    # st.write("Select 1 out of these sub-categories:")
    # for i in range(len(categories_range) - 1):
    #     st.write(f"Larger Category: {data_category['larger_category'][categories_range[i]]}")

    #     # Generate the list of sub-categories in the range and display them
    #     sub_categories = data_category.iloc[categories_range[i]:categories_range[i + 1]]
    #     sub_category_options = sub_categories["wweia_food_category_description"].to_list()
        
    #     # Create a selectbox for each sub-category
    #     user_input = st.selectbox(f"Select a sub-category in the range {categories_range[i]} to {categories_range[i + 1]}", options=sub_category_options)
        
    #     # Find the index of the selected category and add it to the selected list
    #     selected_category_index = sub_category_options.index(user_input) + categories_range[i]
    #     selected.append(data_category["wweia_food_category_code"][selected_category_index])


    # header_list = ["wweia_food_category_code","wweia_food_category_description","larger_category","same_category"]
    # data_category = pd.read_csv('wweia_dataset/wweia_food_categories_addtl.csv',skiprows=lambda x: x not in selected, names = header_list)
    # def category_code():
    #     # returns category code of foods selected by user
    #     return data_category["wweia_food_category_code"]
    # list_of_categories = category_code().to_list()

    # new_list = []
    # for j in list_of_categories:
    #     categ = synonims[j]
    #     categ = list(categ)
    #     new_list.append(categ)
    # # new_list
    # food_ = []
    # an_list = []
    # i = 0
    # for x in new_list:
    #     print("Sub-category {}".format(i+1))
    #     for y in x:
    #         z_ = data_norm.loc[data_norm["wweia_category_code"]==y]
    #         food_.append(z_)
    #     data_f = pd.concat(food_)
    #     if i == 0:
    #         k = data_f.shape[0]
    #         print("Total food items=",k)
    #     elif i==1:
    #         num = data_f.shape[0]
    #         m = num - k
    # #         print(num)
    #         print("Total food items=",m)
    #     else:    
    #         m = data_f.shape[0] - num
    #         num = data_f.shape[0]
    # #         print(num)
    #         print("Total food items=",m)
    #     i+=1

    # k_v = 100
    # splitting = 70

    # def Select(df3, data):
    #     # List to store user-selected indexes
    #     final = []
        
    #     # Get the list of all indexes in the DataFrame
    #     new_indexes = list(df3.index)
        
    #     # Randomly select 10 items from the DataFrame
    #     randm = random.sample(new_indexes, 10)
        
    #     st.write("Select from the following options:")
    #     options = []
        
    #     # Prepare the options for display
    #     for x in randm:
    #         options.append(f"{x} -> {data['description'][x]}")  # Show index and description
        
    #     # Use a multiselect for the user to choose multiple items
    #     user_sel = st.multiselect("Select food items", options)
        
    #     # Convert the user input back to index numbers
    #     selected_indexes = [int(option.split(" -> ")[0]) for option in user_sel]  # Extract indexes
        
    #     return selected_indexes


    # # Example Streamlit UI to select categories
    # select_user = []
    # final = []
    # check = []
    # food2 = []

    # # Assuming 'new_list' contains the list of category codes
    # for x in new_list:
    #     food2 = []  # Reset food2 list for each iteration
    #     for y in x:
    #         z = data.loc[data["wweia_category_code"] == y]
    #         food2.append(z)  # Collect food items of the current category
        
    #     df3 = pd.concat(food2)  # Concatenate food items
    #     check.append(df3)  # Track the filtered data
    #     select_user.append(food2)  # Track the food items for selection
    #     selected_items = Select(df3, data)  # Get selected items for this category
    #     final.append(selected_items)  # Store the selected indexes

    # # Show the selected food items after user input
    # st.write("User Selected Food Items:")

    # # Loop through the final selections and show the descriptions
    # for selection in final:
    #     for idx in selection:
    #         st.write(data["description"][idx])  # Show the description of selected food items




    # def recommend_1(k_value):
    #     # TF-IDF vectorizer for transforming the descriptions into vectors
    #     tfidf = TfidfVectorizer(stop_words="english")
    #     vectors = tfidf.fit_transform(data["description"]).toarray()
        
    #     # Nearest Neighbors model with cosine distance
    #     knn = NearestNeighbors(n_neighbors=k_value, algorithm='brute', metric='cosine')
    #     knn.fit(vectors)

    #     def recommend(index):
    #         calories = []
    #         energy = 0
    #         remaining = cal_per_cat  # Total calories remaining to be filled
            
    #         # Finding the most similar food items (neighbors)
    #         recipe_index = index
    #         _, indices = knn.kneighbors(vectors[recipe_index].reshape(1, -1))
            
    #         # We will show the top k most similar recipes, taking into account calorie limits
    #         recommended_items = []
    #         for i in indices[0][1::10]:  # Skipping the first item (which is the item itself)
    #             if energy < cal_per_cat:
    #                 if data.iloc[i].energy_kcal < remaining: 
    #                     recommended_items.append((data.iloc[i].description, data.iloc[i].energy_kcal))
    #                     calories.append(data.iloc[i].energy_kcal)
    #                     energy += data.iloc[i].energy_kcal
    #                     remaining = cal_per_cat - energy
            
    #         # Return the list of recommended items and the total calorie count
    #         return recommended_items, np.sum(calories)

    #     # Start Streamlit UI
    #     st.write("THE RECOMMENDED FOODS ARE:")

    #     total_cal = []
    #     all_recommended_items = []
        
    #     # Loop through user-selected items in 'final' to generate recommendations
    #     for item in final:
    #         recommended_items, energy = recommend(item)
    #         all_recommended_items.extend(recommended_items)
    #         total_cal.append(energy)

    #     total_energy = np.sum(total_cal)
        
    #     # Calculate the error (difference between recommended and desired calories)
    #     error = abs(total_energy - predictions[0, 0]) / predictions[0, 0]
        
    #     # Display the recommended items and calorie details
    #     for description, kcal in all_recommended_items:
    #         st.write(f"{description} -- Calories = {kcal} Kcal")
        
    #     st.write("\n---")
    #     st.write(f"Total Calories of Recommended Food: {total_energy} Kcal")
    #     st.write(f"Error between recommended and actual calories: {error*100:.2f}%")
        
    #     # If error is larger than 15%, show additional items to reduce the error
    #     b = 0  # Start from the first item in the list
    #     while error > 0.15:
    #         b += 1
    #         st.write(f"{data.iloc[b].description} -- Calories = {data.iloc[b].energy_kcal} Kcal")
    #         total_energy += data.iloc[b].energy_kcal
    #         error = abs(total_energy - cal_per_cat) / cal_per_cat  # Recalculate the error
    #     st.write(f"The final total calories: {total_energy} Kcal")
    #     st.write(f"Error percentage: {error * 100:.2f}%")

