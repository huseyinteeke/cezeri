import pandas
#seaborn , numpy  ve matplot yorum satırlarında belirtilen bazı grafikleri görmek için kullanıldı.
import numpy
import seaborn
import matplotlib.pyplot as plt


FONT = ("Times New Roman",16,"normal")
YELLOW ="#f7f5dd"

#----------------taking data from the database---------------#

data = pandas.read_csv("Housing.csv")
#print(data.info())  #using this method for understanding columns and data types



#---------------data preprocessing------------------------#


def convert_binary(column):
    """columns with values yes / no turning into boolean value , then turning into an integer"""
    global data
    data[f"{column}_yes"] = (data[column] == "yes").astype(int)
    data = data.drop(column, axis=1)

convert_binary("mainroad")
convert_binary("guestroom")
convert_binary("basement")
convert_binary("hotwaterheating")
convert_binary("airconditioning")
convert_binary("prefarea")
data  = data.join(pandas.get_dummies(data.furnishingstatus)).drop(["furnishingstatus"] ,axis =1).astype(int)


#train_data.hist(figsize=(15,8)) ///observing some graphs to see how data affect each other
#plt.show()




#-------------------feature engineering-------------#

data["total_rooms"] = data.bedrooms + data.bathrooms + data.guestroom_yes
data["bathroom_per_bedroom"] = data.bathrooms/data.bedrooms
data["area_per_room"] = data.area/data.total_rooms
data["area_per_story"] = data.area/data.stories
data["luxury_score"] = data.airconditioning_yes +data.guestroom_yes +data.prefarea_yes + data.hotwaterheating_yes + data.airconditioning_yes + data.furnished*1.5 + data["semi-furnished"]


#print(data.info())

#plt.figure(figsize=(15,8))
#seaborn.heatmap(data.corr(),annot=True,cmap="YlGnBu") /////corrolation map useful to see kovaryans values
#seaborn.scatterplot(x="area" , y="guestroom_yes" , data=train_data, hue="price",palette= "coolwarm")
#plt.show()



#----------------first we need to seperate our price value---------#

x = data.drop(["price"] , axis=1)
y = data["price"]



#----------------splitting the train set(%80) and test set(%20)--------------#

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test =train_test_split(x,y , test_size=0.2)

train_data = x_train.join(y_train)
test_data = x_test.join(y_test)

#plt.figure(figsize=(15,8))
#seaborn.heatmap(data.corr(),annot= True,cmap="coolwarm")
#plt.show()
#target value doesn't have a good corr. so linear regression model is useless for this dataset

#-----------------------linear regression model---------------------------#
print(x_train.info())
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)
reg = LinearRegression()
reg.fit(x_train_s,y_train)
linear_r2 = reg.score(x_test_s,y_test)

#--------------------random forest model-------------------#
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()


#random forest model is leearning train set very good , but there is over fitting
#print("Random Forest Train Score:", forest.score(x_train, y_train))# 91
#print("Random Forest Test Score:", forest.score(x_test, y_test))# 60

#---------------------finding the best parameters-------------#
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100,200,300],
    "max_features" : [3,4,5],
    "min_samples_split" : [2,4,6],
    "max_depth": [None,4,6,8]

}

grid_search = GridSearchCV(forest , param_grid, cv=5, scoring="neg_mean_squared_error"
                           ,return_train_score=True)
grid_search.fit(x_train_s,y_train)
best_forest = grid_search.best_estimator_
forest_r2 = best_forest.score(x_test_s,y_test)


#---------------------XGB REGRESSİON MODEL------------------------#
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

xgb = XGBRegressor()
param_dist = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.5, 0.7, 1]
}

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=5,
                                   scoring="neg_mean_squared_error", n_jobs=-1)#via all processors , counting will be faster
random_search.fit(x_train_s, y_train)

best_xgb = random_search.best_estimator_
xgb_r2 = best_xgb.score(x_test_s, y_test)


#---------------------------USER INTERFACE-----------------------#
from tkinter import *
from tkinter import messagebox


user_input = {}


root = Tk()
root.title("House value prediction")

#fields for entries and labels
fields = [
    ("Area (m²):", "area"),
    ("Number of bedrooms:", "bedrooms"),
    ("Number of bathrooms:", "bathrooms"),
    ("Stories:", "stories"),
    ("How many prk areas in there?", "parking"),
    ("Is it close to mainroad? (1: Yes, 0: No):", "mainroad_yes"),
    ("Does it have a guestroom? (1: Yes, 0: No):", "guestroom_yes"),
    ("Does it have basement? (1: Yes, 0: No):", "basement_yes"),
    ("Does it have hot water? (1: Yes, 0: No):", "hotwaterheating_yes"),
    ("Does it have an airconditioner? (1: Yes, 0: No):", "airconditioning_yes"),
    ("Is there a prefarea? (1: Yes, 0: No):", "prefarea_yes"),
    ("Does it furnished? (1: Yes, 0: No):", "furnished"),
    ("Does it semi-furnished? (1: Yes, 0: No):", "semi_furnished"),
    ("Does it unfurnished? (1: Yes, 0: No):", "unfurnished")
]

#entries dictionary will hold the Entry values
entries = {}

# Placement and establishing the interface
for i, (label_text, key) in enumerate(fields):
    label = Label(root, text=label_text)
    label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    entry = Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[key] = entry

#this function will make our dataframe and try to predict a value for house
def save_predict():
    """user_inputtaki tüm keyleri entriesteki keylerle eşleştiriyor,
     oluşan dataframei en yüksek R2 skoru alan metodla tahminde kullanıyor"""
    try:
        user_input["area"] = float(entries["area"].get())
        user_input["bedrooms"] = int(entries["bedrooms"].get())
        user_input["bathrooms"] = int(entries["bathrooms"].get())
        user_input["stories"] = int(entries["stories"].get())
        user_input["parking"] = int(entries["parking"].get())
        user_input["mainroad_yes"] = int(entries["mainroad_yes"].get())
        user_input["guestroom_yes"] = int(entries["guestroom_yes"].get())
        user_input["basement_yes"] = int(entries["basement_yes"].get())
        user_input["hotwaterheating_yes"] = int(entries["hotwaterheating_yes"].get())
        user_input["airconditioning_yes"] = int(entries["airconditioning_yes"].get())
        user_input["prefarea_yes"] = int(entries["prefarea_yes"].get())
        user_input["furnished"] = int(entries["furnished"].get())
        user_input["semi-furnished"] = int(entries["semi_furnished"].get())
        user_input["unfurnished"] = int(entries["unfurnished"].get())
        #added features
        user_input["total_rooms"] = user_input["bedrooms"] + user_input["bathrooms"] + user_input["guestroom_yes"]
        user_input["bathroom_per_bedroom"] = user_input["bathrooms"] / user_input["bedrooms"]
        user_input["area_per_room"] = user_input["area"] / user_input["total_rooms"]
        user_input["area_per_story"] = user_input["area"]/ user_input["stories"]
        user_input[
            "luxury_score"] = (user_input["airconditioning_yes"] + user_input["guestroom_yes"] + user_input["prefarea_yes"]
                               + user_input["hotwaterheating_yes"] + user_input["airconditioning_yes"] + user_input["furnished"] * 1.5 +
                              user_input["semi-furnished"])


        print(user_input)  #To see what's going on(for bug dedtection)
    except ValueError:
        messagebox.showerror("Error", "Lütfen tüm alanlara geçerli bir değer girin!")

    #making predictions for the price
    df = pandas.DataFrame([list(user_input.values())], columns=user_input.keys())
    scaled_df = scaler.transform(df)

    #This implementations are enough to detect the bigeest R2 score.
    if linear_r2 > forest_r2:
        if linear_r2>xgb_r2:
            prediction = reg.predict(scaled_df)
        else:
            prediction = best_xgb.predict(scaled_df)
    elif forest_r2 > xgb_r2:
        prediction = best_forest.predict(scaled_df)
    else:
        prediction = best_xgb.predict(scaled_df)
    #Final step
    messagebox.showinfo(title="predicted price",message=f"your predicted house price is {prediction}")


# Saving button
save_button = Button(root, text="Save", command=save_predict)
save_button.grid(row=len(fields), column=0, columnspan=2, pady=10)


root.mainloop()
