import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

iris= datasets.load_iris()
X= iris.data
y= iris.target
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape)
print(y_train.shape)

#decision tree
dt= DecisionTreeClassifier(random_state=0)
dt.fit(X_train,y_train)
dt_pred= dt.predict(X_test)

dt_acc= accuracy_score(y_test,dt_pred)
print("dt accuracy: "+ str(dt_acc))

###svm
from sklearn import svm
svm_cls= svm.SVC(kernel="linear")
svm_cls.fit(X_train,y_train)
svm_pred = svm_cls.predict(X_test)
svm_acc= accuracy_score(y_test,svm_pred)
print("svm accuracy: " + str(svm_acc))




#linear regression

#home m^2 price estimation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([50,100,150,200,250])
y= np.array([1000000,2000000,3000000,4000000,5000000])
"""""
plt.style.use("ggplot")
plt.plot()
plt.scatter(X,y)
plt.xlabel("Evin metre karesi")
plt.ylabel("evin fiyatı")
plt.title("Ev metre karesine göre fiyatlar")
#plt.show()
"""""

model = LinearRegression()
model.fit(X.reshape(-1,1),y)

#custom_m2= int(input("Ev metrekareniz"))
#est_price= model.predict([[custom_m2]])
#print("{0} metrekare evin tahmini fiyatı: {1}".format(custom_m2,est_price[0]))


tahminler = model.predict(np.array([50,100,150,200,250]).reshape(-1,1))
print(tahminler)
"""
plt.style.use("ggplot")
plt.scatter(X,y)
plt.plot(X,tahminler,label="Regresyon tahmini",color="blue")
plt.xlabel("Evin metre karesi")
plt.ylabel("evin fiyatı")
plt.title("Ev metre karesine göre fiyatlar")
plt.legend()
plt.show()
"""



weather_data = {
    "hava durumu": [
        "güneşli",
        "bulutlu",
        "yağmurlu",
        "güneşli",
        "bulutlu",
        "yağmurlu",
        "güneşli",
        "bulutlu",
        "yağmurlu",
        "güneşli"
    ],
    "sıcaklık": [
        "Sıcak",
        "Ilık",
        "Soğuk",
        "Ilık",
        "Sıcak",
        "Soğuk",
        "Ilık",
        "Sıcak",
        "Soğuk",
        "Ilık"
    ],
    "rüzgar durumu": ['rüzgarlı', 'rüzgarlı', 'rüzgarsız', 'rüzgarlı', 'rüzgarsız', 'rüzgarlı', 'rüzgarsız', 'rüzgarsız', 'rüzgarlı', 'rüzgarsız'],  
    "olumlu": [
        "E",
        "H",
        "E",
        "H",
        "E",
        "H",
        "E",
        "H",
        "E",
        "H"
    ]
}

#hava durumunda
# Create DataFrame
df = pd.DataFrame(weather_data)

# Define mappings
hava_map = {"güneşli": 0, "bulutlu": 1, "yağmurlu": 2}
sicaklik_map = {"Soğuk": 0, "Ilık": 1, "Sıcak": 2}
ruzgar_map = {"rüzgarsız": 0, "rüzgarlı": 1}
olumlu_map = {"H": 0, "E": 1}

# Apply mappings
df_mapped = df.copy()
df_mapped["hava durumu"] = df["hava durumu"].map(hava_map)
df_mapped["sıcaklık"] = df["sıcaklık"].map(sicaklik_map)
df_mapped["rüzgar durumu"] = df["rüzgar durumu"].map(ruzgar_map)
df_mapped["olumlu"] = df["olumlu"].map(olumlu_map)

print(df_mapped)

#bağımlı değişken
X= df_mapped.drop("olumlu",axis=1)
y= df_mapped["olumlu"]
print(X)
print(y)

from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,y)



import tkinter as tk
from tkinter import ttk, messagebox

# 🖼️ GUI setup
root = tk.Tk()
root.title("Makine Öğrenmesi Eğitim Tahmini")
root.geometry("400x300")

# 🎛️ Dropdowns
hava_var = tk.StringVar()
sicaklik_var = tk.StringVar()
ruzgar_var = tk.StringVar()

ttk.Label(root, text="Hava Durumu:").pack(pady=5)
hava_combo = ttk.Combobox(root, textvariable=hava_var, values=list(hava_map.keys()), state="readonly")
hava_combo.pack()

ttk.Label(root, text="Sıcaklık:").pack(pady=5)
sicaklik_combo = ttk.Combobox(root, textvariable=sicaklik_var, values=list(sicaklik_map.keys()), state="readonly")
sicaklik_combo.pack()

ttk.Label(root, text="Rüzgar Durumu:").pack(pady=5)
ruzgar_combo = ttk.Combobox(root, textvariable=ruzgar_var, values=list(ruzgar_map.keys()), state="readonly")
ruzgar_combo.pack()

# 🔮 Prediction function
def predict():
    try:
        new_data = pd.DataFrame({
            "hava durumu": [hava_map[hava_var.get()]],
            "sıcaklık": [sicaklik_map[sicaklik_var.get()]],
            "rüzgar durumu": [ruzgar_map[ruzgar_var.get()]]
        })
        tahmin = model.predict(new_data)
        if tahmin[0] == 1:
            messagebox.showinfo("Sonuç", "✅ Eğitime girebilirsiniz.")
        else:
            messagebox.showinfo("Sonuç", "🚫 Eğitime gitmenizi tavsiye etmiyorum.")
    except Exception as e:
        messagebox.showerror("Hata", f"Girdi hatası: {e}")

# 🧮 Predict button
ttk.Button(root, text="Tahmin Et", command=predict).pack(pady=20)

# 🚀 Run GUI
root.mainloop()

"""
hava_durumu= input("Lütfen hava durumunu giriniz")
sicaklik= input("Lütfen sıcaklık giriniz.")
ruzgar = input("Lütfen rüzgar bilgisini giriniz.")
new_data = pd.DataFrame({"hava durumu":[hava_durumu],"sıcaklık":[sicaklik],"rüzgar durumu":[ruzgar]})


new_data["hava durumu"] = new_data["hava durumu"].map(hava_map)
new_data["sıcaklık"] = new_data["sıcaklık"].map(sicaklik_map)
new_data["rüzgar durumu"] = new_data["rüzgar durumu"].map(ruzgar_map)
tahmin = model.predict(new_data)
if tahmin[0]==1:
    print("Makine öğrenmesine giriş eğitimine girebilirsiniz")
else:
    print("Makine öğrenmesi giriş eğitimine gitmenizi tavsiye etmiyorum.")

"""
#0537 993 41 02 hocanın telefon. Ömer Faruk Doğan
