#naive bayes
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt


data = {
    "text": ["Merhaba","toplantı notları ektedir.","kredi kartı bilgileriniz güncellenmelidir acil!","yarınki randevuyu onaylamak için buraya tıklayın.","şimdi kaydolun ve büyük indirimlerden yararlanın.",
             "proje raporu tamalandı geri bildiriminizi bekliyorum","Piyango kazandınız! bilgilerinizi girin.","haftalık bülteniniz yayınlandı!","Acil:hesabınız askıya alındı. doğrulama yapın.","öğle yemeği için ne düşünüyorsunuz",
             ],
    "label": ["spam","ham","spam","ham","spam","ham","spam","ham","spam","ham"]
}

df= pd.DataFrame(data)
vectorizer= CountVectorizer()  #change text to count
x= vectorizer.fit_transform(df["text"])
#y = df["label"]
y= df['label'].map({'spam': 1, 'ham': 0})

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=15)
model = MultinomialNB()
model.fit(x_train,y_train)


y_pred= model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:"+ str(accuracy*100))


custom_message = ["ntı notları"]
custom_vector = vectorizer.transform(custom_message)
prediction = model.predict(custom_vector)
print("Prediction for custom message:", "Spam" if prediction[0] == 1 else "ham")




