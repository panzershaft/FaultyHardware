import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical

X = preprocessor.data.drop(columns=['Label'])
y = preprocessor.data['Label']
#X = X.to_numpy()
#y = y.to_numpy()
smote = SMOTE()
X, y = smote.fit_resample(X, y)
#y = pd.DataFrame(to_categorical(y),columns=['Label0','Label1'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

model = Sequential()
model.add(Input(55, name="Input_layer_1"))
model.add(Dense(165, activation='sigmoid', name="Hidden_layer_1"))
model.add(Dense(330, activation='sigmoid', name="Hidden_layer_2"))
model.add(Dense(1, activation='sigmoid', name="Output_layer"))

model.build()
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'], run_eagerly=True)

model.fit(X_train, y_train, batch_size=20,
                    epochs=50, verbose='auto',
                    validation_data=(X_test, y_test),
                    shuffle=True)
eval = model.evaluate(X_test, y_test, batch_size=20, verbose='auto')

predictions = model.predict(X_test) > 0.5
#predictions.shape
#y_test.shape
#print(predictions)
my_cm = confusion_matrix(y_test, predictions)
snn_df_cm = pd.DataFrame(my_cm)
sns.heatmap(snn_df_cm, annot=True,cbar=False, cmap='Blues', fmt='g')
plt.ylabel('Predicted Label')
plt.xlabel('Actual Label')
plt.show()