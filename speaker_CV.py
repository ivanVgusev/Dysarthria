import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten


# ============================
# 1. FEATURE EXTRACTION
# ============================

def feature_extraction(df):
    nums_left_out = 0

    features = []
    speaker_ids = []
    classes = []

    for _, record in tqdm(df.iterrows(), total=df.shape[0]):
        path = record["filename"]
        try:
            x, sr = librosa.load(path)
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128)
            
            mean_mfcc = np.mean(mfcc, axis=1)
            features.append(mean_mfcc)

            classes.append(record["is_dysarthria"])
            speaker_ids.append(record["speaker_id"])

        except:
            print(f"Ошибка чтения файла: {path}")
            continue

    dataf = pd.DataFrame(features)
    dataf['class'] = classes
    dataf['speaker_id'] = speaker_ids
    print(f"ВСЕГО УДАЛЕНО ИЗ ВЫБОРКИ: {nums_left_out}")
    return dataf


# ============================
# 2. ЗАГРУЗКА CSV
# ============================

df = pd.read_csv("dataset/data_more_info.csv")
dataf = feature_extraction(df)

dataf.loc[dataf['class'] == 'non_dysarthria', 'class'] = 0.0
dataf.loc[dataf['class'] == 'dysarthria', 'class'] = 1.0
dataf['class'] = dataf['class'].astype(float)


# ============================
# 3. МОДЕЛЬ
# ============================

def create_model():
    model = Sequential([
        InputLayer(input_shape=(16, 8, 1)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# ============================
# 4. ПОЛНОЕ ПЕРЕБОРНОЕ HOLDOUT-ТЕСТИРОВАНИЕ
# ============================

speakers_0 = dataf[dataf['class'] == 0.0]['speaker_id'].unique()
speakers_1 = dataf[dataf['class'] == 1.0]['speaker_id'].unique()

accuracy = []
precision = []
recall = []

for spk0 in speakers_0:
    for spk1 in speakers_1:

        print(f"\n=== Тестируем пару: {spk0} (0.0) + {spk1} (1.0) ===")

        test_speakers = [spk0, spk1]

        train_df = dataf[~dataf['speaker_id'].isin(test_speakers)]
        test_df  = dataf[dataf['speaker_id'].isin(test_speakers)]

        X_train = train_df.iloc[:, :128].values
        X_test  = test_df.iloc[:, :128].values
        y_train = train_df['class'].values
        y_test  = test_df['class'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = X_train.reshape(-1, 16, 8, 1)
        X_test  = X_test.reshape(-1, 16, 8, 1)

        model = create_model()
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

        preds = model.predict(X_test)
        preds = (preds > 0.5).astype(int).reshape(-1)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print("Classification report:")
        print(classification_report(y_test, preds))

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)

print("\n=== ИТОГ ===")
print("Средний accuracy по всем парам дикторов:", np.mean(accuracy))
print("Средний precision по всем парам дикторов:", np.mean(precision))
print("Средний recall по всем парам дикторов:", np.mean(recall))
