import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from xgboost import XGBClassifier


# 1. FEATURE EXTRACTION

def extract_features(path):
    try:
        x, sr = librosa.load(path)
    except:
        return None

    feats = []

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 3:
        return None
    
    delta = librosa.feature.delta(mfcc)
    deltadelta = librosa.feature.delta(mfcc, order=2)

    for mat in [mfcc, delta, deltadelta]:
        feats.extend(np.mean(mat, axis=1))
        feats.extend(np.std(mat, axis=1))

    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=x)

    feats.append(np.mean(centroid))
    feats.append(np.mean(bandwidth))
    feats.append(np.mean(rolloff))
    feats.append(np.mean(flatness))
    feats.extend(np.mean(contrast, axis=1))

    rms = librosa.feature.rms(y=x)
    zcr = librosa.feature.zero_crossing_rate(y=x)

    feats.append(np.mean(rms))
    feats.append(np.mean(zcr))

    chroma = librosa.feature.chroma_stft(y=x, sr=sr)
    feats.extend(np.mean(chroma, axis=1))

    return np.array(feats)


def feature_extraction(df):
    features = []
    for _, record in tqdm(df.iterrows(), total=df.shape[0]):
        path = record["filename"]
        extracted = extract_features(path)
        if extracted is None:
            continue
        features.append(extracted)

    dataf = pd.DataFrame(features)
    dataf["class"] = df["is_dysarthria"]
    dataf["speaker_id"] = df["speaker_id"]
    return dataf



# 2. LOAD DATA

df = pd.read_csv("dataset/data_more_info.csv")
dataf = feature_extraction(df)

dataf.loc[dataf["class"] == "non_dysarthria", "class"] = 0.0
dataf.loc[dataf["class"] == "dysarthria", "class"] = 1.0
dataf["class"] = dataf["class"].astype(float)

feature_cols = [c for c in dataf.columns if c not in ["class", "speaker_id"]]


# 3. XGBOOST MODEL

def create_xgb():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1
    )


# 4. CROSS-VALIDATION (leave-two-speakers-out)

speakers_0 = dataf[dataf["class"] == 0.0]["speaker_id"].unique()
speakers_1 = dataf[dataf["class"] == 1.0]["speaker_id"].unique()

accuracy = []
precision = []
recall = []

for spk0 in speakers_0:
    for spk1 in speakers_1:

        print(f"\n=== Тестируем пару: {spk0} (0.0) + {spk1} (1.0) ===")

        test_speakers = [spk0, spk1]

        train_df = dataf[~dataf["speaker_id"].isin(test_speakers)]
        test_df = dataf[dataf["speaker_id"].isin(test_speakers)]

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["class"].values
        y_test = test_df["class"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = create_xgb()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(classification_report(y_test, preds))

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)

print("\n=== ИТОГ ===")
print("Средний accuracy:", np.mean(accuracy))
print("Средний precision:", np.mean(precision))
print("Средний recall:", np.mean(recall))
