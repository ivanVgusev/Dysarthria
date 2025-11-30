# Описание файлов и хода работы

==================================================

УЛУЧШЕНИЕ КЛАССИФИКАТОРА

==================================================

### base_code.ipynb
Изначальный ноутбук с kaggle

### speaker_CV.py
В изначальной работе данные при обучении брались для всех дикторов:

```python
dataf.loc[dataf['class']=='non_dysarthria','class'] = 0.0
dataf.loc[dataf['class']=='dysarthria','class'] = 1.0
dataf['class'] = dataf['class'].astype(float)

X = dataf.iloc[:,:-1].values
y = dataf.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
X_train = X_train.reshape(-1,16,8,1)
X_test = X_test.reshape(-1,16,8,1)
```

Из-за этого вызникло предположение, что высокие показатели precision и recall могли быть в действительности связаны не результативностью алгоритма, а с тем, что модель научилась запоминать дикторов. 

<i> Решение </i> ==> посмотреть, что будет, если обучать и тестировать модель на <b> разных </b> дикторах: применить кросс-валидацию. 

Перебором считаем результаты с "выкинутыми дикторами":
```python
speakers_0 = dataf[dataf['class'] == 0.0]['speaker_id'].unique()
speakers_1 = dataf[dataf['class'] == 1.0]['speaker_id'].unique()

results = []

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

        X_train = X_train.reshape(-1, 16, 8, 1)
        X_test  = X_test.reshape(-1, 16, 8, 1)

        model = create_model()
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

        preds = model.predict(X_test)
        preds = (preds > 0.5).astype(int).reshape(-1)

        acc = accuracy_score(y_test, preds)

        print(f"Точность: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, preds))

        results.append(acc)

print("\n=== ИТОГ ===")
print("Средняя точность по всем парам дикторов:", np.mean(results))
```

<b> Результаты показали: </b>

1) Средний accuracy по всем парам дикторов: 0.6504301745889672

2) Средний precision по всем парам дикторов: 0.622147737872381

3) Средний recall по всем парам дикторов: 0.7489537753121956

### speaker_CV_broaden_features.py

Расширяет список рассматриваемых характеристик (в оригинальной работе только mean MFCC), добавляет: 

```python
def extract_features(path):
    try:
        x, sr = librosa.load(path)
    except:
        return None

    feats = []

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 3:
        return None
    
    delta = librosa.feature.delta(mfcc)
    deltadelta = librosa.feature.delta(mfcc, order=2)

    for mat in [mfcc, delta, deltadelta]:
        feats.extend(np.mean(mat, axis=1))
        feats.extend(np.std(mat, axis=1))

    # spectral features
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

    # RMS + ZCR
    rms = librosa.feature.rms(y=x)
    zcr = librosa.feature.zero_crossing_rate(y=x)

    feats.append(np.mean(rms))
    feats.append(np.mean(zcr))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=x, sr=sr)
    feats.extend(np.mean(chroma, axis=1))

    return np.array(feats)
```

<b> Результаты показали: </b>

1) Средний accuracy: 0.8658976645093469

2) Средний precision: 0.8254479456766258

3) Средний recall: 0.9768521345949631


==================================================

СЛУЖЕБНЫЕ ФАЙЛЫ

==================================================

### data_restructuring.py
Модифицирует изначальный .csv файл и добавляет в него колонку 'speaker_id' с метками дикторов (файл data_more_info.py). Это полезно при дальнейшей проверке достоверности полученных оригинальным исследованием результатов <i>(см. УЛУЧШЕНИЕ КЛАССИФИКАТОРА.diff_speakers для более подробного объяснения). </i>


==================================================

ДАТАСЕТЫ

==================================================

### data.csv

Изначальный файл с данными, содержит столбцы [is_dysarthria,gender,filename].

### data_more_info.csv

Модифицированный файл, в который был добавлен столбец [speaker_id] и был обновлен столбец filename.

