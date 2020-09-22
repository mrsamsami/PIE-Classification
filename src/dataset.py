import json
import sys
import pandas as pd

threshold = float(sys.argv[1])
biased = False if sys.argv[2] == 'fair' else True

with open('../dataset/raw/MAGPIE_filtered_split_random.jsonl', 'r') as file:
    list = list(file)

datapoints = []

for str in list:
    datapoint = json.loads(str)
    datapoints.append(datapoint)

df = pd.DataFrame(datapoints)
df = df[df['confidence'] >= threshold]
df = df.drop(['document_id', 'id', 'judgment_count', 'label_distribution', 'confidence', 'non_standard_usage_explanations', 'offsets', 'sentence_no', 'variant_type'], axis = 1)
df['label'] = df.label.apply(lambda x: 1 if x == 'i' else 0)

train = df[df.split.eq('training')].drop(['split'], axis = 1)
dev = df[df.split.eq('development')].drop(['split'], axis = 1)
test = df[df.split.eq('test')].drop(['split'], axis = 1)

def unbias(df, type):
    if type == 'max':
        n = len(df[df['label'] == 1]) - len(df[df['label'] == 0])
        df1 = df[df['label'] == 1].nlargest(n, 'confidence')
    else:
        frac = (len(df[df['label'] == 1]) - len(df[df['label'] == 0])) / len(df[df['label'] == 1])
        df1 = df[df['label'] == 1].sample(frac = 1 - frac)

    df2 = df[df['label'] == 0]
    return pd.concat([df1, df2])

if not biased:
    type = sys.argv[3]
    train = unbias(train, type)
    test = unbias(test, type)
    dev = unbias(dev, type)

train.to_csv('../dataset/processed/train.csv', index = False)
test.to_csv('../dataset/processed/test.csv', index = False)
dev.to_csv('../dataset/processed/dev.csv', index = False)
