# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers import losses
from tensorflow import keras
from keras import layers
import argparse
from sklearn.metrics import classification_report,  confusion_matrix
from transformers import AutoModel, AutoTokenizer
import torch


def print_final_report(full_report, cf_matrix):
    class_label = ['0','1','macro avg','weighted avg']
    df_final = pd.DataFrame(columns=['class_label', 'Precision_mean',"Precision_std", 'Recall_mean', 'Recall_std',
                       'F1-score_mean','F1-score_std','support_mean'],index = range(0,4))
    for index, i in enumerate(class_label):
        p = []
        r = []
        f = []
        s = []
        for j in range(3):
            p.append(full_report[j][i]['precision'])
            r.append(full_report[j][i]['recall'])
            f.append(full_report[j][i]['f1-score'])
            s.append(full_report[j][i]['support'])
        row = [i,np.mean(p),np.std(p),np.mean(r),np.std(r),np.mean(f),np.std(f),np.mean(s)]
        df_final.loc[index] = row
    results_cf = np.round(np.stack([cf_matrix[0],cf_matrix[1],cf_matrix[2]]).mean(axis=0),1)
    np.set_printoptions(suppress=True)
    return df_final,results_cf

def sim_embeddings(text_list, batch_size=16):
    if isinstance(text_list, (np.ndarray, pd.Series)):
        text_list = text_list.tolist()
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_sentences = text_list[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value for key, value in inputs.items()}

        with torch.no_grad():
            outputs = sim_model(**inputs)

        embeddings = outputs.pooler_outpu
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)

parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--sbert_model_path', type=str)
parser.add_argument('--gpt2_model_path', type=str)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()
df = pd.read_csv(args.csv_path)
if args.data_name in ['open','uav','wv','pure', 'cn']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral'],[1,0])
    class_labels = [0,1]
    print(df.Class.value_counts())
elif args.data_name in ['cdn','NLI_dev','NLI_train','NLI_test']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral','Duplicate'],[1,0,2])
    class_labels = [0,1,2]
    print(df.Class.value_counts())
else:
    print("wrong dataset name \n")

sim_model = AutoModel.from_pretrained("/content/sup-simcse-roberta-large/")
tokenizer = AutoTokenizer.from_pretrained("/content/sup-simcse-roberta-large/")
model=SentenceTransformer(args.sbert_model_path)
loss = losses.SoftmaxLoss(model=model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=args.num_labels)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps=10000, decay_rate=0.95, staircase=True)
full_report = []
cf_matrix = []
skf = StratifiedKFold(n_splits=args.n_splits, random_state = 1,shuffle = True)
t = df.Class
epochs=10
batch_size=16
for train_index, test_index in skf.split(np.zeros(len(t)), t):
    train_data = df.loc[train_index]
    test_data = df.loc[test_index]
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    train_examples= [InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)  for e in train_data.iterrows()]
    train_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)  for e in train_data.iterrows()]
    test_examples=[InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)    for e in test_data.iterrows()]
    test_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)   for e in test_data.iterrows()]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=batch_size)
    train_evaluator = LabelAccuracyEvaluator(train_dataloader, softmax_model=loss)
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=loss)
    warmup_steps = epochs * len(train_data) / batch_size * 0.1
    model.fit(train_objectives=[(train_dataloader, loss)], epochs=epochs, warmup_steps=665)
    print(model.evaluate(test_evaluator))
    test1_enc=model.encode(test_data.Text1.values)
    test2_enc=model.encode(test_data.Text2.values)
    test_diff_s= np.subtract(test1_enc, test2_enc)
    testIN_s= np.concatenate((test1_enc,test2_enc, test_diff_s), axis=1)
    tr1_enc=model.encode(train_data.Text1.values)
    tr2_enc=model.encode(train_data.Text2.values)
    tr_diff_s= np.subtract(tr1_enc, tr2_enc)
    trainIN_s= np.concatenate((tr1_enc,tr2_enc, tr_diff_s), axis=1)
    test1_sim = sim_embeddings(test_data.Text1.values)
    test2_sim = sim_embeddings(test_data.Text2.values)
    test_diff_sim = np.subtract(test1_sim, test2_sim)
    testIN_sim = np.concatenate((test1_sim, test2_sim, test_diff_sim), axis=1)
    train1_sim = sim_embeddings(train_data.Text1.values)
    train2_sim = sim_embeddings(train_data.Text2.values)
    train_diff_sim = np.subtract(train1_sim, train2_sim)
    trainIN_sim = np.concatenate((train1_sim, train2_sim, train_diff_sim), axis=1)
    testIN = np.concatenate((testIN_s, testIN_sim), axis=1)
    trainIN = np.concatenate((trainIN_s, trainIN_sim), axis=1)
    y_train = tf.keras.utils.to_categorical(train_data.Class, num_classes=args.num_labels)
    y_test = tf.keras.utils.to_categorical(test_data.Class, num_classes=args.num_labels)
    in_emb = keras.Input(shape=(testIN.shape[1],), name="in")
    x = layers.Dense(1500, activation="relu")(in_emb)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(args.num_labels, activation="softmax")(x)
    cls_model = keras.Model(in_emb, output)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    cls_model.compile(optimizer=opt, loss= tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
    cls_model.fit(trainIN ,y_train,  epochs=150, validation_data=(testIN, y_test))
    pred=np.argmax(cls_model.predict(testIN), axis=1)
    real= test_data.Class.values
    print(confusion_matrix(real, pred))
    cf_matrix.append(confusion_matrix(real, pred))
    print(classification_report(real, pred, labels=class_labels, digits = 6))
    full_report.append(classification_report(real, pred, labels=class_labels,output_dict = True, digits = 6))
final_df, final_cf_matrix = print_final_report(full_report,cf_matrix)
print(final_df,final_cf_matrix)