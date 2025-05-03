# coding:utf-8
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from keras.models import load_model
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers import losses
from tensorflow import keras
from keras import layers
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import classification_report,  confusion_matrix
from transformers import TFGPT2Model, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer

class CompleteDynamicLoss(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=1.0,
                 gamma_base=2.0,
                 eta=1.0,
                 lambda_conf=0.1,
                 lambda_domain=0.1,
                 domain_target=None,
                 accuracy_val=0.0,
                 from_logits=False,
                 name="CompleteDynamicLoss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma_base = gamma_base
        self.eta = eta
        self.lambda_conf = lambda_conf
        self.lambda_domain = lambda_domain
        self.domain_target = domain_target
        self.accuracy_val = accuracy_val
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        gamma_dynamic = self.gamma_base + self.eta * self.accuracy_val
        focal_term = -self.alpha * tf.pow(1.0 - y_pred, gamma_dynamic) * y_true * tf.math.log(y_pred)
        focal_loss = tf.reduce_sum(focal_term, axis=-1)
        conf_penalty = tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
        if self.domain_target is not None:
            p_avg = tf.reduce_mean(y_pred, axis=0)
            p_avg = tf.clip_by_value(p_avg, epsilon, 1.0)
            q = tf.convert_to_tensor(self.domain_target, dtype=y_pred.dtype)
            q = tf.clip_by_value(q, epsilon, 1.0)
            domain_loss = tf.reduce_sum(q * tf.math.log(q / p_avg))
        else:
            domain_loss = 0.0
        total_loss = focal_loss + self.lambda_conf * conf_penalty + self.lambda_domain * domain_loss
        return total_loss

    def update_accuracy(self, new_accuracy):
        self.accuracy_val = new_accuracy

def print_final_report(full_report, cf_matrix):
    if args.num_labels == 3 :
        class_label = ['0', '1', '2', 'macro avg', 'weighted avg']
    else:
        class_label = ['0','1','macro avg','weighted avg']
    df_final = pd.DataFrame(columns=['class_label', 'Precision', 'Recall', 'F1-score','support'],index = range(0,4))
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
        row = [i,np.mean(p),np.mean(r),np.mean(f),np.mean(s)]
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
        embeddings = outputs.pooler_output
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)

def train_sbert_model(train_data, test_data):
    model = SentenceTransformer(args.sbert_model_path)
    loss = losses.SoftmaxLoss(model=model,
                              sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                              num_labels=args.num_labels)
    train_examples = [InputExample(texts=[e[1].Text1, e[1].Text2], label=e[1].Class) for e in train_data.iterrows()]
    train_examples += [InputExample(texts=[e[1].Text2, e[1].Text1], label=e[1].Class) for e in train_data.iterrows()]
    test_examples = [InputExample(texts=[e[1].Text1, e[1].Text2], label=e[1].Class) for e in test_data.iterrows()]
    test_examples += [InputExample(texts=[e[1].Text2, e[1].Text1], label=e[1].Class) for e in test_data.iterrows()]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=16)
    train_evaluator = LabelAccuracyEvaluator(train_dataloader, softmax_model=loss)
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=loss)
    model.fit(train_objectives=[(train_dataloader, loss)], epochs=1, warmup_steps=665)
    print(model.evaluate(test_evaluator))
    return model

def prepare_embeddings(data):
    data1_s = model.encode(data.Text1.values)
    data2_s = model.encode(data.Text2.values)
    data_diff_s = np.subtract(data1_s, data2_s)
    dataIN_s = np.concatenate((data1_s, data2_s, data_diff_s), axis=1)

    data1_gpt = sim_embeddings(data.Text1.values)
    data2_gpt = sim_embeddings(data.Text2.values)
    data_diff_gpt = np.subtract(data1_gpt, data2_gpt)
    dataIN_gpt = np.concatenate((data1_gpt, data2_gpt, data_diff_gpt), axis=1)

    dataIN = np.concatenate((dataIN_s, dataIN_gpt), axis=1)
    return dataIN

def get_combinations(args):
    """Create combinations of dataset for training"""
    if args.combination == '1':
        source_df = pd.read_csv(args.source_path[0])
    elif args.combination == '2':
        data_1,data_2 = args.source_path
        source_1 = pd.read_csv(data_1)
        source_2 = pd.read_csv(data_2)
        source_df = pd.concat([source_1,source_2])
    elif args.combination == '3':
        data_1,data_2,data_3 = args.source_path
        source_1 = pd.read_csv(data_1)
        source_2 = pd.read_csv(data_2)
        source_3 = pd.read_csv(data_3)
        source_df = pd.concat([source_1,source_2,source_3])
    elif args.combination == '4':
        data_1,data_2,data_3,data_4 = args.source_path
        source_1 = pd.read_csv(data_1)
        source_2 = pd.read_csv(data_2)
        source_3 = pd.read_csv(data_3)
        source_4 = pd.read_csv(data_4)
        source_df = pd.concat([source_1,source_2,source_3,source_4])
    else:
        raise ValueError("Invalid combination type. Expected 'single', 'double', or 'triple'.")
    print(source_df.Class.value_counts())
    df, class_labels = Data_preprocessing(source_df,)
    return df, class_labels

def Data_preprocessing(df):
    if 'Class' not in df.columns:
        raise ValueError("DataFrame does not contain 'Class' column.")
    if args.num_labels == 2:
        df['Class'] = df['Class'].replace(['Conflict', 'Neutral'], [1, 0])
        class_labels = [0, 1]
        print(df.Class.value_counts())
    if args.num_labels == 3:
        df['Class'] = df['Class'].replace(['Conflict', 'Neutral', 'Duplicate'], [1, 0, 2])
        class_labels = [0, 1, 2]
        print(df.Class.value_counts())
    return df, class_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', type=str, required= True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--source_name', type=str, nargs='+', required = True)
    parser.add_argument('--source_path', type=str, nargs='+', required=True)
    parser.add_argument('--combination', type=str, choices=['1', '2', '3','4'], required=True)
    parser.add_argument('--sbert_model_path', type=str)
    parser.add_argument('--sim_model_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    df_source, source_class_labels = get_combinations(args)
    df_target, target_class_labels = Data_preprocessing(pd.read_csv(args.target_path))
    target_data = df_target.iloc[df_target.index]
    source_data = df_source.iloc[df_source.index]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps=10000, decay_rate=0.95, staircase=True)
    sim_model = AutoModel.from_pretrained(args.sim_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.sim_model_path)
    full_report = []
    cf_matrix = []
    skf = StratifiedKFold(n_splits=args.n_splits, random_state = 1,shuffle = True)
    t = df_target.Class

    for test_index, train_index in skf.split(np.zeros(len(t)), t):
        test_data = df_target.iloc[test_index]
        train_data_target = df_target.iloc[train_index]
        train_data = pd.concat([train_data_target, source_data], axis=0)
        model = train_sbert_model(train_data, test_data)
        trainIN = prepare_embeddings(train_data)
        testIN = prepare_embeddings(test_data)
        targetIN = prepare_embeddings(target_data)
        sourceIN = prepare_embeddings(source_data)
        y_train = tf.keras.utils.to_categorical(train_data.Class, num_classes=args.num_labels)
        y_test = tf.keras.utils.to_categorical(test_data.Class, num_classes=args.num_labels)
        y_source = tf.keras.utils.to_categorical(source_data.Class, num_classes=args.num_labels)
        custom_loss = CompleteDynamicLoss(
            alpha=1.0,
            gamma_base=2.0,
            eta=1.0,
            lambda_conf=0.1,
            lambda_domain=0.1,
            domain_target=[0.3, 0.4, 0.3],
            accuracy_val=0.0,
            from_logits=False
        )
        in_emb = keras.Input(shape=(trainIN.shape[1],), name="in")
        x = layers.Dense(1500, activation="relu")(in_emb)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(args.num_labels, activation="softmax")(x)
        cls_model = keras.Model(in_emb, output)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        cls_model.compile(optimizer=opt, loss= custom_loss, metrics=["accuracy"])
        cls_model.fit(trainIN ,y_train, epochs=10, validation_data=(testIN, y_test))
        pred = np.argmax(cls_model.predict(targetIN), axis=1)
        real= target_data.Class.values
        print(confusion_matrix(real, pred))
        cf_matrix.append(confusion_matrix(real, pred))
        print(classification_report(real, pred, labels=target_class_labels, digits=6))
        full_report.append(classification_report(real, pred, labels=target_class_labels, output_dict=True, digits=6))
        with open(args.output_path, "a") as f:
            f.write(classification_report(real, pred, labels=target_class_labels, digits=6))
    final_df, final_cf_matrix = print_final_report(full_report,cf_matrix)
    print(final_df,final_cf_matrix)
