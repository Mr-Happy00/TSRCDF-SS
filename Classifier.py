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
from tensorflow.keras import layers
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import classification_report,  confusion_matrix

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

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--sbert_model_path', type=str)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--n_splits', type=int, default=5)
args = parser.parse_args()

df = pd.read_csv(args.csv_path)
if args.data_name in ['open','uav','wv','pure', 'cn']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral'],[1,0])
    class_labels = [0,1]
    print(df.Class.value_counts())
elif args.data_name in ['cdn','NLI_dev','NLI_train','NLI_test','NLI_70000','NLI_50000','NLI_30000','NLI_20000']:
    df['Class'] = df['Class'].replace(['Conflict','Neutral','Duplicate'],[1,0,2])
    class_labels = [0,1,2]
    print(df.Class.value_counts())
else:
    print("wrong dataset name \n")

model=SentenceTransformer(args.sbert_model_path)
loss = losses.SoftmaxLoss(model=model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=args.num_labels)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps=10000, decay_rate=0.95, staircase=True)
full_report = []
cf_matrix = []
skf = StratifiedKFold(n_splits=args.n_splits, random_state = 1,shuffle = True)
t = df.Class
for train_index, test_index in skf.split(np.zeros(len(t)), t):
    train_data = df.loc[train_index]
    test_data = df.loc[test_index]
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    train_examples= [InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)  for e in train_data.iterrows()]
    train_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)  for e in train_data.iterrows()]
    test_examples=[InputExample(texts=[e[1].Text1,e[1].Text2], label= e[1].Class)    for e in test_data.iterrows()]
    test_examples+=[InputExample(texts=[e[1].Text2,e[1].Text1], label= e[1].Class)   for e in test_data.iterrows()]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=16)
    train_evaluator = LabelAccuracyEvaluator(train_dataloader, softmax_model=loss)
    test_evaluator = LabelAccuracyEvaluator(test_dataloader, softmax_model=loss)
    model.fit(train_objectives=[(train_dataloader, loss)], epochs=10, warmup_steps=665)
    test1_enc=model.encode(test_data.Text1.values)
    test2_enc=model.encode(test_data.Text2.values)
    test_diff= np.subtract(test1_enc, test2_enc)
    testIN= np.concatenate((test1_enc,test2_enc, test_diff), axis=1)
    tr1_enc=model.encode(train_data.Text1.values)
    tr2_enc=model.encode(train_data.Text2.values)
    tr_diff= np.subtract(tr1_enc, tr2_enc)
    trainIN= np.concatenate((tr1_enc,tr2_enc, tr_diff), axis=1)
    y_train = keras.utils.to_categorical(train_data.Class ,num_classes = args.num_labels)
    y_test = keras.utils.to_categorical(test_data.Class,num_classes = args.num_labels)
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
    cls_model.compile(optimizer=opt, loss=custom_loss, metrics=["accuracy"])
    cls_model.fit(trainIN, y_train, epochs=10, validation_data=(testIN, y_test))
    pred=np.argmax(cls_model.predict(testIN), axis=1)
    real= test_data.Class.values
    print(confusion_matrix(real, pred))
    cf_matrix.append(confusion_matrix(real, pred))
    print(classification_report(real, pred, labels=class_labels, digits = 6))
    full_report.append(classification_report(real, pred, labels=class_labels,output_dict = True, digits = 6))
final_df, final_cf_matrix = print_final_report(full_report,cf_matrix)
print(final_df,final_cf_matrix)