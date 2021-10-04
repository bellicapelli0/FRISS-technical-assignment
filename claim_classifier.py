import pandas as pd
from preprocessing import preprocess

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np

import datetime


class ClaimDataset(Dataset):

    def __init__(self, df):
        self.X = df.drop(columns=["sys_claimid", "sys_fraud"]).to_numpy().astype('float32')
        self.y = df["sys_fraud"].to_numpy().astype('int64')
        self.IDs = df["sys_claimid"].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        tensor = torch.tensor(self.X[index,:])
        #     label = torch.LongTensor(self.y[index])
        label = self.y[index]
        ID = self.IDs[index]

        return tensor, label, ID

    def number_of_features(self):
        return self.X.shape[1]


class ClaimClassifier(nn.Module):
    def __init__(self, input_neurons):
        super(ClaimClassifier, self).__init__()

        #Linear
        self.classifier1 = nn.Linear(input_neurons, 64)

        #activation
        self.activation = nn.ReLU()
        
        #dropout
        self.dropout = nn.Dropout(0.1)

        #Linear
        self.classifier2 = nn.Linear(64, 32)
        
        

        #Linear
        self.classifier3 = nn.Linear(32, 2)


    def forward(self, tensor):

        x = self.classifier1(tensor)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier2(x)
        x = self.activation(x)
        logits = self.classifier3(x)

        return logits

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, device, epochs, max_patience):
    loss_values = []

    #Training
    print("- - T R A I N I N G - -")

    model_name = "{}.mdl".format(datetime.datetime.now().replace(microsecond=0).isoformat()).replace("/","_")
    patience = 0
    scores = []

    for epoch_i in range(epochs):

        print()
        print('> Epoch {:} of {:}'.format(epoch_i + 1, epochs))

        t0 = time.time()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):

            if step % 5000 == 0 and not step == 0:
                print('├───Batch {:>3,} of {:>3,}. time={:.1f}s.'.format(step, len(train_dataloader)-1, time.time() - t0))
            if step == len(train_dataloader)-1:
                print('└───Batch {:>3,} of {:>3,}. time={:.1f}s.'.format(step, len(train_dataloader)-1, time.time() - t0))

            x = batch[0].to(device)
            y = torch.LongTensor(batch[1].to(device))
            #ID = batch[2].to(device)

            model.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)
            total_loss += loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO:check

            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)

        print("")
        print("- Training loss: {0:.2f}".format(avg_train_loss))
        print("- Training Epoch took: {:.1f}s".format(time.time() - t0))

        # Validation

        print()
        print("- - V A L I D A T I N G - -")

        t0 = time.time()

        total_loss = 0

        model.eval()

        eval_loss, eval_accuracy, eval_auroc, eval_precision, eval_recall = 0, 0, 0, 0, 0
        nb_acc_steps, nb_auroc_steps, nb_eval_examples, nb_prc_steps, nb_rcl_steps = 0, 0, 0, 0, 0



        softmax = nn.Softmax(dim=1)

        eval_preds = np.array([])
        eval_probs = np.array([])
        eval_labels = np.array([])


        for batch in test_dataloader:
            x = batch[0].to(device)
            y = torch.LongTensor(batch[1].to(device))
            #ID = batch[2].to(device)

            with torch.no_grad():
                logits = model(x)

            loss = criterion(logits, y)

            total_loss += loss.item()

            probs = softmax(logits).detach().cpu().numpy()[:,1]
            logits = logits.detach().cpu().numpy()
            labels = y.to('cpu').numpy()
            preds = np.argmax(logits, axis=1).flatten()

            eval_preds = np.append(eval_preds, preds)
            eval_probs = np.append(eval_probs, probs)
            eval_labels = np.append(eval_labels, labels)


        eval_loss = total_loss / len(test_dataloader)

        print("")
        print("- Validation Accuracy: {0:.4f}".format(accuracy_score(eval_preds, eval_labels)))
        print("- Validation Precision: {0:.4f}".format(precision_score(eval_preds, eval_labels)))
        print("- Validation Recall: {0:.4f}".format(recall_score(eval_preds, eval_labels)))
        print("- Validation F1: {0:.4f}".format(f1_score(eval_preds, eval_labels)))
        print("- Validation AUROC: {0:.4f}".format(roc_auc_score(eval_labels, eval_probs)))
        print("- Validation loss: {0:.4f}".format(eval_loss))
        print("- Validation took: {:.1f}s".format(time.time() - t0))

        f1 = f1_score(eval_preds, eval_labels)
        # Save if new best score has been achieved
        if len(scores)==0:
            print("")
            print("Saving model...")
            torch.save(model.state_dict(), "savedmodels/{}".format(model_name))
            best_preds = eval_preds
            best_labels = eval_labels
        elif f1 > max(scores):
            print("")
            print("New best F1 score, saving...")
            torch.save(model.state_dict(), "savedmodels/{}".format(model_name))
            best_preds = eval_preds
            best_labels = eval_labels
            patience = 0
        else:
            patience += 1


        if patience >= max_patience:
            "Maximum patience reached."
            break

        if precision_score(eval_preds, eval_labels) == 1:
            scores.append(0.0)
        else:
            scores.append(f1)

    


    print("")
    print("Finished Training")
    print("")
    best_epoch = np.argmax(scores)
    print("Best seen F1 {:.4f} at epoch {}.".format(scores[best_epoch], best_epoch+1))

    print("Best model saved with filename: {}".format(model_name))
    print("Loading best model...")
    best_model = ClaimClassifier(train_dataloader.dataset.number_of_features())
    best_model.load_state_dict(torch.load("savedmodels/{}".format(model_name)))
    best_model.eval()
    print("Done.")

    return best_model, best_preds, best_labels #eval_accuracy/nb_eval_steps, eval_auroc/nb_eval_steps, eval_loss, 


def predict_model(model, criterion, optimizer, test_dataloader, device):

  

    model_name = "{}.mdl".format(datetime.datetime.now().replace(microsecond=0).isoformat()).replace("/","_")


    print("- - P R E D I C T I N G - -")

    t0 = time.time()

    total_loss = 0

    model.eval()



    softmax = nn.Softmax(dim=1)

    preds = np.array([])
    probs = np.array([])
    labels = np.array([])
    IDs = np.array([])

    for batch in test_dataloader:
        x = batch[0].to(device)
        y = torch.LongTensor(batch[1].to(device))
        ID = batch[2][0]

        with torch.no_grad():
            logits = model(x)

        loss = criterion(logits, y)

        total_loss += loss.item()

        batch_probs = softmax(logits).detach().cpu().numpy()[:,1]
        logits = logits.detach().cpu().numpy()
        batch_labels = y.to('cpu').numpy()
        batch_preds = np.argmax(logits, axis=1).flatten()

        preds = np.append(preds, batch_preds)
        probs = np.append(probs, batch_probs)
        labels = np.append(labels, batch_labels)
        IDs = np.append(IDs, ID)

    pred_loss = total_loss / len(test_dataloader)

    print("")
    print("- Prediction Accuracy: {0:.4f}".format(accuracy_score(preds, labels)))
    print("- Prediction Precision: {0:.4f}".format(precision_score(preds, labels)))
    print("- Prediction Recall: {0:.4f}".format(recall_score(preds, labels)))
    print("- Prediction F1: {0:.4f}".format(f1_score(preds, labels)))
    print("- Prediction AUROC: {0:.4f}".format(roc_auc_score(labels, probs)))
    print("- Prediction loss: {0:.4f}".format(pred_loss))
    print("- Prediction took: {:.1f}s".format(time.time() - t0))
    


    print("")
    print("Finished Predicting")
    print("")

    return preds, labels, IDs

