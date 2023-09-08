import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SarcasmDataModule(pl.LightningDataModule):
    def __init__(self, data_file, tokenizer, batch_size=32, max_length=128):
        super().__init__()
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        df = pd.read_json(self.data_file, lines=True)
        # Take only 10% of the dataset for quick testing
        #df = df.sample(frac=0.1)
        train_val, test = train_test_split(df, test_size=0.1)
        train, val = train_test_split(train_val, test_size=0.1)
        self.train_dataset = SarcasmDataset(train['headline'].to_list(), train['is_sarcastic'].to_list(), self.tokenizer, self.max_length)
        self.val_dataset = SarcasmDataset(val['headline'].to_list(), val['is_sarcastic'].to_list(), self.tokenizer, self.max_length)
        self.test_dataset = SarcasmDataset(test['headline'].to_list(), test['is_sarcastic'].to_list(), self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class SarcasmClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accuracy = Accuracy(num_classes=2, average='macro', task="binary")
        self.f1 = F1Score(num_classes=2, average='macro', task="binary")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self.model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.accuracy(preds, labels)
        f1_score = self.f1(preds, labels)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_f1', f1_score)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self.model(input_ids, attention_mask, labels=labels)
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.accuracy(preds, labels)
        f1_score = self.f1(preds, labels)
        #self.log('val_loss', val_loss)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        #self.log('val_acc', acc)
        self.log('val_acc', acc, on_step=True, on_epoch=False)
        #self.log('val_f1', f1_score)
        self.log('val_f1', f1_score, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self.model(input_ids, attention_mask, labels=labels)
        test_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.accuracy(preds, labels)
        f1_score = self.f1(preds, labels)
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_f1', f1_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    #data_module = SarcasmDataModule('../../data/headlines/headlines.csv', tokenizer)
    data_module = SarcasmDataModule('../../data/headlines/Sarcasm_Headlines_Dataset.json', tokenizer)
    classifier = SarcasmClassifier(model)
    #wandb_logger = pl.loggers.WandbLogger(project='sarcasm-detection')
    #trainer = pl.Trainer(max_epochs=1, accelerator='auto', logger=wandb_logger)
    trainer = pl.Trainer(max_epochs=2, accelerator='auto')
    trainer.fit(classifier, data_module)
    trainer.test(datamodule=data_module)

    checkpoint = {'model': classifier.state_dict(), 'optimizer': classifier.configure_optimizers().state_dict()}
    torch.save(checkpoint, '../../checkpoints/textmodal.ckpt')

if __name__ == "__main__":
    main()