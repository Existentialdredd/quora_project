import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from transformers import BertForSequenceClassification, AdamW
from hf_utilities import QuoraBertDataSet, bert_tensor_colate
from data.schemas import ModelState


def train_model(train_config_json, valid_config_json, model_name='bert-base-uncased', n_epochs=1):
    """
    PURPOSE:
    """
    train_data = QuoraBertDataSet(config_json=train_config_json)
    valid_data = QuoraBertDataSet(config_json=valid_config_json)

    train_dataloader = DataLoader(train_data, batch_size=50, collate_fn=bert_tensor_colate, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data), collate_fn=bert_tensor_colate)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
    for param in model.base_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_state = ModelState()

    for epoch in range(n_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            model_output = model(input_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
            loss = functional.binary_cross_entropy_with_logits(model_output.logits, batch['label'])
            loss.backward()
            optimizer.step()
            train_accuracy = torch.eq(torch.argmax(model_output.logits, dim=1),
                                      torch.argmax(batch['label'], dim=1)).float().mean()
            train_loss = loss.data.item()
            train_state.train_loss.append(train_loss)
            train_state.train_acc.append(train_accuracy.item())

        model.eval()
        for valid_batch in valid_dataloader:
            model_output = model(input_ids=valid_batch['token_ids'], attention_mask=valid_batch['attention_mask'])
            loss = functional.binary_cross_entropy_with_logits(model_output.logits, valid_batch['label'])
            valid_accuracy = torch.eq(torch.argmax(model_output.logits, dim=1),
                                      torch.argmax(valid_batch['label'], dim=1)).float().mean()
            valid_loss = loss.data.item()

            train_state.val_loss.append(valid_loss)
            train_state.val_acc.append(valid_accuracy.item())

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_state.train_loss[-1]}")
        print(f"Train Acc: {train_state.train_acc[-1]}")
        print(f"Valid Loss: {train_state.val_loss[-1]}")
        print(f"Valid Acc: {train_state.val_acc[-1]}")

