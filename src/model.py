# Written by Harvey Coombs, 2023-2024
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

json_data = json.load(open("messages.json"))

class Model(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)
        output = self.fc(output)

        return output

class MaskedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["content"]
        tokens = self.tokenizer(text, max_length=self.max_length)

        masked_indices = random.sample(range(len(tokens)), int(0.15 * len(tokens)))
        masked_tokens = tokens.copy()

        for index in masked_indices:
            masked_tokens[index] = 0

        return torch.tensor(tokens), torch.tensor(masked_tokens)

def build_vocab_mapping(tokenized_texts):
    unique_tokens = set(token for tokens in tokenized_texts for token in tokens)
    token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
    return token_to_id

def vlm_tokenizer(text, max_length, vocab_mapping):
    tokens = text.split()
    numerical_tokens = [vocab_mapping.get(token, 0) for token in tokens]

    padded_tokens = numerical_tokens[:max_length] + [0] * (max_length - len(numerical_tokens))

    return padded_tokens

tokenized_texts = [item["content"].split() for item in json_data]
built_vocab_mapping = build_vocab_mapping(tokenized_texts)

tokenized_texts = [vlm_tokenizer(item["content"], 10, built_vocab_mapping) for item in json_data]

def tokenizer(text, max_length):
    return vlm_tokenizer(text, max_length, built_vocab_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(vocab_size=len(built_vocab_mapping) + 1).to(device)

mlm_dataset = MaskedDataset(json_data, tokenizer, max_length=10)
mlm_dataloader = DataLoader(mlm_dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
x = 0

for epoch in range(num_epochs):
    for batch in mlm_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(torch.long).to(device), targets.to(torch.long).to(device)

        targets = targets.unsqueeze(-1).expand(-1, -1, inputs.size(-1))

        outputs = model(inputs, targets)
        outputs = outputs[:, :-1, :].contiguous().view(-1, len(built_vocab_mapping) + 1)
        targets = targets[:, 1:].contiguous().view(-1)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        print(f"({x}): Epoch = {epoch + 1}, Batch loss = {loss.item()}")
        x += 1

torch.save(model.state_dict(), "model.pth")
