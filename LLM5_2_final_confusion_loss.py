
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 시 텍스트 파일 로드
file_path = os.path.join("data", "poetry.txt")
if not os.path.exists(file_path):
    print(f"Warning: {file_path} not found!")
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. 텍스트 정리
lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 0]
all_text = "\n".join(lines)

# 3. 문자 단위 토큰화
chars = sorted(list(set(all_text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

vocab_size = len(chars)
seq_length = 20
embedding_dim = 128
hidden_dim = 256
data = [char2idx[ch] for ch in all_text]

if len(data) < seq_length + 1:
    print("⚠️ 데이터가 너무 적습니다.")
    exit()

# 4. Dataset 정의
class PoemDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length]),
            torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        )

# 5. DataLoader 생성
dataset = PoemDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 6. 모델 정의
class LSTMPoet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# 7. 모델, 손실함수, 옵티마이저 초기화
model = LSTMPoet(vocab_size, embedding_dim, hidden_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. 학습 함수 정의 (정확도, F1-score, 손실 그래프, Confusion Matrix 포함)
def train(model, dataloader, epochs=50, save_path="poet_model.pt"):
    model.train()
    best_loss = float("inf")
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_targets = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = output.argmax(dim=-1).view(-1).detach().cpu().numpy()
            targets = y.view(-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at Epoch {epoch+1} with loss {avg_loss:.4f}")

    # 🔍 손실 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, marker='o')
    plt.title("에포크별 손실")
    plt.xlabel("에포크")
    plt.ylabel("손실")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔍 Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(vocab_size)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=[idx2char[i] for i in range(vocab_size)],
                yticklabels=[idx2char[i] for i in range(vocab_size)])
    plt.title("Confusion Matrix (문자 예측)")
    plt.xlabel("예측 문자")
    plt.ylabel("실제 문자")
    plt.tight_layout()
    plt.show()

# 9. 시 생성 함수
def generate_poem(model, start_text="달", length=80, temperature=0.8):
    model.eval()
    max_seed_length = 30
    valid_chars = [ch for ch in start_text if ch in char2idx]
    if not valid_chars:
        return "⚠️ 입력된 문자들이 학습된 문자 집합에 없습니다."
    if len(valid_chars) > max_seed_length:
        valid_chars = valid_chars[-max_seed_length:]
    input_seq = torch.tensor([char2idx[ch] for ch in valid_chars]).unsqueeze(0).to(device)
    generated = "".join(valid_chars)
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            last_char_logits = output[0, -1]
            prob = torch.softmax(last_char_logits, dim=0).cpu().numpy()
            next_idx = np.random.choice(len(prob), p=prob)
            next_char = idx2char[next_idx]
            generated += next_char
            input_seq = torch.tensor([[next_idx]]).to(device)
    return generated

# 10. 실행 블록
if __name__ == "__main__":
    model_path = "poet_model.pt"
    if not os.path.exists(model_path):
        print("Training model...")
        train(model, dataloader, epochs=100, save_path=model_path)
    else:
        print("Loading trained model...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    while True:
        user_input = input("시작 문장을 입력하세요 (종료: exit): ")
        if user_input.lower() == "exit":
            break
        poem = generate_poem(model, start_text=user_input, temperature=0.8)
        print(f"\n[temperature=0.8] 생성된 시:\n" + poem + "\n")
