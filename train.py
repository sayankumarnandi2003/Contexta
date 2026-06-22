import torch
import data_loader
from gpt import GPTLanguageModel

# Hyperparameters
max_iters = 3000
eval_interval = 250
eval_iters = 50
learning_rate = 5e-4
device = data_loader.device

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print("Loading data...")
    # Data is already initialized via import gpt, but just in case
    data_loader.load_data('input.txt')

    print("Initializing model...")
    model = GPTLanguageModel()
    model = model.to(device)

    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting training loop...")
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Explicit checkpointing
            torch.save(model.state_dict(), 'model.pth')

        xb, yb = data_loader.get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("Training finished. Final model saved to model.pth")
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()