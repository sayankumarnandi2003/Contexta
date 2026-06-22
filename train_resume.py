import os
import torch
import data_loader
from gpt import GPTLanguageModel, encode, decode, device
import train

def main():
    print("Loading data...")
    data_loader.load_data('input.txt')

    print("Initializing model...")
    model = GPTLanguageModel()
    model = model.to(device)

    if os.path.exists('model.pth'):
        print("Loading existing checkpoint from model.pth...")
        model.load_state_dict(torch.load('model.pth', map_location=device))
    else:
        print("No checkpoint found, starting from scratch...")

    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate)

    max_iters = 1000
    eval_interval = 250
    
    print("Starting resumed training loop...")
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = train.estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(model.state_dict(), 'model.pth')
            
            # Generate a sample
            model.eval()
            context = torch.tensor(encode('The physical features of India '), dtype=torch.long, device=device).unsqueeze(0)
            sample = decode(model.generate(context, max_new_tokens=100, temperature=0.5)[0].tolist())
            print(f"--- Sample at step {iter} ---\n{sample}\n-----------------------")
            model.train()

        xb, yb = data_loader.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("Training finished. Final model saved to model.pth")
    torch.save(model.state_dict(), 'model.pth')

    # Generate final sample
    model.eval()
    context = torch.tensor(encode('The physical features of India '), dtype=torch.long, device=device).unsqueeze(0)
    sample = decode(model.generate(context, max_new_tokens=200, temperature=0.5)[0].tolist())
    print(f"--- Final Sample ---\n{sample}\n-----------------")

if __name__ == '__main__':
    main()