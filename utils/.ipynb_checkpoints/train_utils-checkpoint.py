import torch
import wandb
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_model(cvae, train_loader, optimizer, save_dir,device,num_epochs=10,cond_dim=10):
    cvae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            c = torch.eye(cond_dim)[labels].to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = cvae(data,c)

            # 计算损失并反向传播
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item() / len(data):.4f}')

        Average_loss=train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}, Average loss: {Average_loss:.4f}')
        wandb.log({"epoch_loss": Average_loss},step=epoch)


        if(epoch%200==0):
            save_path=save_dir+f'cvae_{epoch}.pth'
            torch.save(cvae.state_dict(),save_path)
