import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pylab as pl
import math

memories_path = "test/2023-01-29 20_Desire_split_249_149_offline/"
model_path = 'training/training_ae_gcn/2023-01-23 20_DESIRE_split/model_ae_epoch_249_2023-01-23 20'

model_ae_gcn = torch.load(model_path)
decoder = model_ae_gcn.decoder
FC_output = model_ae_gcn.FC_output

memory_past = torch.load(memories_path + 'memory_past.pt')
memory_fut = torch.load(memories_path + 'memory_fut.pt')

zero_padding = torch.zeros(1, 1, 48 * 3)
future_len = 40
prediction = torch.Tensor()

for i_track in range(len(memory_fut)):
    present = torch.tensor([0,0])
    prediction_single = torch.Tensor()  # .cuda()
    info_future = memory_fut[i_track]
    info_past = memory_past[i_track]
    info_total = torch.cat((info_past.unsqueeze(0).unsqueeze(0), info_future.unsqueeze(0).unsqueeze(0)), 2)
    input_dec = info_total
    state_dec = zero_padding
    for i in range(future_len):
        output_decoder, state_dec = decoder(input_dec, state_dec)
        displacement_next = FC_output(output_decoder)
        coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
        prediction_single = torch.cat((prediction_single, coords_next), 1)
        present = coords_next
        input_dec = zero_padding
    prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)

pred = prediction.squeeze(0)

# plot decoding future trajectory
for t in range(len(memory_fut)):
    x = pred[t][:, 0].detach().numpy().tolist()
    y = pred[t][:, 1].detach().numpy().tolist()
    plt.plot(y, x, marker='o', markersize=1)
plt.show()

X_tsne = TSNE(n_components=2,random_state=33).fit_transform(memory_past)
past_tsne = TSNE(n_components=2,random_state=33).fit_transform(memory_past[:,:48])
interact_tsne = TSNE(n_components=2,random_state=33).fit_transform(memory_past[:,48:])
Y_tsne = TSNE(n_components=2,random_state=33).fit_transform(memory_fut)

mem = torch.cat([memory_past, memory_fut], dim=1)
mem_tsne = TSNE(n_components=2,random_state=33).fit_transform(mem)

# plot encoding cluster
plt.figure()
plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1],s=2, c='r')
plt.show()

# plot trajectory start with encoding 2D point
viridis = pl.cm.get_cmap('BuPu', 8)
plt.figure()
for t in range(len(memory_fut)):
    x = pred[t][:, 0].detach().numpy().tolist()
    x.insert(0, 0)
    y = pred[t][:, 1].detach().numpy().tolist()
    y.insert(0, 0)
    dist = list(pred[t][-1].detach().numpy() - pred[t][0].detach().numpy())
    speed = math.hypot(dist[0], dist[1])/4
    plt.plot(x + mem_tsne[:, 0][t], y + mem_tsne[:, 1][t], color=viridis(1-(15-speed)/15), linewidth=0.8)
plt.scatter(mem_tsne[:, 0], mem_tsne[:, 1],s=2, c='r')
plt.show()

