import torch
import os
from safetensors.torch import load_file, save_file
from safetensors.torch import safe_open

devi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m_con = ".\Cleaned_counterfeit\minifigure_cleaned_con-000010.safetensors"
m_chill = ".\Cleaned_chill\minifigure_cleaned_chill-000010.safetensors"
m_v15 = ".\Cleaned_v15\minifigure_cleaned_v15-000010.safetensors"
m_8 = ".\Cleaned_counterfeit_8\minifigure_cleaned_con_small-000010.safetensors"
m_16 = ".\Cleaned_counterfeit_16\minifigure_cleaned_con_16-000010.safetensors"

con = 'D:/Fun/Webui/stable-diffusion-webui/models/Stable-diffusion/CounterfeitV25_25.safetensors'
chill = 'D:/Fun/Webui/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors'
v15 = 'D:/Fun/Webui/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors'

s1 = 'lora_te_text_model_encoder_layers_0_mlp_fc1'
LoRA_dim = '.alpha'
m_type = ['.lora_down.weight', '.lora_up.weight']
m_name = ['con', 'chill', 'v15']

LoRA_con = {}
with safe_open(m_con, framework='pt', device=0) as f:
    for k in f.keys():
        LoRA_con[k] = f.get_tensor(k).squeeze().type('torch.FloatTensor')

LoRA_chill = {}
with safe_open(m_chill, framework='pt', device=0) as f:
    for k in f.keys():
        LoRA_chill[k] = f.get_tensor(k).squeeze().type('torch.FloatTensor')

LoRA_v15 = {}
with safe_open(m_v15, framework='pt', device=0) as f:
    for k in f.keys():
        LoRA_v15[k] = f.get_tensor(k).squeeze().type('torch.FloatTensor')

LoRA_8 = {}
with safe_open(m_8, framework='pt', device=0) as f:
    for k in f.keys():
        LoRA_8[k] = f.get_tensor(k).squeeze().type('torch.FloatTensor')

LoRA_16 = {}
with safe_open(m_16, framework='pt', device=0) as f:
    for k in f.keys():
        LoRA_16[k] = f.get_tensor(k).squeeze().type('torch.FloatTensor')

S = []
for key in list(LoRA_con.keys()):
    key = key.split('.')
    if 'attn' not in key[0]:
        continue
    if key[0] not in S:
        # print(key[0])
        S.append(key[0])

def sub_space_sim(A, B, k=8):
    i_s = k-1
    j_s = k-1
    i_r = i_s + 1
    j_r = j_s + 1
    minimum = 1.0
    maximum = 0.0
    tracker = []
    for i in range(i_s, i_r):
        t_tmp = []
        for j in range(j_s, j_r):
            i_t = i+1
            j_t = j+1
            p = min(i_t, j_t)
            Prod = torch.matmul(A[:, :i_t].t(), B[:, :j_t])
            _, S, _ = torch.svd(Prod, compute_uv=False)

            if len(S.shape) == 2:
                S = torch.diagonal(S, 0)
            S = torch.square(S).sum()
            v = S/p

            t_tmp.append(v)
            minimum = min(minimum, v)
            maximum = max(maximum, v)
        tracker.append(t_tmp)

    return tracker, minimum, maximum

m1_track = []
m2_track = []

for s in S:

    A_con = LoRA_con[s+m_type[0]]
    A_chill = LoRA_chill[s+m_type[0]]
    A_v15 = LoRA_v15[s+m_type[0]]

    U_con, S_con, V_con = torch.svd(A_con)
    U_chill, S_chill, V_chill = torch.svd(A_chill)
    U_v15, S_v15, V_v15 = torch.svd(A_v15)

    d_1, mini1, maxi1 = sub_space_sim(U_con, U_v15)
    d_2, mini2, maxi2 = sub_space_sim(U_chill, U_v15)

    m1_track.append([maxi1, mini1, s, d_1])
    m2_track.append([maxi2, mini2, s, d_2])

for p in range(4):
    k = 2**(p+1)
    m3_track = []
    m4_track = []
    print(len(LoRA_16.keys()))
    for s in S:

        A_con = LoRA_con[s+m_type[1]]
        A_16 = LoRA_16[s+m_type[1]]
        A_8 = LoRA_8[s+m_type[1]]

        U_con, S_con, V_con = torch.svd(A_con)
        U_16, S_16, V_16 = torch.svd(A_16)
        U_8, S_8, V_8 = torch.svd(A_8)

        d_1, mini1, maxi1 = sub_space_sim(U_8, U_con, k=k)
        d_2, mini2, maxi2 = sub_space_sim(U_16, U_con, k=k)

        m3_track.append([maxi1, mini1, s, d_1])
        m4_track.append([maxi2, mini2, s, d_2])

    d3_unet = [e[0].item() for e in m3_track if 'unet' in e[2]]
    d3_text = [e[0].item() for e in m3_track if 'text' in e[2]]
    d4_unet = [e[0].item() for e in m4_track if 'unet' in e[2]]
    d4_text = [e[0].item() for e in m4_track if 'text' in e[2]]

    print("k = ", k)

    print("d3_unet: ", sum(d3_unet)/len(d3_unet), d3_unet[len(d3_unet)//2])
    print("d3_text: ", sum(d3_text)/len(d3_text), d3_text[len(d3_text)//2])
    print("d4_unet: ", sum(d4_unet)/len(d4_unet), d4_unet[len(d4_unet)//2])
    print("d4_text: ", sum(d4_text)/len(d4_text), d4_text[len(d4_text)//2])
# # sort by norm
# display_r = 20
# m1_track.sort(key=lambda x: x[0])
# for i in range(display_r):
#     print(m1_track[i][0], m1_track[i][1], m1_track[i][2])
# print("========================================")
# m2_track.sort(key=lambda x: x[0])
# for i in range(display_r):
#     print(m2_track[i][0], m2_track[i][1], m2_track[i][2])
# print("++++++++++++++++++++++++++++++++++++++++")
# # # sort by minimum
# m1_track.sort(key=lambda x: x[1], reverse=True)
# for i in range(display_r):
#     print(m1_track[i][0], m1_track[i][1], m1_track[i][2])
# print("========================================")
# m2_track.sort(key=lambda x: x[1], reverse=True)
# for i in range(display_r):
#     print(m2_track[i][0], m2_track[i][1], m2_track[i][2])

#plot histogram of m1_track and m2_track
# import matplotlib.pyplot as plt

# d1_unet = [e[0].item() for e in m1_track if 'unet' in e[2]]
# d1_text = [e[0].item() for e in m1_track if 'text' in e[2]]
# d2_unet = [e[0].item() for e in m2_track if 'unet' in e[2]]
# d2_text = [e[0].item() for e in m2_track if 'text' in e[2]]

# d3_unet = [e[0].item() for e in m3_track if 'unet' in e[2]]
# d3_text = [e[0].item() for e in m3_track if 'text' in e[2]]
# d4_unet = [e[0].item() for e in m4_track if 'unet' in e[2]]
# d4_text = [e[0].item() for e in m4_track if 'text' in e[2]]
# print(len(m1_track))
# extra = [e[2] for e in m1_track if ('unet' not in e[2] and 'text' not in e[2])]
# print(extra)
# print(len(extra))

# plt.hist(d1_unet, bins=20, alpha=0.5, label='unet')
# plt.hist(d1_text, bins=20, alpha=0.5, label='text')
# plt.xlabel("Similarity")
# plt.ylabel("Frequency")
# plt.legend(loc='upper right')
# plt.show()

# plt.hist(d2_unet, bins=20, alpha=0.5, label='unet')
# plt.hist(d2_text, bins=20, alpha=0.5, label='text')
# plt.xlabel("subspace similarity between Chill and V15")
# plt.legend(loc='upper right')
# plt.show()

# plt.hist(d3_unet, bins=20, alpha=0.5, label='unet')
# plt.hist(d3_text, bins=20, alpha=0.5, label='text')
# plt.xlabel("Similarity")
# plt.ylabel("Frequency")
# plt.legend(loc='upper right')
# plt.show()

# plt.hist(d4_unet, bins=20, alpha=0.5, label='unet')
# plt.hist(d4_text, bins=20, alpha=0.5, label='text')
# plt.xlabel("Similarity")
# plt.ylabel("Frequency")
# plt.legend(loc='upper right')
# plt.show()

# print("d1_unet: ", sum(d1_unet)/len(d1_unet), d1_unet[len(d1_unet)//2])
# print("d1_text: ", sum(d1_text)/len(d1_text), d1_text[len(d1_text)//2])
# print("d2_unet: ", sum(d2_unet)/len(d2_unet), d2_unet[len(d2_unet)//2])
# print("d2_text: ", sum(d2_text)/len(d2_text), d2_text[len(d2_text)//2])


# print(len(d1_unet), len(d1_text), len(d2_unet), len(d2_text))
# d1_unet:  0.4724764891434461 0.44465169310569763
# d1_text:  0.9098749620219072 0.9075452089309692
# d2_unet:  0.49458356318064034 0.4718474745750427
# d2_text:  0.9065756176908811 0.9040594100952148