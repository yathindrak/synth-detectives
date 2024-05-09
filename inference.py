lr_checkpoint_path = 'lightning_logs/version_4/checkpoints/model=lr--dev=False.ckpt'
# Update the following checkpoints in the following order: albert, electra, roberta, xlnet
checkpoints = [
    'lightning_logs/version_0/checkpoints/model=albert--dev=False--epoch=89-step=10170--val_loss=0.35.ckpt',
    'lightning_logs/version_1/checkpoints/model=electra--dev=False--epoch=297-step=33674--val_loss=0.39.ckpt',
    'lightning_logs/version_2/checkpoints/model=roberta--dev=False--epoch=299-step=33900--val_loss=0.36.ckpt',
    'lightning_logs/version_3/checkpoints/model=xlnet--dev=False--epoch=85-step=9718--val_loss=0.38.ckpt'
]

import torch
from helper import load_dataset
from model import TransformerModel, SoftMaxLit

DEV = False
# device = torch.cuda.current_device()
device = 'mps'
df = load_dataset('./dataset/training.json', test=True)

validation_df = load_dataset('./dataset/test_new.json', test=True)
model_names = ['albert', 'electra', 'roberta', 'xlnet'] #albert: 128, electra: 64, roberta: 128, xlnet: 128

# print(validation_df.head())
model_y_arr = []
for model_name, ckpt in zip(model_names, checkpoints):
    n_inputs = TransformerModel.MODELS[model_name]['dim']
    model = SoftMaxLit(n_inputs, 2).load_from_checkpoint(n_inputs=n_inputs, n_outputs=2, checkpoint_path=ckpt)
    print(f'Loaded model {model_name} from checkpoint {ckpt}')
    x = TransformerModel(model_name).dataset(validation_df, DEV, save=False, delete=False).x.to(device)
    y_hat = model(x)

    # Free up memory
    del x
    torch.cuda.empty_cache()
    y_first = y_hat

    model_y_arr.append(y_first)
lr_dataset_x = torch.cat(model_y_arr, dim=1).detach()
x = lr_dataset_x.to(device)

lr_model = SoftMaxLit(lr_dataset_x.shape[1], 2).load_from_checkpoint(n_inputs=lr_dataset_x.shape[1], n_outputs=2, checkpoint_path=lr_checkpoint_path).to(device)

validation_out = lr_model(x)
validation_out = validation_out.detach()
out = torch.argmax(validation_out, dim=1)
f = open('answer.json', 'w')
f.write('')
f.close()

f = open('answer.json', 'a')
for idx, label_out in enumerate(out.tolist()):
    to_write = '{"id": ' + str(idx) + ', "label": ' + str(label_out) + '}\n'
    f.write(to_write)
f.close()
# {"id": 0, "label": 1}