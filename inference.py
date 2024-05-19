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
from sklearn.metrics import confusion_matrix, classification_report

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

# Get model predictions
validation_out = lr_model(x)
validation_out = validation_out.detach()
out = torch.argmax(validation_out, dim=1)

# Write predictions to JSON file
with open('answer.json', 'w') as f:
    for idx, label_out in enumerate(out.tolist()):
        to_write = '{"id": ' + str(idx) + ', "label": ' + str(label_out) + '}\n'
        f.write(to_write)

# Calculate confusion matrix
true_labels = validation_df['label'].tolist()  # Replace 'label' with the actual column name in your dataset
predicted_labels = out.tolist()
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1'], output_dict=True)
print("Classification Report:")
for key, value in report.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

# Extract and print additional metrics
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")