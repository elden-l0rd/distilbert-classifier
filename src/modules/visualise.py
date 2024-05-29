import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# Visualise word frequencies using word cloud
def word_occurrence_by_group(df):
    token_counts_by_group = {}
    grouped_df = df.groupby('STRIDE')
    for stride_value, group_df in grouped_df:
        all_tokens = []
        for tokens in group_df['NameDesc']:
            all_tokens.extend(tokens)
        token_count = Counter(all_tokens)
        token_counts_by_group[stride_value] = token_count
    return token_counts_by_group

def plot_wc(df):
    print("Generating word cloud...")
    output_dir = 'data/results/wordclouds'
    token_counts_by_group = word_occurrence_by_group(df)
    for stride_value, token_count in token_counts_by_group.items():
        wordcloud = WordCloud(background_color="black").generate_from_frequencies(token_count)
        plt.imshow(wordcloud)
        plt.ion()
        plt.show()
        plt.axis("off")
        sv = ['S','E','D','I','R','T']
        plt.title(f"Word Cloud for STRIDE '{stride_value}': {sv[stride_value]}")
        plt.savefig(f"{output_dir}/wordcloud_{stride_value}.png")
        plt.pause(.3)
        plt.close()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    print("Generated all word clouds...")
    return

def plot_data(hist, model, X_val_padded, y_val, classes):
    output_dir = 'data/results'

    # acc = hist.history['accuracy']
    # val_acc = hist.history['val_accuracy']
    # loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    # epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    # plt.subplot(1, 3, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend()
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 3, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')

    y_pred = np.argmax(model.predict(X_val_padded), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, square=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix\nF1 Score: {f1:.4f}', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusionmatrix.png", bbox_inches='tight')
    plt.show()
    plt.pause(2)
    return f1

def plot_cm(cm, model=0, X_val_padded=0, y_val=0, classes=0):
    output_dir = 'data/results'
    if (model or X_val_padded or y_val or classes):
        y_pred = np.argmax(model.predict(X_val_padded), axis=1)
        cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, square=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusionmatrix.png", bbox_inches='tight')
    plt.show()
    plt.pause(2)
    return

def plot_kfold(X, y, models, n_splits=5):
    f1_scores = []
    confusion_matrices = []
    output_dir = 'data/results'
    
    nrows = int(np.ceil(n_splits / 2))
    ncols = 2 if n_splits > 1 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 6))
    axes = axes.flatten() if n_splits > 1 else [axes]
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y)):
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        y_pred = models[fold].predict(X_val_fold)
        y_pred = np.argmax(y_pred, axis=1)
        
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        cm = confusion_matrix(y_val_fold, y_pred)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        
        sns.heatmap(cm, annot=True, fmt='g', ax=axes[fold], cmap='Blues')
        axes[fold].set_title(f'Fold {fold+1} Confusion Matrix\nF1 Score: {f1:.4f}', fontsize=10)
        axes[fold].set_xlabel('Predicted labels', fontsize=8)
        axes[fold].set_ylabel('True labels', fontsize=8)
        axes[fold].tick_params(axis='both', which='major', labelsize=8)
        
        print(f"Fold {fold+1} Confusion Matrix:\n", cm)
        print(f"Fold {fold+1} F1 Score: {f1}\n")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(f"{output_dir}/confusionmatrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    return f1_scores, confusion_matrices
