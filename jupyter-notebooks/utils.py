import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_predictions(X_line, y_line, X_train, y_train, X_test, y_test, title=None, name=None):
    plt.figure(figsize=(7, 5))

    default_colors = sns.color_palette()
    seaborn_blue = default_colors[0]
    seaborn_orange = default_colors[1]
    
    sns.scatterplot(x=X_train.flatten(), y=y_train, alpha=0.5, label='Train', color=seaborn_blue)
    sns.scatterplot(x=X_test.flatten(), y=y_test, alpha=0.5, label='Test', color=seaborn_orange)
    
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Prediction')
    
    plt.xlabel('V', fontsize=12)
    plt.ylabel('PE', fontsize=12)
    plt.legend(fontsize=10)
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(name)


def plot_predictions_trunc(X_line, y_line, X_train, y_train, X_test, y_test, title=None, name=None):
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    default_colors = sns.color_palette()
    seaborn_blue = default_colors[0]
    seaborn_orange = default_colors[1]
    
    # Create dataframes for seaborn
    train_df = pd.DataFrame({'V': X_train.flatten(), 'PE': y_train})
    test_df = pd.DataFrame({'V': X_test.flatten(), 'PE': y_test})
    
    sns.scatterplot(data=train_df, x='V', y='PE', ax=axes[0], color=seaborn_blue, alpha=0.5, label='Train')
    sns.scatterplot(data=test_df, x='V', y='PE', ax=axes[0], color=seaborn_orange, alpha=0.5, label='Test')
    axes[0].plot(X_line, y_line, 'r-', linewidth=2, label='Prediction')
    axes[0].set_xlabel('V', fontsize=12)
    axes[0].set_ylabel('PE', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_title(f'{title}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    sns.scatterplot(data=train_df, x='V', y='PE', ax=axes[1], color=seaborn_blue, alpha=0.5, label='Train')
    sns.scatterplot(data=test_df, x='V', y='PE', ax=axes[1], color=seaborn_orange, alpha=0.5, label='Test')
    axes[1].plot(X_line, y_line, 'r-', linewidth=2, label='Prediction')
    
    y_min = min(y_train.min(), y_test.min())
    y_max = max(y_train.max(), y_test.max())
    y_padding = (y_max - y_min) * 0.05
    axes[1].set_ylim(y_min - y_padding, y_max + y_padding)
    
    axes[1].set_xlabel('V', fontsize=12)
    axes[1].set_ylabel('PE', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_title(f'{title}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()