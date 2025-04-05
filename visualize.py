import matplotlib.pyplot as plt
import pickle
import os
def plot_metrics(train_loss, val_loss, val_acc):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig("metrics.png")
    plt.show()

# def visualize_weights():
#     with open("best_model.pkl", "rb") as f:
#         params = pickle.load(f)
#     W1 = params['W1']
#     fig, axes = plt.subplots(4, 4, figsize=(8, 8))
#     for i, ax in enumerate(axes.flat):
#         img = W1[:, i].reshape(32, 32, 3)
#         img = (img - img.min()) / (img.max() - img.min())
#         ax.imshow(img)
#         ax.axis('off')
#     plt.suptitle("First Layer Weights")
#     plt.savefig("weights.png")

def visualize_first_layer(W1, save_path='figs/weights_input_hidden.png'):
    """
    可视化输入层到隐藏层的权重（W1 shape: [3072, hidden_dim]）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    hidden_dim = W1.shape[1]
    num_visual = min(16, hidden_dim)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_visual):
        weight = W1[:, i]
        img = weight.reshape(3, 32, 32).transpose(1, 2, 0)  # (32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min())
        axes.flat[i].imshow(img)
        axes.flat[i].axis('off')
    for j in range(num_visual, 16):
        axes.flat[j].axis('off')
    plt.suptitle("First Layer Weights (Input -> Hidden)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_second_layer(W2, save_path='figs/weights_hidden_output.png'):
    """
    可视化隐藏层到输出层的权重（W2 shape: [hidden_dim, 10]）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    for i in range(W2.shape[1]):
        plt.plot(W2[:, i], label=f'Class {i}')
    plt.title("Second Layer Weights (Hidden -> Output)")
    plt.xlabel("Hidden Neuron Index")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_output_distribution(X, params, forward_propagation_fn, save_path='figs/output_distribution.png'):
    """
    可视化模型输出层的 softmax 分布（前100个样本）
    """
    _, _, _, A2 = forward_propagation_fn(X[:100], params)
    plt.figure(figsize=(12, 6))
    for i in range(A2.shape[1]):
        plt.plot(A2[:, i], label=f'Class {i}')
    plt.title("Softmax Output Distribution (First 100 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
