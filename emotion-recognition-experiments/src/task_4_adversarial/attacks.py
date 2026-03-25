import torch
import torch.nn as nn


def _to_channel_tensor(values, reference_tensor):
    tensor = torch.tensor(values, device=reference_tensor.device, dtype=reference_tensor.dtype)
    return tensor.view(1, -1, 1, 1)


def denormalize(images, mean, std):
    mean_t = _to_channel_tensor(mean, images)
    std_t = _to_channel_tensor(std, images)
    return images * std_t + mean_t


def normalize(images, mean, std):
    mean_t = _to_channel_tensor(mean, images)
    std_t = _to_channel_tensor(std, images)
    return (images - mean_t) / std_t


def fgsm_attack(model, images, labels, eps=8 / 255, mean=None, std=None):
    pixel_images = denormalize(images, mean, std) if mean is not None and std is not None else images
    attack_images = pixel_images.clone().detach().requires_grad_(True)
    labels = labels.clone().detach()

    model_inputs = normalize(attack_images, mean, std) if mean is not None and std is not None else attack_images
    outputs = model(model_inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = attack_images + eps * attack_images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return normalize(adv_images.detach(), mean, std) if mean is not None and std is not None else adv_images.detach()


def pgd_attack(model, images, labels, eps=8 / 255, alpha=1 / 255, steps=10, mean=None, std=None):
    pixel_images = denormalize(images, mean, std) if mean is not None and std is not None else images
    ori_images = pixel_images.clone().detach()
    adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        model_inputs = normalize(adv_images, mean, std) if mean is not None and std is not None else adv_images
        outputs = model(model_inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return normalize(adv_images, mean, std) if mean is not None and std is not None else adv_images
