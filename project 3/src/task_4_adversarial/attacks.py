import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, eps=8/255):
    images = images.clone().detach().requires_grad_(True)
    labels = labels.clone().detach()

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = images + eps * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


def pgd_attack(model, images, labels, eps=8/255, alpha=1/255, steps=10):
    ori_images = images.clone().detach()
    adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()

        # projection
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images